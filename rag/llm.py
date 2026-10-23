"""One interface for several model providers.

The rest of the code talks to an Anthropic-style client:
    client.messages.create(model=, max_tokens=, system=, messages=, tools=, tool_choice=)
    client.messages.stream(...)  ->  context manager with .text_stream

Claude uses the Anthropic SDK directly. Groq and Gemini expose OpenAI-compatible
endpoints, so `OpenAICompat` translates the same calls (including forced tool
use) into chat-completions requests. Each provider also has a size budget:
Groq's free tier allows ~8,000 tokens a minute, so the app sends it fewer and
smaller excerpts than it sends Claude.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from types import SimpleNamespace as NS


@dataclass(frozen=True)
class Provider:
    name: str
    key_name: str              # secret / environment variable holding the key
    key_url: str
    base_url: str | None       # None = Anthropic SDK
    context_tokens: int        # excerpt budget for one question (estimated tokens)
    batch_tokens: int          # excerpt budget for one extraction / drafting call
    max_tokens: int            # output budget
    free: bool
    data_note: str
    history_turns: int = 6     # earlier chat turns sent with a question
    tool_tokens: int = 4000    # output allowance for structured output (figures, requests), trimmed to fit
    request_tokens: int | None = None   # hard cap per request (input + output), if the provider has one


PROVIDERS = {
    "groq": Provider("Groq", "GROQ_API_KEY", "console.groq.com/keys", "https://api.groq.com/openai/v1",
                     context_tokens=3_000, batch_tokens=2_500, max_tokens=1_200, free=True,
                     data_note="Excerpts are sent to Groq to generate results.",
                     # free tier: 8,000 tokens per minute, counting the requested output allowance
                     history_turns=2, request_tokens=6_500, tool_tokens=3_500),
    "gemini": Provider("Google Gemini", "GEMINI_API_KEY", "aistudio.google.com/apikey",
                       "https://generativelanguage.googleapis.com/v1beta/openai/",
                       context_tokens=25_000, batch_tokens=10_000, max_tokens=3_000, free=True,
                       data_note="Excerpts are sent to Google. On Gemini's free tier, Google may use "
                                 "inputs to improve its models and human reviewers may read them."),
    "anthropic": Provider("Anthropic", "ANTHROPIC_API_KEY", "console.anthropic.com", None,
                          context_tokens=30_000, batch_tokens=15_000, max_tokens=4_000, free=False,
                          data_note="Excerpts are sent to the Anthropic API to generate results."),
}

# label -> (provider, model id). Model ids can be overridden with GROQ_MODEL / GEMINI_MODEL secrets.
MODELS = {
    "Groq · GPT-OSS 120B (free)": ("groq", "openai/gpt-oss-120b"),
    "Groq · Llama 3.3 70B (free)": ("groq", "llama-3.3-70b-versatile"),
    "Gemini Flash (free tier)": ("gemini", "gemini-3.8-flash"),
    "Claude Sonnet 5.5 (balanced)": ("anthropic", "claude-sonnet-5-5"),
    "Claude Haiku 5.5 (fastest, cheapest)": ("anthropic", "claude-haiku-5-5"),
    "Claude Opus 5.5 (most capable)": ("anthropic", "claude-opus-5-5"),
}
NO_AI = "No AI · search and checks only"
PREFERENCE = ["groq", "anthropic", "gemini"]       # default when several keys are configured


_TOKEN = re.compile(r"[A-Za-z]+|\d{1,3}|[^\sA-Za-z\d]")


def estimate_tokens(text: str) -> int:
    """Conservative token estimate without downloading a tokenizer.
    Words count once, digits in groups of three, and each symbol ($ . , | ( ) once, plus 25% margin.
    Number-heavy schedules come out near 1 token per 2.5 characters; prose near 1 per 4."""
    return int(len(_TOKEN.findall(text or "")) * 1.25) + 4


def _messages_tokens(messages) -> int:
    return sum(estimate_tokens(str(m.get("content", ""))) + 4 for m in messages)


class TooLarge(Exception):
    """Raised before sending when a request cannot fit the provider's per-request cap."""


def make_client(provider: str, key: str):
    p = PROVIDERS[provider]
    if p.base_url is None:
        import anthropic
        return anthropic.Anthropic(api_key=key, max_retries=3)
    return OpenAICompat(provider, key)


# --------------------------------------------------------------------------
# OpenAI-compatible adapter with an Anthropic-style surface
# --------------------------------------------------------------------------
def _to_openai_tool(tool: dict) -> dict:
    return {"type": "function", "function": {"name": tool["name"], "description": tool.get("description", ""),
                                             "parameters": tool["input_schema"]}}


def _array_key(tool: dict) -> str | None:
    props = tool.get("input_schema", {}).get("properties", {})
    return next((k for k, v in props.items() if v.get("type") == "array"), None)


def salvage(generation: str, tool: dict) -> dict | None:
    """Recover the complete items from a malformed or cut-off tool call.
    Open models sometimes repeat keys, misspell one, or run out of tokens mid-list; every item that is
    itself valid JSON is kept, and the result is marked _partial so the caller can retry the rest."""
    try:
        whole = json.loads(generation)
        args = whole.get("arguments", whole) if isinstance(whole, dict) else None
        if isinstance(args, str):
            args = json.loads(args)
        if isinstance(args, dict):
            return args
    except (json.JSONDecodeError, TypeError):
        pass
    key = _array_key(tool)
    if not key:
        return None
    m = re.search(r'"%s"\s*:\s*\[' % re.escape(key), generation)
    if not m:
        return None
    items, depth, start, in_str, esc = [], 0, None, False, False
    for i in range(m.end(), len(generation)):
        ch = generation[i]
        if in_str:
            esc = (ch == "\\") and not esc
            if ch == '"' and not esc:
                in_str = False
            continue
        if ch == '"':
            in_str, esc = True, False
        elif ch == "{":
            depth += 1
            if depth == 1:
                start = i
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    obj = json.loads(generation[start:i + 1])
                    if isinstance(obj, dict):
                        items.append(obj)
                except json.JSONDecodeError:
                    pass
                start = None
        elif ch == "]" and depth == 0:
            return {key: items}
    return {key: items, "_partial": True}


def _failed_generation(e: Exception) -> str | None:
    """Groq returns the model's unparseable tool call in the error body (code tool_use_failed)."""
    body = getattr(e, "body", None)
    err = body.get("error", body) if isinstance(body, dict) else None
    if isinstance(err, dict) and err.get("code") == "tool_use_failed":
        return str(err.get("failed_generation") or "")
    return None


def _json_from_text(text: str) -> dict | None:
    """Some models answer a forced tool call with JSON in plain text; recover it."""
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        out = json.loads(m.group(0))
        return out if isinstance(out, dict) else None
    except json.JSONDecodeError:
        return None


class _Stream:
    def __init__(self, it):
        self._it = it

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    @property
    def text_stream(self):
        for chunk in self._it:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class OpenAICompat:
    def __init__(self, provider: str, key: str, client=None):
        self.provider = provider
        if client is None:
            from openai import OpenAI
            client = OpenAI(api_key=key, base_url=PROVIDERS[provider].base_url, max_retries=4, timeout=120)
        self._c = client
        self.messages = self

    def _extra(self, model: str) -> dict:
        # GPT-OSS is a reasoning model; low effort keeps answers inside the free-tier token budget
        return {"reasoning_effort": "low"} if self.provider == "groq" and "gpt-oss" in model else {}

    def _msgs(self, system, messages):
        return [{"role": "system", "content": system}] + [
            {"role": m["role"], "content": m["content"]} for m in messages]

    def _fit(self, kw: dict, extra_tokens: int = 0) -> dict:
        """Last guard before sending: keep input + output under the provider's per-request cap.
        Drops earlier chat turns first, then lowers the output allowance; never cuts the sources."""
        cap = PROVIDERS[self.provider].request_tokens
        if not cap:
            return kw
        msgs = kw["messages"]
        size = lambda: _messages_tokens(msgs) + extra_tokens
        while size() + kw["max_tokens"] > cap and len(msgs) > 2:
            del msgs[1]                                  # oldest turn after the system prompt
        room = cap - size()
        if room < 400:
            raise TooLarge(f"This request needs about {size():,} tokens of input, more than "
                           f"{PROVIDERS[self.provider].name}'s free-tier limit allows in one request.")
        kw["max_tokens"] = min(kw["max_tokens"], room)
        return kw

    def create(self, model, max_tokens, system, messages, tools=None, tool_choice=None, **_):
        kw = dict(model=model, max_tokens=max_tokens, messages=self._msgs(system, messages), **self._extra(model))
        kw = self._fit(kw, extra_tokens=sum(estimate_tokens(json.dumps(t)) for t in tools or []))
        if tools:
            kw["tools"] = [_to_openai_tool(t) for t in tools]
            name = tool_choice.get("name") if isinstance(tool_choice, dict) else None
            # Gemini documents only "auto"/"required"; Groq accepts a named function
            kw["tool_choice"] = ("required" if self.provider == "gemini"
                                 else {"type": "function", "function": {"name": name}} if name else "auto")
            kw["messages"][0]["content"] += f"\n\nRecord your output by calling the `{tools[0]['name']}` tool."
        try:
            resp = self._c.chat.completions.create(**kw)
        except Exception as e:
            gen = _failed_generation(e) if tools else None
            if gen is not None:                       # malformed tool call: keep what is complete
                data = salvage(gen, tools[0])
                if data is not None:
                    return NS(content=[NS(type="tool_use", name=tools[0]["name"], input=data)])
                raise
            if tools and _is_bad_request(e):         # fall back to "auto" if forced choice is rejected
                kw["tool_choice"] = "auto"
                resp = self._c.chat.completions.create(**kw)
            else:
                raise
        msg = resp.choices[0].message
        blocks = []
        for tc in getattr(msg, "tool_calls", None) or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = salvage(tc.function.arguments or "", tools[0]) if tools else None
                args = args or _json_from_text(tc.function.arguments) or {}
            blocks.append(NS(type="tool_use", name=tc.function.name, input=args))
        if tools and not blocks:
            parsed = _json_from_text(msg.content or "")
            if parsed is not None:
                blocks.append(NS(type="tool_use", name=tools[0]["name"], input=parsed))
        if msg.content and not blocks:
            blocks.append(NS(type="text", text=msg.content))
        return NS(content=blocks)

    def stream(self, model, max_tokens, system, messages, **_):
        kw = self._fit(dict(model=model, max_tokens=max_tokens, messages=self._msgs(system, messages)))
        return _Stream(self._c.chat.completions.create(stream=True, **kw, **self._extra(model)))


# --------------------------------------------------------------------------
# errors
# --------------------------------------------------------------------------
def _is_bad_request(e) -> bool:
    return type(e).__name__ in ("BadRequestError", "UnprocessableEntityError")


def describe_error(e: Exception, provider: str) -> str:
    kind = type(e).__name__
    p = PROVIDERS[provider]
    if kind == "TooLarge" or getattr(e, "status_code", None) == 413 or "Request too large" in str(e):
        return (f"This request is too large for {p.name}'s free tier (about 8,000 tokens per minute). "
                "Select fewer files under 'Search within', ask about one schedule at a time, or pick "
                "Gemini or Claude for large filings.")
    if kind in ("AuthenticationError", "PermissionDeniedError"):
        return f"The {p.name} API key was rejected. Check it in the sidebar or the {p.key_name} secret."
    if kind == "RateLimitError":
        hint = " The free tier allows a limited number of tokens per minute; wait a minute and try again." \
            if p.free else " Wait a moment and try again."
        return f"{p.name} rate limit reached.{hint}"
    if "tool_use_failed" in str(e):
        return (f"{p.name}'s model returned output that could not be read. Try again, select fewer files, or "
                "pick another model.")
    if kind == "NotFoundError":
        return (f"{p.name} did not recognise the model name. Set the correct id in the "
                f"{'GROQ_MODEL' if provider == 'groq' else 'GEMINI_MODEL' if provider == 'gemini' else 'model'} "
                f"secret. ({e})")
    return f"{p.name} returned an error: {e}"
