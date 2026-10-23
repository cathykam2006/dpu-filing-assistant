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
    context_chars: int         # excerpt budget for one question
    batch_chars: int           # excerpt budget for one extraction call
    max_tokens: int            # output budget
    free: bool
    data_note: str


PROVIDERS = {
    "groq": Provider("Groq", "GROQ_API_KEY", "console.groq.com/keys", "https://api.groq.com/openai/v1",
                     context_chars=14_000, batch_chars=12_000, max_tokens=1500, free=True,
                     data_note="Excerpts are sent to Groq to generate results."),
    "gemini": Provider("Google Gemini", "GEMINI_API_KEY", "aistudio.google.com/apikey",
                       "https://generativelanguage.googleapis.com/v1beta/openai/",
                       context_chars=100_000, batch_chars=40_000, max_tokens=3000, free=True,
                       data_note="Excerpts are sent to Google. On Gemini's free tier, Google may use "
                                 "inputs to improve its models and human reviewers may read them."),
    "anthropic": Provider("Anthropic", "ANTHROPIC_API_KEY", "console.anthropic.com", None,
                          context_chars=120_000, batch_chars=60_000, max_tokens=4000, free=False,
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

    def create(self, model, max_tokens, system, messages, tools=None, tool_choice=None, **_):
        kw = dict(model=model, max_tokens=max_tokens, messages=self._msgs(system, messages), **self._extra(model))
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
                args = _json_from_text(tc.function.arguments) or {}
            blocks.append(NS(type="tool_use", name=tc.function.name, input=args))
        if tools and not blocks:
            parsed = _json_from_text(msg.content or "")
            if parsed is not None:
                blocks.append(NS(type="tool_use", name=tools[0]["name"], input=parsed))
        if msg.content and not blocks:
            blocks.append(NS(type="text", text=msg.content))
        return NS(content=blocks)

    def stream(self, model, max_tokens, system, messages, **_):
        return _Stream(self._c.chat.completions.create(
            model=model, max_tokens=max_tokens, messages=self._msgs(system, messages), stream=True,
            **self._extra(model)))


# --------------------------------------------------------------------------
# errors
# --------------------------------------------------------------------------
def _is_bad_request(e) -> bool:
    return type(e).__name__ in ("BadRequestError", "UnprocessableEntityError")


def describe_error(e: Exception, provider: str) -> str:
    kind = type(e).__name__
    p = PROVIDERS[provider]
    if kind in ("AuthenticationError", "PermissionDeniedError"):
        return f"The {p.name} API key was rejected. Check it in the sidebar or the {p.key_name} secret."
    if kind == "RateLimitError":
        hint = " The free tier allows a limited number of tokens per minute; wait a minute and try again." \
            if p.free else " Wait a moment and try again."
        return f"{p.name} rate limit reached.{hint}"
    if kind == "NotFoundError":
        return (f"{p.name} did not recognise the model name. Set the correct id in the "
                f"{'GROQ_MODEL' if provider == 'groq' else 'GEMINI_MODEL' if provider == 'gemini' else 'model'} "
                f"secret. ({e})")
    return f"{p.name} returned an error: {e}"
