"""Groq/Gemini adapter, excerpt budgets and the no-AI mode. No network calls."""
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


from rag import discovery as D
from rag import ingest, llm, rules, search

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"


def load(name):
    return ingest.load(name, (SAMPLES / name).read_bytes())


class FakeCompletions:
    """Stands in for openai.OpenAI().chat.completions."""

    def __init__(self, reply=None, tool_args=None, reject_forced=False):
        self.reply, self.tool_args, self.reject_forced, self.calls = reply, tool_args, reject_forced, []

    def create(self, **kw):
        self.calls.append(kw)
        if self.reject_forced and kw.get("tool_choice") not in (None, "auto"):
            raise type("BadRequestError", (Exception,), {})("tool_choice not supported")
        if kw.get("stream"):
            return iter([NS(choices=[NS(delta=NS(content=p))]) for p in ("GAF is ", "$0.9120 [S1].")])
        calls = None
        if self.tool_args is not None and kw.get("tools"):
            calls = [NS(function=NS(name=kw["tools"][0]["function"]["name"], arguments=json.dumps(self.tool_args)))]
        return NS(choices=[NS(message=NS(content=self.reply, tool_calls=calls))])


def compat(provider, fake):
    return llm.OpenAICompat(provider, "k", client=NS(chat=NS(completions=fake)))


def test_tool_call_translation_groq():
    fake = FakeCompletions(tool_args={"figures": []})
    out = D._call_tool(compat("groq", fake), "openai/gpt-oss-120b", "sys", "user", D.FIGURE_TOOL)
    kw = fake.calls[0]
    assert out == {"figures": []}
    assert kw["tool_choice"] == {"type": "function", "function": {"name": "record_figures"}}
    assert kw["tools"][0]["function"]["parameters"] == D.FIGURE_TOOL["input_schema"]
    assert kw["messages"][0]["role"] == "system" and kw["reasoning_effort"] == "low"


def test_gemini_uses_required_and_falls_back_to_auto():
    fake = FakeCompletions(tool_args={"requests": []}, reject_forced=True)
    out = D._call_tool(compat("gemini", fake), "gemini-3.8-flash", "s", "u", D.IR_TOOL)
    assert [c["tool_choice"] for c in fake.calls] == ["required", "auto"]
    assert out == {"requests": []} and "reasoning_effort" not in fake.calls[0]


def test_json_in_plain_text_is_recovered():
    fake = FakeCompletions(reply='Here you go:\n```json\n{"grades": [{"ir": "DPU-1-1"}]}\n```')
    out = D._call_tool(compat("groq", fake), "llama-3.3-70b-versatile", "s", "u", D.GRADE_TOOL)
    assert out["grades"][0]["ir"] == "DPU-1-1"


def test_streaming_answer_through_adapter():
    from rag import answer
    hits = [(c, 1.0) for c in load("sample_pgaf_filing.pdf").chunks]
    gen, _ = answer.stream_answer(compat("groq", FakeCompletions()), "m", "GAF?", hits, [], 500)
    assert "".join(gen) == "GAF is $0.9120 [S1]."


def test_open_model_quirks_in_extraction():
    """Strings for numbers, unknown metrics and bad source ids are tolerated."""
    chunks = load("sample_pgaf_filing.pdf").chunks
    numeric = [c for c in chunks if D.has_numbers(c)]
    sid = next(i for i, c in enumerate(numeric, 1) if "LNG peaking |" in c.text)
    args = {"figures": [
        {"metric": "lng_cost", "label": "LNG", "value": "$4,817,420", "unit": "$",
         "quote": "LNG peaking | 4,817,420", "source_id": str(sid)},
        {"metric": "made_up", "label": "x", "value": 1, "unit": "", "quote": "", "source_id": 99},
        "not a dict"]}
    figs = D.extract_figures(compat("groq", FakeCompletions(tool_args=args)), "m", chunks, max_chars=10_000)
    assert len(figs) == 1 and figs[0].value == 4_817_420 and figs[0].verified


def test_error_messages():
    rate = type("RateLimitError", (Exception,), {})("429")
    auth = type("AuthenticationError", (Exception,), {})("401")
    assert "free tier" in llm.describe_error(rate, "groq")
    assert "GEMINI_API_KEY" in llm.describe_error(auth, "gemini")


def test_search_respects_char_budget():
    chunks = [c for n in ("sample_pgaf_filing.pdf", "sample_ir_responses.pdf") for c in load(n).chunks]
    idx = search.Index(chunks)
    hits = idx.search("LNG peaking supply cost", max_chars=1200)
    assert sum(len(c.text) for c, _ in hits) <= 1200 or len(hits) == 1
    assert any("4,817,420" in c.text for c, _ in hits)
    assert len(idx.search("anything")) == len(chunks)                 # no budget: everything still fits
    top = idx.search("hedged winter purchases", k=1, always_rank=True)
    assert top and top[0][0].ir == "DPU-1-3"


# ------------------------------------------------------------------ no-AI mode
def test_rules_figures_and_checks():
    figs = rules.extract_figures(load("sample_pgaf_filing.pdf").chunks + load("sample_schedule3.xlsx").chunks)
    by = {(f.metric, f.rate_class, f.status): f.value for f in figs}
    assert by[("gaf", "R-3", "proposed")] == 0.912 and by[("gaf", "R-3", "prior")] == 0.8455
    assert by[("cost_of_gas_total", "", "")] == 142_650_000 and by[("lng_cost", "", "")] == 4_817_420
    assert by[("forecast_sales", "", "")] == 156_410_000
    assert ("ldaf", "residential", "proposed") in by
    assert all(f.verified for f in figs)
    kinds = {i.kind for i in D.run_checks(figs)}
    assert "tie-out" not in kinds and "unverified" not in kinds


def test_rules_bill_impact_from_prose():
    figs = rules.extract_figures(load("sample_ir_responses.pdf").chunks)
    vals = {(f.metric, f.value) for f in figs}
    assert ("bill_impact_dollars", 7.98) in vals and ("bill_impact_percent", 2.8) in vals
    assert ("bill_impact_dollars", 283.41) not in vals             # the "from" amount is not the change


def test_rules_ir_templates():
    chunks = load("sample_pgaf_filing.pdf").chunks
    figs = rules.extract_figures(chunks)
    bad = [D.Figure(**{**f.__dict__, "value": 150_000_000}) if f.metric == "cost_of_gas_total" else f
           for f in figs]
    df = rules.draft_irs(bad, D.run_checks(bad), " ".join(c.text for c in chunks), "DPU", 3, 8)
    assert df["No."].iloc[0] == "DPU 3-1" and df["Topic"].iloc[0] == "Cost of gas tie-out"
    assert all(r.startswith("Refer to ") for r in df["Request"])
    assert any("Hedging" == t for t in df["Topic"])                     # hedging not mentioned in filing
    assert any("4,817,420" in r for r in df["Request"]) and len(df) <= 8


def test_rules_response_checks():
    chunks = load("sample_ir_responses.pdf").chunks
    df = rules.grade_responses(chunks, ["sample_ir_responses.pdf"], ["DPU-1-7"]).set_index("IR")
    assert df.loc["DPU-1-2", "Status"] == "refers to attachment"
    assert df.loc["DPU-1-2", "Attachments not uploaded"] == "Attachment DPU-1-2"
    assert df.loc["DPU-1-7", "Status"] == "no response found"
    obj = ingest.load("o.txt", b"Information Request DPU-2-1\nRequest: Provide X.\nResponse: The Company "
                               b"objects to this request as overly broad.\n\nInformation Request DPU-2-2\n"
                               b"Request: Provide Y.\nResponse: The Company will supplement this response.")
    g = rules.grade_responses(obj.chunks, ["o.txt"]).set_index("IR")["Status"]
    assert g["DPU-2-1"] == "objection" and g["DPU-2-2"] == "partial"


def test_highlight():
    assert "**LNG**" in rules.highlight("LNG peaking supply", "What is the LNG cost?")


# ------------------------------------------------------------------ Groq's 8,000 tokens-per-minute limit
class Capture(FakeCompletions):
    def __init__(self):
        super().__init__(tool_args={"figures": [], "requests": [], "grades": []})

    def size(self, i=-1):
        kw = self.calls[i]
        return llm._messages_tokens(kw["messages"]) + kw["max_tokens"] + sum(
            llm.estimate_tokens(json.dumps(t)) for t in kw.get("tools") or [])


def every_file():
    names = ["sample_pgaf_filing.pdf", "sample_ir_responses.pdf", "sample_schedule3.xlsx",
             "real_NSTAR_Gas_LDAC_DPU_26-PGAF-NSTAR_corrected.pdf"]
    return [c for n in names for c in load(n).chunks]


def test_estimate_is_conservative_for_numbers():
    row = "19 LDAF = Total Local Distribution Adjustment Factor $0.7434 $0.5718 $0.7434 $0.5718 $0.5085"
    assert llm.estimate_tokens(row) >= len(row) / 2.5          # number-heavy text: ~2 chars per token
    assert llm.estimate_tokens("The Company proposes a new factor.") <= 12


def test_groq_question_with_everything_loaded_fits_the_limit():
    from rag import answer
    G = llm.PROVIDERS["groq"]
    chunks = every_file()
    hits = search.Index(chunks).search("What is the LDAF for R-3 and the LNG cost?", max_chars=G.context_tokens,
                                       measure=llm.estimate_tokens)
    history = [{"role": r, "content": "A long earlier answer with $0.7434 and $4,817,420 figures. " * 60}
               for r in ("user", "assistant") * 5]
    fake = Capture()
    gen, _ = answer.stream_answer(compat("groq", fake), "openai/gpt-oss-120b", "LDAF?", hits, history,
                                  G.max_tokens, G.history_turns)
    "".join(gen)
    assert fake.size() <= G.request_tokens < 8000


def test_groq_extraction_drafting_and_grading_fit_the_limit():
    G = llm.PROVIDERS["groq"]
    chunks = every_file()
    fake = Capture()
    client = compat("groq", fake)
    D.extract_figures(client, "m", chunks, max_chars=G.batch_tokens, max_tokens=G.max_tokens,
                      measure=llm.estimate_tokens)
    issues = [D.Issue("missing", "low", "No typical customer bill impact found. " * 5, "x.pdf")] * 6
    D.draft_irs(client, "m", chunks, issues, max_chars=G.batch_tokens, max_tokens=G.max_tokens,
                measure=llm.estimate_tokens)
    D.grade_responses(client, "m", chunks, [], max_chars=G.batch_tokens, max_tokens=G.max_tokens,
                      measure=llm.estimate_tokens)
    assert len(fake.calls) >= 4
    assert max(fake.size(i) for i in range(len(fake.calls))) <= G.request_tokens


def test_oversized_request_is_stopped_before_sending():
    fake = Capture()
    with pytest.raises(llm.TooLarge):
        compat("groq", fake).create(model="m", max_tokens=1200, system="s",
                                    messages=[{"role": "user", "content": "$0.7434 " * 3000}])
    assert fake.calls == []                                     # nothing was sent
    msg = llm.describe_error(llm.TooLarge("x"), "groq")
    assert "too large" in msg and "Search within" in msg
