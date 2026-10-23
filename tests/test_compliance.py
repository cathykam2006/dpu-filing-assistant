"""Compliance check: order / cover letter vs filing."""
import json
from pathlib import Path
from types import SimpleNamespace as NS

from rag import compliance as C
from rag import ingest, llm, search

SAMPLES = Path(__file__).resolve().parents[1] / "samples"
REAL = "real_NSTAR_Gas_LDAC_DPU_26-PGAF-NSTAR_corrected.pdf"


def load(name):
    return ingest.load(name, (SAMPLES / name).read_bytes())


def sample_filing():
    return [load(n) for n in ("sample_pgaf_filing.pdf", "sample_ir_responses.pdf", "sample_schedule3.xlsx")]


def by_text(reqs, needle):
    return next(r for r in reqs if needle in r.text)


def test_sample_order_without_ai():
    reqs = C.check([load("order_sample_98-PGAF-SAMPLE.pdf")], sample_filing())
    assert len(reqs) == 7
    cap = by_text(reqs, "(G-51)")
    assert cap.text.startswith("ORDERED: That") and cap.status == "✗ Discrepancy"
    assert "cap of $0.8200; the filing shows 0.8302" in cap.discrepancy
    assert by_text(reqs, "Schedule 4").status == "✗ Not provided"
    att = by_text(reqs, "Attachment DPU-1-2")
    assert att.status == "◐ Partly met" and "mentioned in the filing" in att.discrepancy
    assert by_text(reqs, "$0.4310").status == "✓ Items found"
    deadline = by_text(reqs, "FURTHER ORDERED")
    assert deadline.status == "? Review" and "no filing date" in deadline.discrepancy


def test_cover_letter_enclosures():
    reqs = C.check([load("order_sample_cover_letter.pdf")], sample_filing())
    status = {r.text.split(": ", 1)[1].split(" –")[0]: r.status for r in reqs}
    assert status == {"Schedule 1": "✗ Not provided", "Schedule 2": "✓ Items found",
                      "Schedule 3": "✓ Items found", "Schedule 4": "✗ Not provided",
                      "Attachment A": "✗ Not provided"}


def test_values_checked_in_the_right_rate_class_column():
    order = ingest.load("o.txt", b"The Department approves an Off-Peak GAF of $0.3900 per therm.\n"
                                 b"The Residential Assistance Adjustment Factor for G-41 shall be $0.1716 per therm.\n"
                                 b"The Company shall set the Residential Assistance Adjustment Factor for R-3 at "
                                 b"$0.1716 per therm.\n"
                                 b"The Energy Efficiency Surcharge for G-41 shall not exceed $0.1300 per therm.\n"
                                 b"The Energy Efficiency Surcharge for R-1 shall not exceed $0.3500 per therm.")
    reqs = C.check([order], [load(REAL)])
    s = {r.id: (r.status, r.discrepancy) for r in reqs}
    assert s["R1"][0] == "✗ Discrepancy" and "$0.3842 in all 10 rate classes" in s["R1"][1]
    assert s["R2"][0] == "✗ Discrepancy" and "G-41: $0.1388" in s["R2"][1]      # $0.1716 is residential
    assert s["R3"][0] == "✓ Items found"
    assert s["R4"][0] == "✗ Discrepancy" and "G-41: $0.1385, which is above it" in s["R4"][1]
    assert s["R5"][0] == "✓ Items found"                                         # $0.3299 is under the cap


class FakeTools:
    def __init__(self, by_tool):
        self.by_tool, self.calls = by_tool, []
        self.messages = self

    def create(self, **kw):
        self.calls.append(kw)
        name = kw["tools"][0]["name"]
        data = self.by_tool[name](kw) if callable(self.by_tool[name]) else self.by_tool[name]
        return NS(content=[NS(type="tool_use", name=name, input=data)])


def test_model_cannot_override_code_checks():
    order, filing = load("order_sample_98-PGAF-SAMPLE.pdf"), sample_filing()
    idx = search.Index([c for d in filing for c in d.chunks])
    extra = {"requirements": [
        {"quote": "the Company shall include a typical bill impact analysis for each rate class",  # already found
         "summary": "bill impacts", "source_id": 1},
        {"quote": "The Company must file a redlined tariff.", "summary": "redline", "source_id": 1}]}

    def judge(kw):
        ids = [s.split('"')[1] for s in kw["messages"][-1]["content"].split("<requirement id=")[1:]]
        out = []
        for i in ids:
            out.append({"id": i, "status": "met", "explanation": "Looks fine.",
                        "evidence_quote": "A typical residential heating customer using 120 therms", "evidence_source_id": 1})
        return {"results": out}

    client = FakeTools({"record_requirements": extra, "record_compliance": judge})
    reqs = C.check([order], filing, client, "m", idx)
    assert len(reqs) == 8                                       # 7 by pattern + 1 new from the model
    redline = by_text(reqs, "redlined tariff")
    assert redline.verified is False and "not found verbatim" in redline.discrepancy
    sched4 = by_text(reqs, "Schedule 4")
    assert sched4.status == "✗ Not provided"                    # model said met; Schedule 4 is not in the filing
    assert "code check disagrees" in sched4.discrepancy
    cap = by_text(reqs, "(G-51)")
    assert cap.status == "✗ Discrepancy"
    hedge = by_text(reqs, "hedged")
    assert hedge.status == "✓ Met"                              # no code-checkable item; model's view used
    assert any(r.ai_where for r in reqs)                        # evidence quote verified in the filing


def test_contradiction_from_model_and_unverified_evidence():
    order = ingest.load("o.txt", b"The Company shall use the approved weather normalization method.")
    filing = sample_filing()
    idx = search.Index([c for d in filing for c in d.chunks])
    client = FakeTools({"record_requirements": {"requirements": []},
                        "record_compliance": {"results": [{"id": "R1", "status": "contradicts",
                                                           "explanation": "Uses a different method.",
                                                           "evidence_quote": "invented sentence"}]}})
    r = C.check([order], filing, client, "m", idx)[0]
    assert r.status == "✗ Discrepancy" and r.discrepancy == "Uses a different method."
    assert r.ai_evidence.startswith("(quote not found in the filing)")


def test_follow_up_irs_for_gaps():
    reqs = C.check([load("order_sample_98-PGAF-SAMPLE.pdf")], sample_filing())
    df = C.results_frame(reqs)
    irs = C.follow_up_irs(df, "DPU", 2, 4)
    assert irs["No."].iloc[0] == "DPU 2-4" and len(irs) == (~df["Status"].str.startswith("✓")).sum()
    assert irs["Request"].iloc[0].startswith("Refer to order_sample_98-PGAF-SAMPLE.pdf at 1, which states: “")
    assert df["Status"].iloc[0] == "✗ Discrepancy"              # sorted worst first


def test_groq_compliance_calls_fit_the_limit():
    G = llm.PROVIDERS["groq"]
    order, filing = load("order_sample_98-PGAF-SAMPLE.pdf"), sample_filing() + [load(REAL)]
    idx = search.Index([c for d in filing for c in d.chunks])
    sent = []

    class Comp:
        def create(self, **kw):
            sent.append(llm._messages_tokens(kw["messages"]) + kw["max_tokens"] +
                        sum(llm.estimate_tokens(json.dumps(t)) for t in kw["tools"]))
            name = kw["tools"][0]["function"]["name"]
            data = {"requirements": []} if name == "record_requirements" else {"results": []}
            calls = [NS(function=NS(name=name, arguments=json.dumps(data)))]
            return NS(choices=[NS(message=NS(content=None, tool_calls=calls))])

    client = llm.OpenAICompat("groq", "k", client=NS(chat=NS(completions=Comp())))
    C.check([order], filing, client, "m", idx, G.batch_tokens, G.max_tokens, llm.estimate_tokens)
    assert len(sent) >= 2 and max(sent) <= G.request_tokens


def test_model_met_without_evidence_or_with_open_deadline_is_review():
    filing = sample_filing()
    idx = search.Index([c for d in filing for c in d.chunks])
    order = ingest.load("o.txt", b"The Company shall describe its weather normalization method.\n\n"
                                 b"The Company shall file no later than September 1, 2026.")
    client = FakeTools({"record_requirements": {"requirements": []},
                        "record_compliance": {"results": [
                            {"id": "R1", "status": "met", "explanation": "ok", "evidence_quote": ""},
                            {"id": "R2", "status": "met", "explanation": "ok",
                             "evidence_quote": "Sample Gas Company"}]}})
    r1, r2 = C.check([order], filing, client, "m", idx)
    assert r1.status == "? Review" and "no quote" in r1.discrepancy
    assert r2.status == "? Review" and "no filing date" in r2.discrepancy
