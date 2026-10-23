"""Discovery and evaluation tests. Claude is replaced by a fake that returns tool-use blocks,
so these check the plumbing, verification and arithmetic, not model quality."""
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from rag import discovery as D
from rag import evaluate as E
from rag import ingest

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"


@pytest.fixture(scope="session", autouse=True)
def samples():
    if not (SAMPLES / "sample_pgaf_filing.pdf").exists():
        subprocess.run([sys.executable, str(SAMPLES / "make_samples.py")], check=True)


def load(name, rename=None):
    return ingest.load(rename or name, (SAMPLES / name).read_bytes())


def src_id(chunks, needle):
    """1-based id of the first numeric chunk containing needle (as the extractor numbers them)."""
    numeric = [c for c in chunks if D.has_numbers(c)]
    return next(i for i, c in enumerate(numeric, 1) if needle in c.text)


class FakeTools:
    """messages.create returning a tool_use block chosen by tool name."""

    def __init__(self, by_tool, text=""):
        self.by_tool, self.text, self.calls = by_tool, text, []
        self.messages = self

    def create(self, **kw):
        self.calls.append(kw)
        if kw.get("tools"):
            name = kw["tools"][0]["name"]
            data = self.by_tool[name]
            data = data(kw) if callable(data) else data
            return NS(content=[NS(type="tool_use", name=name, input=data)])
        return NS(content=[NS(type="text", text=self.text)])

    def stream(self, **kw):   # used by evaluate.run through answer.stream_answer
        self.calls.append(kw)
        text = self.text

        class S:
            text_stream = iter([text])
            def __enter__(s): return s
            def __exit__(s, *a): return False
        return S()


def fake_figures(chunks, total=142_650_000):
    t = src_id(chunks, "Residential heating (R-3) |")
    c = src_id(chunks, "Commodity | 98,300,000")
    return {"figures": [
        dict(metric="gaf", label="Residential heating", value=0.9120, unit="$/therm", rate_class="R-3",
             status="proposed", quote="Residential heating (R-3) | 0.8455 | 0.9120 | +0.0665", source_id=t),
        dict(metric="gaf", label="Residential heating", value=0.8455, unit="$/therm", rate_class="R-3",
             status="prior", quote="Residential heating (R-3) | 0.8455 | 0.9120", source_id=t),
        dict(metric="commodity_cost", label="Commodity", value=98_300_000, unit="$",
             quote="Commodity | 98,300,000", source_id=c),
        dict(metric="demand_charges", label="Pipeline & storage demand", value=36_120_000, unit="$",
             quote="Pipeline & storage demand | 36,120,000", source_id=c),
        dict(metric="lng_cost", label="LNG peaking", value=4_817_420, unit="$",
             quote="LNG peaking | 4,817,420", source_id=c),
        dict(metric="reconciliation", label="Prior-period reconciliation", value=3_412_580, unit="$",
             quote="Prior-period reconciliation | 3,412,580", source_id=c),
        dict(metric="cost_of_gas_total", label="Total", value=total, unit="$",
             quote=f"Total | {total:,}", source_id=c),
        # a hallucinated figure: the quote is not in the filing
        dict(metric="return_on_equity", label="ROE", value=9.7, unit="%",
             quote="allowed return on equity of 9.7%", source_id=c),
    ]}


def test_extract_verifies_quotes_and_tool_call():
    chunks = load("sample_pgaf_filing.pdf").chunks
    client = FakeTools({"record_figures": fake_figures(chunks)})
    figs = D.extract_figures(client, "m", chunks)
    call = client.calls[0]
    assert call["tool_choice"] == {"type": "tool", "name": "record_figures"}
    assert '<source id="1"' in call["messages"][0]["content"]
    ok = {f.metric: f.verified for f in figs}
    assert ok["gaf"] and ok["cost_of_gas_total"] and ok["lng_cost"]
    assert ok["return_on_equity"] is False
    assert all(f.file == "sample_pgaf_filing.pdf" and f.location.startswith("p.") for f in figs)


def test_checks_tie_out_large_change_and_unverified():
    chunks = load("sample_pgaf_filing.pdf").chunks
    good = D.extract_figures(FakeTools({"record_figures": fake_figures(chunks)}), "m", chunks)
    kinds = [i.kind for i in D.run_checks(good)]
    assert "tie-out" not in kinds                      # 98.3+36.12+4.82+3.41 = 142.65
    assert "unverified" in kinds                       # the invented ROE
    assert "missing" in kinds                          # no bill impact in the figures given
    # break the total: the tie-out check must catch it
    bad = D.extract_figures(FakeTools({"record_figures": fake_figures(chunks, total=150_000_000)}), "m", chunks)
    tie = [i for i in D.run_checks(bad) if i.kind == "tie-out"]
    assert tie and tie[0].severity == "high" and "7,350,000" in tie[0].description
    # +7.9% is under the 10% threshold; lower the threshold and it is flagged
    old = D.BIG_CHANGE_PCT
    try:
        D.BIG_CHANGE_PCT = 5
        assert any(i.kind == "large-change" and "+7.9%" in i.description for i in D.run_checks(good))
    finally:
        D.BIG_CHANGE_PCT = old


def test_draft_irs_numbering_and_citations():
    chunks = load("sample_pgaf_filing.pdf").chunks
    reqs = {"requests": [
        {"topic": "Commodity", "request": "Refer to page 1. Please explain the Algonquin forecast.",
         "rationale": "Main driver.", "source_ids": [1]},
        {"topic": "LNG", "request": "Please provide workpapers for LNG peaking costs.",
         "rationale": "No support.", "source_ids": [4, 99]}]}
    issues = [D.Issue("missing", "low", "No typical customer bill impact found.", "x.pdf")]
    client = FakeTools({"record_requests": reqs})
    df = D.draft_irs(client, "m", chunks, issues, prefix="DPU", set_no=2)
    assert list(df["No."]) == ["DPU-2-1", "DPU-2-2"]
    assert "p. 2 (table)" in df["Based on"].iloc[1] and "99" not in df["Based on"].iloc[1]
    assert "No typical customer bill impact" in client.calls[0]["messages"][0]["content"]
    assert "Information Request DPU-2-1" in D.irs_to_text(df)


def test_grade_responses_flags_missing_attachment_and_unanswered():
    chunks = load("sample_ir_responses.pdf").chunks
    grades = {"grades": [
        {"ir": "DPU-1-2", "status": "refers to attachment", "summary": "Gives the calculation.",
         "gaps": "Monthly detail is only in the attachment.", "attachments": ["Attachment DPU-1-2"],
         "follow_up": "Provide Attachment DPU-1-2 in Excel with formulas intact."},
        {"ir": "DPU-1-3", "status": "answered", "summary": "28% hedged at $3.95.", "gaps": "",
         "attachments": [], "follow_up": ""}]}
    df = D.grade_responses(FakeTools({"record_grades": grades}), "m", chunks,
                           ["sample_ir_responses.pdf"], drafted=["DPU-1-3", "DPU-1-9"])
    row = df.set_index("IR")
    assert row.loc["DPU-1-2", "Attachments not uploaded"] == "Attachment DPU-1-2"
    assert row.loc["DPU-1-9", "Status"] == "no response found"
    assert "p. 1" in row.loc["DPU-1-3", "Where"]


def test_compare_joins_by_metric_and_rate_class():
    a, b = load("sample_pgaf_filing.pdf", "prior.pdf"), load("sample_pgaf_filing.pdf", "current.pdf")
    figs = []
    for doc, gaf in ((a, 0.8455), (b, 0.9120)):
        t = src_id(doc.chunks, "Residential heating (R-3) |")
        figs += D.extract_figures(FakeTools({"record_figures": {"figures": [
            dict(metric="gaf", label="Residential heating", value=gaf, unit="$/therm", rate_class="R-3",
                 status="proposed", quote=f"{gaf:.4f}", source_id=t)]}}), "m", doc.chunks)
    cmp = D.compare(figs, "prior.pdf", "current.pdf")
    r = cmp.iloc[0]
    assert r["value (prior)"] == 0.8455 and r["value (current)"] == 0.912
    assert round(r["Change %"], 1) == 7.9


# ------------------------------------------------------------------ evaluation
def all_sample_chunks():
    return [c for n in ("sample_pgaf_filing.pdf", "sample_ir_responses.pdf", "sample_schedule3.xlsx")
            for c in load(n).chunks]


def test_value_matching():
    assert E.value_found("0.9120", "a GAF of $0.912 per therm")
    assert E.value_found("3,412,580", "under-collection of $3412580")
    assert not E.value_found("3,412,580", "$3,412,000")
    assert E.value_found("7.9", "an increase of 7.9%")


def test_sample_question_set_retrieval_only():
    qs = E.load_questions(ROOT / "eval" / "questions_sample.csv")
    assert len(qs) >= 15 and sum(q.abstain for q in qs) >= 2
    df = E.results_frame(E.run(qs, all_sample_chunks(), k=4))
    assert df["Retrieval hit"].dropna().all() and df["Figures correct"].isna().all()


def test_answer_scoring_and_baseline():
    qs = [q for q in E.load_questions(ROOT / "eval" / "questions_sample.csv") if q.id in ("S08", "S16")]
    client = FakeTools({}, text="LNG peaking supply is $4,817,420 [S1]. The filings do not state an ROE.")
    df = E.results_frame(E.run(qs, all_sample_chunks(), client=client, k=4, baseline=True)).set_index("ID")
    assert df.loc["S08", "Figures correct"] == True          # noqa: E712
    assert df.loc["S16", "Figures correct"] == True          # abstained correctly
    assert df.loc["S08", "Baseline correct"] == True         # noqa: E712
    md = E.report_markdown(df.reset_index())
    assert "Answer correct (assistant)" in md and "whole-document baseline" in md
    assert "ChatGPT file upload" in E.manual_template(qs)


def test_wrong_source_id_is_relocated():
    chunks = load("sample_pgaf_filing.pdf").chunks
    wrong = {"figures": [dict(metric="lng_cost", label="LNG peaking", value=4_817_420, unit="$",
                              quote="LNG peaking | 4,817,420", source_id=1)]}
    f = D.extract_figures(FakeTools({"record_figures": wrong}), "m", chunks)[0]
    assert f.verified and f.location == "p. 2 (table)"


def test_edited_ir_table_with_blank_and_added_rows():
    import pandas as pd
    df = pd.DataFrame([
        {"No.": "DPU-1-1", "Topic": "LNG", "Request": "Explain LNG costs.", "Why": None, "Based on": float("nan")},
        {"No.": None, "Topic": None, "Request": None, "Why": None, "Based on": None},          # empty new row
        {"No.": None, "Topic": "Added", "Request": "Provide workpapers.", "Why": "", "Based on": None},
    ])
    clean = D.clean_irs(df, "DPU", 1)
    assert list(clean["No."]) == ["DPU-1-1", "DPU-1-2"] and clean["Why"].iloc[0] == ""
    text = D.irs_to_text(df, "Set 1")
    assert "Information Request DPU-1-2\nProvide workpapers." in text and "None" not in text
