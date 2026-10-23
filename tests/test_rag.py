"""Tests run on the fictional sample filings; the Claude API is replaced by a fake."""
import subprocess
import sys
from pathlib import Path

import pytest

from rag import answer, ingest, search

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"


@pytest.fixture(scope="session", autouse=True)
def samples():
    if not (SAMPLES / "sample_pgaf_filing.pdf").exists():
        subprocess.run([sys.executable, str(SAMPLES / "make_samples.py")], check=True)


def load(name):
    return ingest.load(name, (SAMPLES / name).read_bytes())


def test_pdf_pages_tables_and_docket():
    d = load("sample_pgaf_filing.pdf")
    assert d.doc_type == "GAF / cost of gas filing" and d.pages == 2
    assert d.dockets == ["99-PGAF-SAMPLE"]
    tables = [c for c in d.chunks if c.kind == "table"]
    assert any("0.9120" in c.text and "Residential heating" in c.text for c in tables)
    assert any(c.location == "p. 2 (table)" and "4,817,420" in c.text for c in tables)


def test_ir_responses_split_per_question():
    d = load("sample_ir_responses.pdf")
    assert d.doc_type == "Information request response"
    by_ir = {c.ir: c.text for c in d.chunks if c.ir}
    assert set(by_ir) == {"DPU-1-1", "DPU-1-2", "DPU-1-3", "AG-1-1"}
    assert "3,412,580" in by_ir["DPU-1-2"] and "hedge" in by_ir["DPU-1-3"]


def test_excel_keeps_header_and_real_rows():
    d = load("sample_schedule3.xlsx")
    c = d.chunks[0]
    assert c.location == "Sheet 'Schedule 3', rows 5-10"
    assert "Commodity price ($/MMBtu)" in c.text and "Jan-27 | 33870000 | 4.38" in c.text


def test_tokenizer_keeps_regulatory_ids():
    toks = search.tokenize("IR DPU-1-2: D.P.U. 25-PGAF-GRID rate G-41 at $0.9120 and $3,412,580")
    for t in ("dpu-1-2", "25-pgaf-grid", "g-41", "0.9120", "3412580"):
        assert t in toks


def test_search_prefers_named_ir(monkeypatch):
    monkeypatch.setattr(search, "WHOLE_CORPUS_CHARS", 0)     # force ranking even on small docs
    chunks = [c for n in ("sample_ir_responses.pdf", "sample_pgaf_filing.pdf") for c in load(n).chunks]
    hits = search.Index(chunks).search("What does the company say in DPU-1-3?", k=2)
    assert any(c.ir == "DPU-1-3" for c, _ in hits)
    hits = search.Index(chunks).search("LNG peaking supply cost", k=2)
    assert any("4,817,420" in c.text for c, _ in hits)


def test_small_corpus_sends_everything_and_scope_filters():
    chunks = [c for n in ("sample_ir_responses.pdf", "sample_schedule3.xlsx") for c in load(n).chunks]
    idx = search.Index(chunks)
    assert len(idx.search("anything")) == len(chunks)
    only = idx.search("anything", files=["sample_schedule3.xlsx"])
    assert {c.file for c, _ in only} == {"sample_schedule3.xlsx"}


class FakeStream:
    def __init__(self, text):
        self.text_stream = iter([text[:20], text[20:]])

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeClient:
    def __init__(self, reply):
        self.reply, self.calls = reply, []
        self.messages = self

    def stream(self, **kw):
        self.calls.append(kw)
        return FakeStream(self.reply)


def test_answer_prompt_and_citations():
    chunks = load("sample_ir_responses.pdf").chunks
    hits = search.Index(chunks).search("hedging")
    client = FakeClient("The Company hedged 28% of winter purchases [S4]. Bill impact is +$7.98 [S5][S4].")
    gen, sources = answer.stream_answer(client, "claude-sonnet-5-5", "Is supply hedged?", hits, [])
    text = "".join(gen)
    call = client.calls[0]
    assert call["model"] == "claude-sonnet-5-5" and "Answer ONLY from the numbered sources" in call["system"]
    user = call["messages"][-1]["content"]
    assert '<source id="S1"' in user and 'ir="DPU-1-3"' in user and "Question: Is supply hedged?" in user
    assert answer.cited_ids(text) == [4, 5]
    assert len(sources) == len(hits)


def test_unsupported_file_type():
    with pytest.raises(ValueError):
        ingest.load("x.pptx", b"123")
