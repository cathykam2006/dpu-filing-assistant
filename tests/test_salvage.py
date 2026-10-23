"""Groq's tool_use_failed error: malformed / cut-off JSON from an open model (the real case reported)."""
import json
from types import SimpleNamespace as NS

from rag import discovery as D
from rag import ingest, llm

# the failed_generation Groq returned (shortened at the same point: cut off mid-item)
FAILED = ('{"name": "record_figures", "arguments": {\n "figures": [\n {\n "label": "Environmental Response Costs to be '
          'Recovered (ERC)",\n "metric": "other",\n "quote": "959,029",\n"sourceid":4,\n"unit":"959,029",\n'
          '"sourceid":4,\n"unit":"",\n "value": 959029\n },\n {\n "label": "Combined Tax Rate",\n "metric": "other",\n'
          ' "quote": "27.320%",\n "source_id": 4,\n "unit": "%",\n "value": 27.32\n },\n {\n "label": "RRAC '
          'Reconciliation Adjustment",\n "metric": "reconciliation",\n "quote": "807,740",\n "source_id": 4,\n'
          ' "unit": "",\n "value": 807740\n },\n {\n "label": "Small Commercial and Industrial Allocator",\n'
          ' "metric": "other",\n "quote": "10.8549%",\n "')


class GroqError(Exception):
    status_code = 400

    def __init__(self, gen):
        super().__init__("Error code: 400 - tool_use_failed")
        self.body = {"error": {"message": "Failed to parse tool call arguments as JSON", "type":
                               "invalid_request_error", "code": "tool_use_failed", "failed_generation": gen}}


def test_salvage_keeps_complete_items_and_marks_partial():
    out = llm.salvage(FAILED, D.FIGURE_TOOL)
    assert out["_partial"] is True
    assert [f["label"] for f in out["figures"]] == [
        "Environmental Response Costs to be Recovered (ERC)", "Combined Tax Rate", "RRAC Reconciliation Adjustment"]
    assert out["figures"][0]["sourceid"] == 4                 # duplicate / misspelled keys survive json.loads
    ok = llm.salvage(json.dumps({"name": "x", "arguments": {"figures": [{"a": 1}]}}), D.FIGURE_TOOL)
    assert ok == {"figures": [{"a": 1}]}


def test_adapter_turns_the_400_into_usable_output_and_retries_the_rest():
    text = ("Line | Item | Amount\n1 | Environmental Response Costs to be Recovered (ERC) | 959,029\n"
            "2 | Combined Tax Rate | 27.320%\n3 | RRAC Reconciliation Adjustment | 807,740\n"
            "4 | Small Commercial and Industrial Allocator | 10.8549%\n")
    chunks = [ingest.Chunk(text, "raf.pdf", "p. 4", kind="table"),
              ingest.Chunk("Residential Allocator 65.6741%", "raf.pdf", "p. 5", kind="table")]
    calls = []

    class Comp:
        def create(self, **kw):
            calls.append(kw)
            if len(calls) == 1:
                raise GroqError(FAILED.replace('"sourceid":4', '"sourceid":1').replace('"source_id": 4', '"source_id": 1'))
            args = {"figures": [{"metric": "other", "label": "Allocator", "value": 65.6741, "unit": "%",
                                 "quote": "Residential Allocator 65.6741%", "source_id": 1}]} if "65.6741" in \
                kw["messages"][-1]["content"] else {"figures": []}
            call = NS(function=NS(name="record_figures", arguments=json.dumps(args)))
            return NS(choices=[NS(message=NS(content=None, tool_calls=[call]))])

    client = llm.OpenAICompat("groq", "k", client=NS(chat=NS(completions=Comp())))
    figs = D.extract_figures(client, "openai/gpt-oss-120b", chunks, max_chars=10_000, max_tokens=3500)
    assert len(calls) == 3                                     # first call cut off -> batch redone in two halves
    vals = {f.label: (f.value, f.verified) for f in figs}
    assert vals["Environmental Response Costs to be Recovered (ERC)"] == (959029, True)   # "sourceid" accepted
    assert vals["Combined Tax Rate"] == (27.32, True)
    assert vals["Allocator"] == (65.6741, True)                # recovered on the retry


def test_unrecoverable_error_still_reported_plainly():
    msg = llm.describe_error(GroqError("garbage"), "groq")
    assert "could not be read" in msg
