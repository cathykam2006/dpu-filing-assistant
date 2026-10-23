"""Measure whether the assistant gets filing questions right.

A question set is a CSV you can edit in Excel, one row per question:

    id, question, files, expected_answer, expected_values, expected_location, category

* files              file names the question is about, separated by ";"
* expected_values    the figures a correct answer must contain, separated by ";"
                     (e.g. "0.9120; 0.8455"). Use NOT_IN_FILING when the right
                     answer is "the filings don't say".
* expected_location  where the answer is, separated by ";" if there are several
                     (e.g. "p. 2" or "DPU-1-2" or "rows 5-10"). Matched against
                     the citation text.

Three things are scored for each question:

1. Retrieval   did search put the right passage in front of the model?
               (hit@k on location, and whether every expected value was in the
               retrieved text)
2. Figures     does the answer contain every expected value, as filed?
3. Citations   does at least one cited source match the expected location and
               contain an expected value?

The same questions can be run against a "whole document" baseline: the full
text pasted into Claude with a plain prompt, the way someone would use a chat
app with file upload. That is the comparison that says whether retrieval and
the citation rules add anything.
"""
from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field

import pandas as pd

from . import answer, search
from .ingest import Chunk

NOT_IN_FILING = "NOT_IN_FILING"
ABSTAIN = re.compile(r"(do(es)?|did) not (contain|include|provide|state|specify|mention|address)|"
                     r"(is|are) not (provided|included|stated|specified|found|available|in the)|"
                     r"no (information|mention|data)|not (found|addressed)|n't (contain|include|say|"
                     r"provide|state|specify)|cannot (find|determine)|unable to (find|determine)", re.I)
BASELINE_SYSTEM = "Answer the question using the documents provided."
BASELINE_CHARS = 400_000   # ~100k tokens


@dataclass
class Question:
    id: str
    question: str
    files: list[str]
    expected_answer: str
    expected_values: list[str]
    expected_location: list[str]
    category: str = ""

    @property
    def abstain(self) -> bool:
        return self.expected_values == [NOT_IN_FILING]


@dataclass
class Result:
    id: str
    category: str
    question: str
    expected: str
    retrieval_hit: bool | None = None
    values_retrieved: bool | None = None
    answer: str = ""
    figures_correct: bool | None = None
    citation_correct: bool | None = None
    baseline_answer: str = ""
    baseline_correct: bool | None = None
    missing_values: list[str] = field(default_factory=list)
    cited: str = ""


# --------------------------------------------------------------------------
# loading questions
# --------------------------------------------------------------------------
def _split(s) -> list[str]:
    return [p.strip() for p in str(s or "").split(";") if p.strip() and p.strip().lower() != "nan"]


def load_questions(source) -> list[Question]:
    """source: path, bytes, or a DataFrame."""
    if isinstance(source, pd.DataFrame):
        df = source
    elif isinstance(source, (bytes, bytearray)):
        df = pd.read_csv(io.BytesIO(source), dtype=str)
    else:
        df = pd.read_csv(source, dtype=str)
    df = df.fillna("")
    need = {"id", "question", "files", "expected_values"}
    if missing := need - set(df.columns):
        raise ValueError(f"Question file is missing columns: {', '.join(sorted(missing))}")
    return [Question(r["id"], r["question"], _split(r["files"]), r.get("expected_answer", ""),
                     _split(r["expected_values"]), _split(r.get("expected_location", "")),
                     r.get("category", "")) for _, r in df.iterrows()]


# --------------------------------------------------------------------------
# scoring helpers
# --------------------------------------------------------------------------
NUM = re.compile(r"-?\$?\d[\d,]*\.?\d*")


def numbers_in(text: str) -> list[float]:
    out = []
    for m in NUM.findall(text.replace("−", "-")):
        s = m.replace("$", "").replace(",", "").rstrip(".")
        try:
            out.append(float(s))
        except ValueError:
            pass
    return out


def value_found(expected: str, text: str) -> bool:
    """Numbers match numerically (0.9120 == 0.912, 3,412,580 == 3412580); words match as text."""
    e = expected.replace("$", "").replace(",", "").replace("%", "").strip()
    try:
        v = float(e)
    except ValueError:
        return expected.lower() in text.lower()
    return any(abs(x - v) <= 1e-9 + 1e-9 * abs(v) for x in numbers_in(text))


def location_match(chunk: Chunk, locations: list[str]) -> bool:
    cite = chunk.citation().lower()
    return any(loc.lower() in cite for loc in locations)


def score_answer(q: Question, text: str) -> tuple[bool, list[str]]:
    if q.abstain:
        return bool(ABSTAIN.search(text)), []
    missing = [v for v in q.expected_values if not value_found(v, text)]
    return not missing, missing


# --------------------------------------------------------------------------
# running
# --------------------------------------------------------------------------
def complete(client, model: str, system: str, user: str, max_tokens: int = 1500) -> str:
    resp = client.messages.create(model=model, max_tokens=max_tokens, system=system,
                                  messages=[{"role": "user", "content": user}])
    return "".join(getattr(b, "text", "") for b in resp.content)


def run(questions: list[Question], chunks: list[Chunk], client=None, model: str = "claude-sonnet-5-5",
        k: int = 8, baseline: bool = False, force_ranking: bool = True, progress=None,
        max_chars: int | None = None, max_tokens: int = 1500,
        baseline_chars: int = BASELINE_CHARS) -> list[Result]:
    """Run every question. Without a client only retrieval is scored (free).
    max_chars / baseline_chars cap what is sent, for models with small token budgets."""
    index = search.Index(chunks)
    saved = search.WHOLE_CORPUS_CHARS
    results = []
    try:
        for n, q in enumerate(questions, start=1):
            if progress:
                progress(n, len(questions), q)
            r = Result(q.id, q.category, q.question,
                       NOT_IN_FILING if q.abstain else "; ".join(q.expected_values))
            files = [f for f in q.files if any(c.file == f for c in chunks)] or None

            # retrieval is scored on ranked search, so small sample files can't pass trivially
            search.WHOLE_CORPUS_CHARS = 0 if force_ranking else saved
            hits = index.search(q.question, k=k, files=files)
            search.WHOLE_CORPUS_CHARS = saved
            text = "\n".join(c.text for c, _ in hits)
            if q.expected_location:
                r.retrieval_hit = any(location_match(c, q.expected_location) for c, _ in hits)
            if not q.abstain:
                r.values_retrieved = all(value_found(v, text) for v in q.expected_values)

            if client is not None:
                live_hits = index.search(q.question, k=k, files=files, max_chars=max_chars)  # as in the app
                gen, sources = answer.stream_answer(client, model, q.question, live_hits, [], max_tokens)
                r.answer = "".join(gen)
                r.figures_correct, r.missing_values = score_answer(q, r.answer)
                cited = [sources[i - 1] for i in answer.cited_ids(r.answer) if 0 < i <= len(sources)]
                r.cited = "; ".join(c.citation() for c in cited)
                if q.abstain:
                    r.citation_correct = None
                else:
                    r.citation_correct = any(
                        (not q.expected_location or location_match(c, q.expected_location))
                        and any(value_found(v, c.text) for v in q.expected_values) for c in cited)
                if baseline:
                    pool = [c for c in chunks if not files or c.file in files]
                    docs = "\n\n".join(f"--- {c.citation()} ---\n{c.text}" for c in pool)[:baseline_chars]
                    r.baseline_answer = complete(client, model, BASELINE_SYSTEM,
                                                 f"Documents:\n\n{docs}\n\nQuestion: {q.question}", max_tokens)
                    r.baseline_correct, _ = score_answer(q, r.baseline_answer)
            results.append(r)
    finally:
        search.WHOLE_CORPUS_CHARS = saved
    return results


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------
def results_frame(results: list[Result]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({"ID": r.id, "Category": r.category, "Question": r.question, "Expected": r.expected,
                     "Retrieval hit": r.retrieval_hit, "Values retrieved": r.values_retrieved,
                     "Figures correct": r.figures_correct, "Citation correct": r.citation_correct,
                     "Baseline correct": r.baseline_correct,
                     "Missing values": "; ".join(r.missing_values), "Cited": r.cited,
                     "Answer": r.answer, "Baseline answer": r.baseline_answer})
    return pd.DataFrame(rows)


def rate(series: pd.Series) -> tuple[int, int]:
    s = series.dropna()
    return int(s.astype(bool).sum()), len(s)


def summary(df: pd.DataFrame) -> pd.DataFrame:
    labels = {"Retrieval hit": "Right passage retrieved (hit@k)",
              "Values retrieved": "All expected figures in retrieved text",
              "Figures correct": "Answer correct (assistant)",
              "Citation correct": "Citation points to the right place",
              "Baseline correct": "Answer correct (whole-document baseline)"}
    rows = []
    for col, label in labels.items():
        ok, n = rate(df[col]) if col in df else (0, 0)
        if n:
            rows.append({"Measure": label, "Score": f"{ok}/{n}", "Rate": f"{ok / n:.0%}"})
    return pd.DataFrame(rows)


def report_markdown(df: pd.DataFrame, title: str = "Evaluation report", notes: str = "") -> str:
    lines = [f"# {title}", ""]
    if notes:
        lines += [notes, ""]
    s = summary(df)
    lines += ["| Measure | Score | Rate |", "|---|---|---|"]
    lines += [f"| {r.Measure} | {r.Score} | {r.Rate} |" for r in s.itertuples()]
    if df["Category"].astype(bool).any() and df["Figures correct"].notna().any():
        lines += ["", "## By category", "", "| Category | Assistant | Baseline |", "|---|---|---|"]
        for cat, g in df.groupby("Category"):
            a, n = rate(g["Figures correct"])
            b, m = rate(g["Baseline correct"])
            lines.append(f"| {cat} | {a}/{n} | {f'{b}/{m}' if m else '–'} |")
    fails = df[(df["Figures correct"] == False) | (df["Retrieval hit"] == False)]  # noqa: E712
    if len(fails):
        lines += ["", "## Misses", ""]
        for _, r in fails.iterrows():
            why = []
            if r["Retrieval hit"] is False:
                why.append("retrieval missed the passage")
            if r["Figures correct"] is False:
                why.append(f"answer missing {r['Missing values'] or 'the expected abstention'}")
            lines.append(f"- **{r['ID']}** {r['Question']} — {'; '.join(why)}")
    return "\n".join(lines) + "\n"


MANUAL_COLUMNS = ["id", "question", "expected_values", "tool", "answer_pasted", "correct (y/n)",
                  "cited the right page (y/n)", "notes"]


def manual_template(questions: list[Question], tools=("ChatGPT file upload", "Claude.ai file upload")) -> str:
    """CSV for scoring other tools by hand on the same questions."""
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(MANUAL_COLUMNS)
    for q in questions:
        for t in tools:
            w.writerow([q.id, q.question, "; ".join(q.expected_values), t, "", "", "", ""])
    return buf.getvalue()
