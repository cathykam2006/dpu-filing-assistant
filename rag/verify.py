"""Checks that make mistakes and misses visible. None of this uses AI.

1. Formula checks   Every formula written in a schedule ("Sum of Line 5 through
                    Line 18", "Line 3 + Line 19 + Line 20 + Line 21") is
                    recomputed for every rate class.
2. Coverage         Every number in the filing that is NOT in the figures table
                    is listed with its page, line and column, so nothing the
                    extraction skipped goes unnoticed.
3. Answer check     Every number in a chat answer is looked up in the source
                    the answer cites. If the sentence names a rate class and
                    the source is a schedule, the number must sit in that
                    class's column. Anything else is flagged.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from . import schedules as S
from .discovery import Figure, Issue
from .ingest import Chunk, Document

# numbers worth checking: money, decimals, thousands, percentages (not years, line or page numbers)
NUMBER = re.compile(r"\(\s?\$?\s?\d(?:[\d,]*\d)?(?:\.\d+)?\s?\)|-?\$\s?\d(?:[\d,]*\d)?(?:\.\d+)?|-?\b\d(?:[\d,]*\d)?\.\d+\s?%?|"
                    r"\b\d{1,3}(?:,\d{3})+\b|\b\d+\s?%")


def _val(tok: str) -> float | None:
    try:
        return S.to_number(tok.replace("%", ""))
    except ValueError:
        return None


def _same(a: float, b: float) -> bool:
    return abs(abs(a) - abs(b)) <= 1e-9 + 1e-9 * abs(b)


def page_texts(doc: Document) -> list[tuple[str, str]]:
    """Full page texts when the reader kept them (PDF), else the chunks."""
    if doc.page_texts:
        return doc.page_texts
    return [(c.location, c.text) for c in doc.chunks]


def document_schedules(doc: Document) -> list[tuple[str, S.Schedule]]:
    out = []
    for loc, text in page_texts(doc):
        sch = S.parse(text)
        if sch:
            out.append((loc, sch))
    return out


# --------------------------------------------------------------------------
# 1. formula checks
# --------------------------------------------------------------------------
def formula_checks(docs: list[Document]) -> pd.DataFrame:
    rows = []
    for d in docs:
        for loc, sch in document_schedules(d):
            for c in S.check_formulas(sch):
                rows.append({"File": d.name, "Where": loc, "Line": c.line, "Rate class": c.column,
                             "Formula": c.formula, "Filed": c.filed, "Recomputed": c.expected,
                             "Difference": c.diff, "Result": {"ok": "✓ ties", "rounding": "≈ rounding",
                                                             "mismatch": "✗ does not tie"}[c.status]})
    return pd.DataFrame(rows, columns=["File", "Where", "Line", "Rate class", "Formula", "Filed", "Recomputed",
                                       "Difference", "Result"])


def formula_issues(checks: pd.DataFrame) -> list[Issue]:
    bad = checks[checks["Result"] == "✗ does not tie"]
    issues = []
    for (file, where, line, formula), g in bad.groupby(["File", "Where", "Line", "Formula"]):
        cols = ", ".join(f"{r['Rate class']} (filed {r['Filed']:g}, recomputed {r['Recomputed']:g})"
                         for _, r in g.iterrows())
        issues.append(Issue("formula", "high", f"Line {line} does not equal {formula} for {cols}.",
                            file, [where]))
    return issues


# --------------------------------------------------------------------------
# schedule cells as figures (no-AI mode)
# --------------------------------------------------------------------------
def _metric(label: str) -> str:
    low = label.lower()
    if re.search(r"\bldaf\b|local distribution adjustment factor", low):
        return "ldaf"
    if re.search(r"\bgaf\b|gas adjustment factor", low):
        return "gaf"
    return "other"


def schedule_figures(doc: Document) -> list[Figure]:
    figs = []
    for loc, sch in document_schedules(doc):
        for r in sch.rows.values():
            for j, col in enumerate(sch.columns[:len(r.values)]):
                unit = "$/month" if re.search(r"per month|customer charge", r.label, re.I) else (
                    "$/therm" if r.raw[j].count(".") and len(r.raw[j].split(".")[-1].rstrip(")")) >= 4 else "$")
                figs.append(Figure(metric=_metric(r.label), label=f"Line {r.line}: {r.label}"[:90],
                                   value=r.values[j], unit=unit, rate_class=col, quote=r.text[:300],
                                   file=doc.name, location=loc, verified=True))
    return figs


# --------------------------------------------------------------------------
# 2. coverage
# --------------------------------------------------------------------------
def coverage(docs: list[Document], figs: list[Figure]) -> pd.DataFrame:
    """Every number in the filings, and whether the figures table has it."""
    by_file: dict[str, list[Figure]] = {}
    for f in figs:
        by_file.setdefault(f.file, []).append(f)
    rows = []
    for d in docs:
        mine = by_file.get(d.name, [])
        cells_done = {(f.rate_class.upper(), round(abs(f.value), 9)) for f in mine if f.rate_class}
        values_done = {round(abs(f.value), 9) for f in mine}
        for loc, text in page_texts(d):
            sch = S.parse(text)
            if sch:
                for r in sch.rows.values():
                    for j, col in enumerate(sch.columns[:len(r.values)]):
                        ok = (col.upper(), round(abs(r.values[j]), 9)) in cells_done
                        rows.append({"File": d.name, "Where": loc, "Line / column": f"Line {r.line} · {col}",
                                     "Number": r.raw[j], "Context": r.label[:80], "In figures table": ok})
            for ln in text.splitlines():
                if sch and any(ln.strip() == r.text for r in sch.rows.values()):
                    continue
                for m in NUMBER.finditer(ln):
                    v = _val(m.group(0))
                    if v is None:
                        continue
                    before = ln[max(0, m.start() - 12):m.start()]
                    if re.search(r"(Line|Page|No\.|D\.P\.U\.|CMR|G\.\s?L\.|c\.|§|Section)\s*$", before):
                        continue
                    ctx = ln[max(0, m.start() - 50):m.end() + 30].strip()
                    rows.append({"File": d.name, "Where": loc, "Line / column": "", "Number": m.group(0).strip(),
                                 "Context": ctx, "In figures table": round(abs(v), 9) in values_done})
    return pd.DataFrame(rows, columns=["File", "Where", "Line / column", "Number", "Context", "In figures table"])


# --------------------------------------------------------------------------
# 3. numbers in an answer
# --------------------------------------------------------------------------
@dataclass
class NumberCheck:
    number: str
    status: str            # verified | wrong column | other source | not found
    detail: str
    sentence: str


CITE = re.compile(r"\[S(\d+)\]")


def _segments(answer: str) -> list[str]:
    segs = []
    for block in re.split(r"\n\s*\n", answer):
        lines = block.splitlines()
        if sum("|" in ln for ln in lines) >= 2:          # markdown table: one row per segment
            segs += [ln for ln in lines if "|" in ln and not re.fullmatch(r"[\s|:\-]+", ln)]
        else:
            segs += re.split(r"(?<=[.!?])\s+(?=[A-Z*(])", block)
    return [s for s in segs if s.strip()]


def check_answer(answer: str, sources: list[Chunk]) -> list[NumberCheck]:
    all_ids = [int(i) for i in CITE.findall(answer)]
    checks = []
    paragraphs = re.split(r"\n\s*\n", answer)
    for seg in _segments(answer):
        text = CITE.sub(" ", seg)
        ids = [int(i) for i in CITE.findall(seg)]
        if not ids:                                       # fall back to the paragraph's, then the answer's
            para = next((p for p in paragraphs if seg in p), "")
            ids = [int(i) for i in CITE.findall(para)] or all_ids
        cited = [sources[i - 1] for i in dict.fromkeys(ids) if 0 < i <= len(sources)]
        mentions = [(c.start(), c.group(0)) for c in S.CLASS.finditer(text)]
        for m in NUMBER.finditer(text):
            tok = m.group(0).strip()
            v = _val(tok)
            if v is None:
                continue
            # the rate class this number belongs to: the closest mention, preferring one before it
            classes = set()
            if mentions:
                classes = {min(mentions, key=lambda c: (m.start() - c[0]) if c[0] <= m.start()
                               else 10_000 + c[0] - m.start())[1]}
            hit = [c for c in cited if any(_same(x, v) for x in (_val(t) for t in NUMBER.findall(c.text)) if x is not None)]
            if hit:
                status, detail = "verified", hit[0].citation()
                if classes:
                    cols = {col for c in hit for sch in [S.parse(c.text)] if sch for _, col in sch.column_of(v)}
                    if cols and not cols & classes:
                        status = "wrong column"
                        detail = (f"{hit[0].citation()}: this value is under {', '.join(sorted(cols))}, "
                                  f"not {', '.join(sorted(classes))}")
            else:
                other = [c for c in sources if c not in cited and any(
                    _same(x, v) for x in (_val(t) for t in NUMBER.findall(c.text)) if x is not None)]
                if other:
                    status, detail = "other source", f"found in {other[0].citation()}, which this sentence does not cite"
                else:
                    status, detail = "not found", "not in any source; if it is a calculation, check the arithmetic"
            checks.append(NumberCheck(tok, status, detail, seg.strip()[:200]))
    return checks


def answer_ok(checks: list[NumberCheck]) -> bool:
    return all(c.status == "verified" for c in checks)
