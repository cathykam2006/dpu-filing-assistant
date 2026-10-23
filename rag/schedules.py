"""Numbered-line rate schedules, as filed in DPU tariff and factor filings.

    Line  Rate Schedule  R-1     R-2     R-3   ...  Reference
    5     EES = ...      $0.3299 $0.3299 ...        Section III
    19    LDAF = ...     $0.7434 $0.5718 ...        Sum of Line 5 through Line 18
    22    Total Peak Rate (Line 3 + Line 19 + Line 20 + Line 21)  $2.9417 ...

PDF text extraction keeps each line on one row but loses the grid, so this
module rebuilds it: the column header (rate classes), then each numbered line's
label, one value per column, and its reference. The formulas written in the
schedule ("Sum of Line 5 through Line 18", "Line 3 + Line 19 + ...") are then
checked for every column with plain arithmetic.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

CLASS = re.compile(r"\b(?:[A-Z]{1,3}-\d{1,3}[A-Z]?)\b")
# a value: $1.2345, $ 0.35, ($0.0021), -0.5, 1,234,567, 12.5%
VALUE = re.compile(r"\(\s?\$?\s?\d(?:[\d,]*\d)?(?:\.\d+)?\s?\)|-?\$\s?\d(?:[\d,]*\d)?(?:\.\d+)?|-?\b\d(?:[\d,]*\d)?\.\d+%?|\b\d{1,3}(?:,\d{3})+\b")
ROW = re.compile(r"^\s*(\d{1,3})\s+(\S.*)$")


@dataclass
class Row:
    line: int
    label: str
    values: list[float]
    raw: list[str]
    reference: str
    text: str                  # the row as filed


@dataclass
class Schedule:
    columns: list[str]
    rows: dict[int, Row] = field(default_factory=dict)
    header: str = ""

    def cell(self, line: int, column: str) -> float | None:
        r = self.rows.get(line)
        if r is None or column not in self.columns:
            return None
        j = self.columns.index(column)
        return r.values[j] if j < len(r.values) else None

    def column_of(self, value: float, tol: float = 1e-9) -> list[tuple[int, str]]:
        """(line, column) cells holding this value (sign ignored)."""
        return [(ln, self.columns[j]) for ln, r in self.rows.items()
                for j, v in enumerate(r.values) if j < len(self.columns) and abs(abs(v) - abs(value)) <= tol]


def to_number(tok: str) -> float:
    t = tok.strip()
    neg = t.startswith("(") or t.startswith("-")
    t = re.sub(r"[()$,%\s-]", "", t)
    v = float(t)
    return -v if neg else v


def header_line(text: str) -> str | None:
    """The line listing rate classes (at least 3), e.g. 'No. Rate Schedule R-1 R-2 R-3 ...'."""
    best, n = None, 2
    for ln in text.splitlines():
        k = len(CLASS.findall(ln))
        if k > n and not VALUE.search(ln):
            best, n = ln.strip(), k
    return best


def _value_runs(s: str) -> list[list[re.Match]]:
    """Groups of value tokens separated only by whitespace."""
    runs, cur = [], []
    for m in VALUE.finditer(s):
        if cur and s[cur[-1].end():m.start()].strip():
            runs.append(cur)
            cur = []
        cur.append(m)
    if cur:
        runs.append(cur)
    return runs


def parse(text: str) -> Schedule | None:
    head = header_line(text)
    if not head:
        return None
    cols = CLASS.findall(head)
    sch = Schedule(columns=cols, header=head)
    n = len(cols)
    for ln in text.splitlines():
        m = ROW.match(ln)
        if not m:
            continue
        line_no, rest = int(m.group(1)), m.group(2)
        run = next((r for r in _value_runs(rest) if len(r) >= n), None)
        if run is None:
            continue
        run = run[:n]
        try:
            values = [to_number(t.group(0)) for t in run]
        except ValueError:
            continue
        label = rest[:run[0].start()].strip()
        reference = rest[run[-1].end():].strip()
        if line_no in sch.rows:            # same line twice (chunk overlap): keep the first
            continue
        sch.rows[line_no] = Row(line_no, label, values, [t.group(0) for t in run], reference, ln.strip())
    return sch if len(sch.rows) >= 2 else None


# --------------------------------------------------------------------------
# formulas
# --------------------------------------------------------------------------
@dataclass
class FormulaCheck:
    line: int
    column: str
    formula: str
    expected: float
    filed: float
    diff: float
    tolerance: float
    status: str               # ok | rounding | mismatch


def formulas(row: Row) -> list[tuple[str, list[tuple[int, int]]]]:
    """[(formula text, [(sign, line), ...])] written in the row's label or reference."""
    text = f"{row.label} {row.reference}"
    out = []
    for m in re.finditer(r"Sum of Lines?\s*(\d+)\s*(?:through|thru|to|-|–)\s*(?:Lines?\s*)?(\d+)", text, re.I):
        a, b = int(m.group(1)), int(m.group(2))
        if a < b <= 200:
            out.append((m.group(0), [(1, i) for i in range(a, b + 1)]))
    for m in re.finditer(r"Lines?\s*\d+(?:\s*[+\-−]\s*Lines?\s*\d+)+", text, re.I):
        expr = m.group(0)
        terms = [(-1 if s and s in "-−" else 1, int(n)) for s, n in re.findall(r"([+\-−]?)\s*Lines?\s*(\d+)", expr)]
        out.append((expr, terms))
    return out


def _decimals(raw: str) -> int:
    m = re.search(r"\.(\d+)", raw)
    return len(m.group(1)) if m else 0


def check_formulas(sch: Schedule) -> list[FormulaCheck]:
    checks = []
    for row in sch.rows.values():
        for text, terms in formulas(row):
            if any(ln not in sch.rows for _, ln in terms) or row.line in [ln for _, ln in terms]:
                continue
            for j, col in enumerate(sch.columns):
                if j >= len(row.values):
                    continue
                parts = [(s, sch.rows[ln]) for s, ln in terms]
                if any(j >= len(r.values) for _, r in parts):
                    continue
                expected = sum(s * r.values[j] for s, r in parts)
                filed = row.values[j]
                d = max([_decimals(row.raw[j])] + [_decimals(r.raw[j]) for _, r in parts])
                # each filed figure is rounded to d decimals: allow half a unit per term
                tol = 0.5 * 10 ** -d * (len(parts) + 1) + 1e-12
                diff = filed - expected
                status = "ok" if abs(diff) < 0.5 * 10 ** -d else ("rounding" if abs(diff) <= tol else "mismatch")
                checks.append(FormulaCheck(row.line, col, text, round(expected, d + 2), filed, round(diff, d + 2),
                                           tol, status))
    return checks
