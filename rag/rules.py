"""No-AI mode: the discovery tools rebuilt from patterns and templates.

Used when no model key is configured, so the demo still works for anyone.
* Figures come from table rows ("Commodity | 98,300,000") and from sentences
  matched by keyword ("... LNG peaking supply of $4,817,420 ..."). Each figure
  is checked against its page with the same verify() the AI path uses.
* Information requests are templates filled from the automatic checks and
  the figures found (reconcile this total, explain this change, provide
  workpapers for this component).
* Responses are checked for objections, deferrals, attachments not uploaded
  and very short answers.
These are dependable on structured tables and standard wording, and weaker
than a model on free-form prose.
"""
from __future__ import annotations

import re

import pandas as pd

from .discovery import (GRADE_COLS, Figure, Issue, METRICS, add_unanswered, dedupe, ir_groups,
                        missing_attachments, verify)
from .ingest import Chunk
from .schedules import parse as parse_schedule

# keyword -> metric, checked in order (first match wins)
LABEL_METRICS = [
    (r"\btotal\b.*(cost|gas)|^total$|\btotal\b", "cost_of_gas_total"),
    (r"commodity", "commodity_cost"),
    (r"pipeline|storage|demand charge|capacity|\bdemand\b", "demand_charges"),
    (r"\blng\b|peaking|liquefied", "lng_cost"),
    (r"reconcil|under-?collection|over-?collection|true-?up", "reconciliation"),
    (r"revenue requirement|revenue (increase|deficiency)", "revenue_requirement"),
]
SALES = re.compile(r"(?:sales|throughput|deliveries)\D{0,60}?(\d{1,3}(?:,\d{3}){2,})\s*(therms|dth|mmbtu|ccf)", re.I)
MONEY = re.compile(r"\$\s?(\d{1,3}(?:,\d{3})+(?:\.\d+)?)")
RATE = re.compile(r"\$\s?(\d?\.\d{3,4})\s*(?:per|/)\s*therm", re.I)
PCT = re.compile(r"(\d{1,2}(?:\.\d+)?)\s?%")
SMALL_MONEY = re.compile(r"\$\s?(\d{1,4}\.\d{2})\b")


def _sentences(text: str) -> list[str]:
    flat = re.sub(r"\s+", " ", text)
    return [s.strip() for s in re.split(r"(?<=[.;?])\s+(?=[A-Z(])|\s(?=Response:|Request:)", flat)
            if s.strip()]


def _quote(s: str, m: re.Match, width: int = 190) -> str:
    """The sentence, or a window of it around the number when the sentence is long."""
    if len(s) <= width:
        return s
    start = max(0, min(m.start() - width // 2, len(s) - width))
    return s[start:start + width]


def _clean(metric: str) -> str:
    return METRICS[metric].split(" (")[0]


def _num(s: str) -> float:
    return float(s.replace(",", "").replace("$", ""))


def _metric_for(label: str) -> str | None:
    low = label.lower()
    for pat, m in LABEL_METRICS:
        if re.search(pat, low):
            return m
    return None


def _rate_class(text: str) -> str:
    m = re.search(r"\(([A-Z]{1,3}-\d{1,3}[A-Z]?)\)", text) or re.search(r"\b([RG]-\d{1,3})\b", text)
    if m:
        return m.group(1)
    for rc in ("residential heating", "residential non-heating", "residential", "c&i", "commercial"):
        if rc in text.lower():
            return rc
    return ""


def _fig(chunk: Chunk, metric: str, label: str, value: float, unit: str, quote: str,
         rate_class: str = "", status: str = "") -> Figure:
    f = Figure(metric=metric, label=label[:80], value=value, unit=unit, rate_class=rate_class, status=status,
               quote=quote[:200], file=chunk.file, location=chunk.location)
    f.verified = verify(f, chunk.text)
    return f


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def _from_table(c: Chunk) -> list[Figure]:
    lines = [ln for ln in c.text.splitlines() if "|" in ln]
    if len(lines) < 2:
        return []
    header = [h.strip().lower() for h in lines[0].split("|")]
    is_rate = any(re.search(r"\b(gaf|ldaf)\b|per therm|/therm", h) for h in header)
    metric_rate = "ldaf" if any("ldaf" in h for h in header) else "gaf"
    out = []
    for ln in lines[1:]:
        cells = [x.strip() for x in ln.split("|")]
        label = cells[0]
        if not label:
            continue
        for j, cell in enumerate(cells[1:], start=1):
            if not re.fullmatch(r"\$?\d[\d,]*\.?\d*", cell):
                continue
            h = header[j] if j < len(header) else ""
            if "change" in h or "difference" in h or "%" in h:
                continue
            quote = " | ".join(cells[:j + 1])
            if is_rate:
                status = "proposed" if "proposed" in h or "new" in h else (
                    "prior" if re.search(r"prior|current|previous|20\d\d", h) else "")
                out.append(_fig(c, metric_rate, label, _num(cell), "$/therm", quote, _rate_class(label), status))
            else:
                m = _metric_for(label)
                if m:
                    out.append(_fig(c, m, label, _num(cell), "$", quote))
                break                                    # one value per component row
    return out


def _from_prose(c: Chunk) -> list[Figure]:
    out = []
    for s in _sentences(c.text):
        low = s.lower()
        # per-therm rates: GAF / LDAF, proposed vs prior by the words just before the number
        if re.search(r"\bgaf\b|gas adjustment factor|\bldaf\b|local distribution adjustment", low):
            metric = "ldaf" if re.search(r"\bldaf\b|local distribution adjustment", low) else "gaf"
            for m in RATE.finditer(s):
                before = low[max(0, m.start() - 45):m.start()]
                if re.search(r"increase of|decrease of|change of|by \$?$", before):
                    continue
                status = "prior" if re.search(r"compared|from|prior|previous|last|current", before) else (
                    "proposed" if "propos" in low else "")
                out.append(_fig(c, metric, _clean(metric), _num(m.group(1)), "$/therm", _quote(s, m),
                                _rate_class(s), status))
        # dollar amounts: label by the closest keyword before the number
        for m in MONEY.finditer(s):
            before = low[max(0, m.start() - 70):m.start()]
            metric = None
            best = -1
            for pat, met in LABEL_METRICS:
                for km in re.finditer(pat, before):
                    if km.start() > best:
                        best, metric = km.start(), met
            if re.search(r"\bbill\b", before):
                metric = "bill_impact_dollars"
            if metric:
                out.append(_fig(c, metric, _clean(metric), _num(m.group(1)), "$", _quote(s, m)))
        # sales volumes
        for m in SALES.finditer(s):
            out.append(_fig(c, "forecast_sales", "Forecast sales", _num(m.group(1)), m.group(2).lower(),
                            _quote(s, m)))
        # bill impacts in dollars and percent
        if re.search(r"\bbill\b", low):
            for m in SMALL_MONEY.finditer(s):
                before = low[max(0, m.start() - 20):m.start()]
                if re.search(r"(increase|decrease|change|impact|rise)( of| by)?\s*$|\bby\s*$", before):
                    out.append(_fig(c, "bill_impact_dollars", "Typical bill change", _num(m.group(1)), "$",
                                    _quote(s, m)))
            for m in PCT.finditer(s):
                out.append(_fig(c, "bill_impact_percent", "Typical bill change", float(m.group(1)), "%",
                                _quote(s, m)))
        if re.search(r"return on equity|\broe\b", low):
            for m in PCT.finditer(s):
                out.append(_fig(c, "return_on_equity", "Return on equity", float(m.group(1)), "%", _quote(s, m)))
    return out


def extract_figures(chunks: list[Chunk]) -> list[Figure]:
    tables, prose = [], []
    for c in chunks:
        if c.kind == "spreadsheet" or parse_schedule(c.text):
            continue          # monthly detail, and numbered-line schedules (read by verify.schedule_figures)
        if c.kind == "table":
            tables += _from_table(c)
        else:
            prose += _from_prose(c)
    # a figure in a table is cleaner than the same number found in a sentence
    in_tables = {(f.file, f.metric, round(f.value, 6)) for f in tables}
    prose = [f for f in prose if (f.file, f.metric, round(f.value, 6)) not in in_tables]
    return dedupe(tables + prose)


# --------------------------------------------------------------------------
# information requests from templates
# --------------------------------------------------------------------------
def _where(i: Issue) -> str:
    return f"Refer to {i.locations[0]} of {i.file}. " if i.locations else (f"Refer to {i.file}. " if i.file else "")


def draft_irs(figs: list[Figure], issues: list[Issue], text: str = "", prefix: str = "DPU",
              set_no: int = 1, n: int = 10) -> pd.DataFrame:
    reqs: list[tuple[str, str, str, str]] = []           # topic, request, why, based on
    for i in issues:
        if i.kind == "tie-out":
            reqs.append(("Cost of gas tie-out", _where(i) + i.description.replace("but the stated", "while the stated")
                         + " Please reconcile the difference and provide corrected schedules if needed.",
                         "Automatic check: components do not sum to the total.", "; ".join(i.locations)))
        elif i.kind == "large-change":
            reqs.append(("Rate change", _where(i) + i.description + " Please identify and quantify each driver "
                         "of the change, with supporting workpapers.",
                         "Automatic check: change above the review threshold.", "; ".join(i.locations)))
        elif i.kind == "missing":
            if "reconciliation" in i.description:
                reqs.append(("Reconciliation", "Please provide the prior-period reconciliation balance, with "
                             "monthly gas costs and revenues, in Excel with formulas intact.",
                             "Not found in the filing.", i.file))
            elif "bill impact" in i.description:
                reqs.append(("Bill impact", "Please provide typical bill impacts for each residential and C&I "
                             "rate class under the proposed factors, by month for the period.",
                             "Not found in the filing.", i.file))
            elif "sales" in i.description:
                reqs.append(("Sales forecast", "Please provide forecast sales by rate class and month, and "
                             "describe the forecasting method and weather assumptions used.",
                             "Not found in the filing.", i.file))
    comps = [f for f in figs if f.metric in ("commodity_cost", "demand_charges", "lng_cost", "reconciliation")]
    done = set()
    for f in sorted(comps, key=lambda f: (-f.value, f.location)):
        if (f.metric, f.value) in done:
            continue
        done.add((f.metric, f.value))
        reqs.append((METRICS[f.metric].split(" (")[0],
                     f"Refer to {f.location} of {f.file}. Please provide workpapers supporting \"{f.label}\" "
                     f"(${f.value:,.0f}), in Excel with formulas intact, and explain the main assumptions.",
                     "Material cost component.", f"{f.file} · {f.location}"))
    if text and not re.search(r"hedg", text, re.I):
        reqs.append(("Hedging", "Please describe the Company's hedging and fixed-price purchases for the "
                     "period, including volumes, prices and their effect on the proposed factor.",
                     "Hedging is not addressed in the filing.", ""))
    if text and re.search(r"nymex|citygate|basis", text, re.I):
        reqs.append(("Commodity forecast", "Please provide the NYMEX and basis price forecasts used to derive "
                     "commodity costs, with the date and source of each.",
                     "Commodity prices drive the cost of gas.", ""))
    seen, rows = set(), []
    for topic, req, why, src in reqs:
        if req in seen:
            continue
        seen.add(req)
        rows.append({"No.": f"{prefix}-{set_no}-{len(rows) + 1}", "Topic": topic, "Request": req,
                     "Why": why, "Based on": src})
        if len(rows) >= n:
            break
    return pd.DataFrame(rows, columns=["No.", "Topic", "Request", "Why", "Based on"])


# --------------------------------------------------------------------------
# response checks
# --------------------------------------------------------------------------
ATTACH = re.compile(r"\b(?:Attachment|Exhibit)\s+[A-Z]{0,5}[-\s]?\d[\w.-]*", re.I)


def grade_responses(chunks: list[Chunk], uploaded_files: list[str],
                    drafted: list[str] | None = None) -> pd.DataFrame:
    groups = ir_groups(chunks)
    rows = []
    for ir, cs in groups.items():
        text = re.sub(r"\s+", " ", " ".join(c.text for c in cs))
        resp = re.split(r"\bResponse:?", text, maxsplit=1, flags=re.I)
        resp = resp[1].strip() if len(resp) > 1 else text
        atts = sorted({re.sub(r"\s+", " ", a).rstrip(".") for a in ATTACH.findall(resp)})
        missing = missing_attachments(atts, uploaded_files)
        low = resp.lower()
        if re.search(r"\bobject(s|ion|ed)?\b", low):
            status, gap = "objection", "The Company objects; review the grounds."
        elif re.search(r"will (supplement|provide|update)|to be provided|forthcoming|not yet available", low):
            status, gap = "partial", "The response defers part of the answer to a later supplement."
        elif atts and len(resp) < 250:
            status, gap = "refers to attachment", "The substance is in the attachment."
        elif len(resp) < 60:
            status, gap = "check", "Very short response; confirm it answers every part of the request."
        else:
            status, gap = "answered (unreviewed)", ""
        follow = (f"Please provide {', '.join(missing)} in native Excel format with formulas intact."
                  if missing else "Please provide the supplemental response and an expected date." if status == "partial"
                  else "")
        summary = resp if len(resp) <= 200 else resp[:197].rsplit(" ", 1)[0] + "…"
        rows.append({"IR": ir, "Status": status, "Summary": summary, "Gaps": gap,
                     "Attachments not uploaded": ", ".join(missing), "Suggested follow-up": follow,
                     "Where": cs[0].citation()})
    return add_unanswered(rows, groups, drafted)


# --------------------------------------------------------------------------
# ask without a model: show the best passages
# --------------------------------------------------------------------------
def highlight(text: str, query: str) -> str:
    """Bold the query's words and numbers in a passage (markdown)."""
    from .search import STOP
    words = {w for w in re.findall(r"[a-z0-9][a-z0-9.,-]{2,}", query.lower()) if w not in STOP}
    if not words:
        return text
    pat = re.compile(r"\b(" + "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True)) + r")\w*",
                     re.I)
    return pat.sub(lambda m: f"**{m.group(0)}**", text)


__all__ = ["extract_figures", "draft_irs", "grade_responses", "highlight", "GRADE_COLS"]
