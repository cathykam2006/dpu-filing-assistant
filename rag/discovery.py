"""Discovery support for public DPU filings.

1. extract_figures  Claude reads the filing and returns every key figure as
                    structured data, each with the exact sentence it came from.
                    Code then verifies the quote and number really appear in the
                    cited source; anything it can't verify is flagged.
2. run_checks       Plain-code review: do components tie to the total? Large
                    year-over-year changes? Missing items (no reconciliation,
                    no bill impact)? Unverified figures?
3. draft_irs        Numbered information requests drawn from those findings,
                    each citing the page that prompted it.
4. grade_responses  For each IR response: answered / partial / non-responsive /
                    objection, missing attachments, suggested follow-up.
5. compare          Prior vs. current filing, figure by figure.

Claude is called through tool use with a JSON schema, so outputs are
structured and machine-checkable rather than free text.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field

import pandas as pd

from .ingest import Chunk

# --------------------------------------------------------------------------
# vocabulary
# --------------------------------------------------------------------------
METRICS = {
    "gaf": "Gas Adjustment Factor ($/therm)",
    "ldaf": "Local Distribution Adjustment Factor ($/therm)",
    "cost_of_gas_total": "Total cost of gas for the period ($)",
    "commodity_cost": "Commodity cost ($)",
    "demand_charges": "Pipeline, storage and capacity (demand) charges ($)",
    "lng_cost": "LNG / peaking supply cost ($)",
    "reconciliation": "Prior-period reconciliation (positive = under-collection) ($)",
    "forecast_sales": "Forecast sales volume (therms or other units)",
    "bill_impact_dollars": "Typical bill change ($)",
    "bill_impact_percent": "Typical bill change (%)",
    "revenue_requirement": "Revenue requirement or requested increase ($)",
    "return_on_equity": "Return on equity (%)",
    "other": "Other material figure",
}
COMPONENTS = ["commodity_cost", "demand_charges", "lng_cost", "reconciliation"]
BIG_CHANGE_PCT = 10.0
TIE_TOLERANCE = 0.005     # 0.5%


@dataclass
class Figure:
    metric: str
    label: str
    value: float
    unit: str
    rate_class: str = ""
    period: str = ""
    status: str = ""          # proposed | current | prior | actual | forecast
    quote: str = ""
    source_id: int = 0
    file: str = ""
    location: str = ""
    verified: bool = False


@dataclass
class Issue:
    kind: str                 # tie-out | large-change | missing | unverified
    severity: str             # high | medium | low
    description: str
    file: str = ""
    locations: list = field(default_factory=list)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _num_variants(v: float) -> list[str]:
    """Ways a number might be written in a filing."""
    out = {f"{v:,.0f}", f"{v:.0f}", f"{v:,.2f}", f"{v:.2f}", f"{v:.4f}", f"{v:.3f}", f"{v:.1f}"}
    if abs(v) < 10:
        out |= {f"{v:.4f}".lstrip("0"), f"{v:.2f}".lstrip("0")}
    return [s for s in out if s]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("−", "-").replace("–", "-")).strip().lower()


def verify(fig: Figure, source_text: str) -> bool:
    """The quote must appear in the source, and the number in the quote."""
    if not fig.quote:
        return False
    src, q = _norm(source_text), _norm(fig.quote)
    if q not in src:
        # tables lose spacing around pipes; compare without spaces
        if q.replace(" ", "") not in src.replace(" ", ""):
            return False
    qn = q.replace("$", "").replace(" ", "")
    return any(var.replace(",", "") in qn.replace(",", "") for var in _num_variants(abs(fig.value)))


def _sources_block(chunks: list[Chunk], start: int = 1) -> str:
    return "\n\n".join(
        f'<source id="{start + i}" file="{c.file}" location="{c.location}"'
        + (f' ir="{c.ir}"' if c.ir else "") + f">\n{c.text}\n</source>"
        for i, c in enumerate(chunks))


def _batches(chunks: list[Chunk], max_chars: int = 60_000, measure=len) -> list[list[Chunk]]:
    """Group chunks so each group stays under max_chars, in units of `measure`."""
    out, cur, n = [], [], 0
    for c in chunks:
        size = measure(c.text)
        if cur and n + size > max_chars:
            out.append(cur)
            cur, n = [], 0
        cur.append(c)
        n += size
    if cur:
        out.append(cur)
    return out


def _call_tool(client, model: str, system: str, user: str, tool: dict, max_tokens: int = 4000) -> dict:
    resp = client.messages.create(model=model, max_tokens=max_tokens, system=system,
                                  messages=[{"role": "user", "content": user}],
                                  tools=[tool], tool_choice={"type": "tool", "name": tool["name"]})
    for block in resp.content:
        if getattr(block, "type", None) == "tool_use":
            return block.input if isinstance(block.input, dict) else json.loads(block.input)
    return {}


def has_numbers(c: Chunk) -> bool:
    return bool(re.search(r"\$\s?\d|\d+\.\d{2,}|\d{1,3}(,\d{3})+|\d+(\.\d+)?\s?%", c.text))


# --------------------------------------------------------------------------
# 1. extraction
# --------------------------------------------------------------------------
FIGURE_TOOL = {
    "name": "record_figures",
    "description": "Record the key figures stated in the filing excerpts.",
    "input_schema": {
        "type": "object",
        "properties": {"figures": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "metric": {"type": "string", "enum": list(METRICS)},
                "label": {"type": "string", "description": "The item name as the filing words it"},
                "value": {"type": "number", "description": "The number exactly as filed (no rounding); "
                                                           "percentages as e.g. 7.9"},
                "unit": {"type": "string", "description": "$/therm, $, therms, %, $/MMBtu, ..."},
                "rate_class": {"type": "string"},
                "period": {"type": "string", "description": "e.g. 2026-27 peak"},
                "status": {"type": "string", "enum": ["proposed", "current", "prior", "actual",
                                                      "forecast", ""]},
                "quote": {"type": "string", "description": "Exact text from the source containing the "
                                                           "number, copied verbatim, under 200 characters"},
                "source_id": {"type": "integer"},
            },
            "required": ["metric", "label", "value", "unit", "quote", "source_id"]}}},
        "required": ["figures"]},
}

EXTRACT_SYSTEM = """You extract figures from public utility filings submitted to the Massachusetts DPU.
Record every material figure in the excerpts: GAF and LDAF rates by rate class (proposed and prior), \
cost-of-gas components and totals, prior-period reconciliation balances, forecast sales, typical bill \
impacts, revenue requirements and returns. Copy each number exactly as filed. The quote must be copied \
verbatim from the source so it can be checked automatically; keep it short (the table row or the phrase \
holding the number, under 120 characters). Do not compute, infer or round. Skip page numbers, dates, docket \
numbers and line numbers."""


def _to_float(v) -> float | None:
    """Open models sometimes return numbers as strings ("$1,234", "7.9%")."""
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(str(v).replace("$", "").replace(",", "").replace("%", "").strip())
    except ValueError:
        return None


def extract_figures(client, model: str, chunks: list[Chunk], max_chars: int = 60_000,
                    max_tokens: int = 4000, measure=len) -> list[Figure]:
    numeric = [c for c in chunks if has_numbers(c)]
    figures: list[Figure] = []
    queue = _batches(numeric, max_chars, measure)
    while queue:
        batch = queue.pop(0)
        out = _call_tool(client, model, EXTRACT_SYSTEM,
                         f"Filing excerpts:\n\n{_sources_block(batch)}", FIGURE_TOOL, max_tokens)
        if out.get("_partial") and len(batch) > 1:      # output was cut off: redo the batch in halves
            half = len(batch) // 2
            queue[:0] = [batch[:half], batch[half:]]
        for f in out.get("figures", []) or []:
            if not isinstance(f, dict):
                continue
            sid = f.get("source_id", f.get("sourceid", f.get("sourceId", 0)))
            sid, val = _to_float(sid), _to_float(f.get("value"))
            if sid is None or val is None or not 1 <= int(sid) <= len(batch):
                continue
            sid = int(sid)
            src = batch[sid - 1]
            metric = f.get("metric", "other")
            fig = Figure(metric=metric if metric in METRICS else "other", label=str(f.get("label", "")),
                         value=val, unit=str(f.get("unit", "")),
                         rate_class=f.get("rate_class", "") or "", period=f.get("period", "") or "",
                         status=f.get("status", "") or "", quote=str(f.get("quote", "")),
                         source_id=sid, file=src.file, location=src.location)
            fig.verified = verify(fig, src.text)
            if not fig.verified:
                # the model sometimes cites the neighbouring source; relocate if the quote is elsewhere
                alt = next((c for c in batch if verify(fig, c.text)), None)
                if alt is not None:
                    fig.file, fig.location, fig.verified = alt.file, alt.location, True
            figures.append(fig)
    return dedupe(figures)


def dedupe(figures: list[Figure]) -> list[Figure]:
    """The same figure is often in a table and the text; keep one, preferring verified."""
    seen, uniq = set(), []
    for f in sorted(figures, key=lambda f: not f.verified):
        key = (f.file, f.metric, f.rate_class.lower(), f.status, round(f.value, 6),
               f.label.lower() if f.metric == "other" else "")
        if key not in seen:
            seen.add(key)
            uniq.append(f)
    return uniq


def figures_frame(figs: list[Figure]) -> pd.DataFrame:
    if not figs:
        return pd.DataFrame(columns=list(Figure.__dataclass_fields__))
    return pd.DataFrame([asdict(f) for f in figs])


# --------------------------------------------------------------------------
# 2. checks (no AI)
# --------------------------------------------------------------------------
def run_checks(figs: list[Figure]) -> list[Issue]:
    issues: list[Issue] = []
    df = figures_frame(figs)
    if df.empty:
        return [Issue("missing", "high", "No figures were extracted from the selected filings.")]
    for file, d in df.groupby("file"):
        # components tie to total
        tot = d[d["metric"] == "cost_of_gas_total"]
        comps = d[d["metric"].isin(COMPONENTS)].drop_duplicates("metric")
        if len(tot) and len(comps) >= 2:
            total = float(tot["value"].iloc[0])
            s = float(comps["value"].sum())
            if total and abs(s - total) / abs(total) > TIE_TOLERANCE:
                issues.append(Issue("tie-out", "high",
                                    f"Cost-of-gas components ({', '.join(comps['label'])}) sum to "
                                    f"${s:,.0f}, but the stated total is ${total:,.0f} "
                                    f"(difference ${s - total:,.0f}).", file,
                                    sorted(set(tot["location"]) | set(comps["location"]))))
        # large changes, proposed vs prior/current, same rate class and metric
        for metric in ("gaf", "ldaf"):
            m = d[d["metric"] == metric]
            for rc, g in m.groupby(m["rate_class"].str.lower()):
                new = g[g["status"] == "proposed"]
                old = g[g["status"].isin(["prior", "current"])]
                if len(new) and len(old):
                    a, b = float(old["value"].iloc[0]), float(new["value"].iloc[0])
                    if a and abs(b / a - 1) * 100 >= BIG_CHANGE_PCT:
                        issues.append(Issue("large-change", "medium",
                                            f"{metric.upper()} for {rc or 'unspecified class'} changes "
                                            f"from {a:.4f} to {b:.4f} ({(b / a - 1) * 100:+.1f}%).", file,
                                            sorted(set(g["location"]))))
        # missing items for a cost-of-gas filing
        if (d["metric"] == "gaf").any() and d["metric"].isin(["cost_of_gas_total", "commodity_cost"]).any():
            for metric, why in (("reconciliation", "No prior-period reconciliation balance found."),
                                ("bill_impact_dollars", "No typical customer bill impact found."),
                                ("forecast_sales", "No forecast sales volume found.")):
                if not (d["metric"] == metric).any():
                    issues.append(Issue("missing", "low", why, file))
        for _, r in d[~d["verified"]].iterrows():
            issues.append(Issue("unverified", "medium",
                                f"Could not confirm \"{r['label']}\" = {r['value']:g} {r['unit']} verbatim "
                                f"in the cited source; check before relying on it.", file, [r["location"]]))
    order = {"high": 0, "medium": 1, "low": 2}
    return sorted(issues, key=lambda i: order[i.severity])


# --------------------------------------------------------------------------
# 3. draft information requests
# --------------------------------------------------------------------------
IR_TOOL = {
    "name": "record_requests",
    "description": "Record draft information requests in the Department's format.",
    "input_schema": {
        "type": "object",
        "properties": {"requests": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "Two to four words"},
                "refer_to": {"type": "string", "description": "What to cite, without the words 'Refer to': the "
                             "document as given in the citations list plus a pinpoint, e.g. \"the Company’s "
                             "September 17, 2026 Peak GAF filing at Schedule 2, Line 4\""},
                "request": {"type": "string", "description": "The request after the citation, beginning "
                            "'Please ...' (or a short context sentence then 'Please ...'). If there are "
                            "subparts, end with 'please provide' or 'Please explain' and put them in subparts"},
                "subparts": {"type": "array", "items": {"type": "string"},
                             "description": "Optional (a), (b), ... items, each a lowercase phrase, no numbering"},
                "rationale": {"type": "string", "description": "Why staff would ask this (one sentence)"},
                "source_ids": {"type": "array", "items": {"type": "integer"}},
            },
            "required": ["topic", "refer_to", "request", "rationale", "source_ids"]}}},
        "required": ["requests"]},
}

IR_SYSTEM = """You assist Massachusetts DPU staff in drafting information requests on public utility \
filings, in the format of the Department's issued requests. Rules:
- One subject per request. Begin by citing the document and pinpoint (page, section, schedule, line), \
using the citation given for each file; then make the request with "Please provide", "Please explain", \
"Please identify", "Please reconcile" or "Please describe".
- Use subparts (a), (b), ... when asking for several related items; each subpart is a lowercase phrase \
(the formatting of "; and" and the final period is added for you). Do not use a single subpart.
- Neutral and specific. Name the rate classes (e.g., R-3, G-41), periods and effective dates.
- Write "percent", not "%". Write dates in full (November 1, 2026). Refer to earlier requests as \
"information request DPU 1-1".
- Prioritize: figures or schedules that do not tie, large changes, unexplained drivers, missing \
workpapers or support, forecast assumptions, and bill impacts. Do not ask for what the excerpts already provide.

Two requests the Department has issued, for style:
Refer to the Company’s September 17, 2026 Peak GAF filing, the Company’s August 3, 2026 LDAF filing, any \
revisions to the Company’s GAF or LDAF filings, and the Company’s response to information request DPU 1-1.  \
For each illustrative residential LDAF that the Company calculated to reallocate 30, 40, and 50 percent of \
costs, please provide: (a) updated residential bill impacts to also include the impact due to the proposed \
peak GAF; and (b) an updated residential “all-in” bill impact analysis, comparing all proposed peak rates \
for effect November 1, 2026, to those rates that were in effect on November 1, 2025.
Refer to the Company’s September 17, 2026 gas adjustment factor (“GAF”) filing cover letter at 1, 3, the \
Company’s September 17, 2026 GAF filing, Section VI, and the Department’s order in Natural Gas Unbundling, \
D.T.E. 04-1, at 45-47 (2005).  Please explain how the Company recovers costs associated with the variance \
between non-daily metered actual customer usage and the adjusted target values."""


def draft_irs(client, model: str, chunks: list[Chunk], issues: list[Issue], prefix: str = "DPU",
              set_no: int = 1, n: int = 10, max_chars: int = 70_000, max_tokens: int = 4000,
              measure=len, refs: dict[str, str] | None = None) -> pd.DataFrame:
    from .irformat import compose, label
    refs = refs or {}
    cites = "\n".join(f"- {f}: cite as “{r}”" for f, r in refs.items()) or "- cite each file as “the Company’s filing”"
    flagged = "\n".join(f"- [{i.severity}] {i.description} ({i.file}, {', '.join(i.locations)})"
                        for i in issues if i.kind != "unverified") or "- (no automated findings)"
    head = f"How to cite each file:\n{cites}\n\nAutomated review findings:\n{flagged}"
    batch = _batches(chunks, max(1, max_chars - measure(head)), measure)[0] if chunks else []
    out = _call_tool(client, model, IR_SYSTEM,
                     f"{head}\n\nFiling excerpts:\n\n{_sources_block(batch)}"
                     f"\n\nDraft up to {n} information requests, most important first.", IR_TOOL, max_tokens)
    rows = []
    reqs = [r for r in out.get("requests", []) or [] if isinstance(r, dict) and r.get("request")]
    for k, r in enumerate(reqs[:n], start=1):
        ids = [int(x) for x in (r.get("source_ids") or []) if _to_float(x) is not None]
        based = [batch[s - 1].citation() for s in ids if 1 <= s <= len(batch)]
        subs = [str(x) for x in (r.get("subparts") or []) if str(x).strip()]
        text = compose(str(r.get("refer_to", "") or ""), str(r.get("request", "")), subs if len(subs) > 1 else None)
        if len(subs) == 1:                                   # a single subpart is folded into the request
            text = text.rstrip(".:") + " " + subs[0].rstrip(".") + "."
        rows.append({"No.": label(prefix, set_no, k), "Topic": r.get("topic", ""),
                     "Request": text, "Why": r.get("rationale", ""), "Based on": "; ".join(based)})
    return pd.DataFrame(rows, columns=["No.", "Topic", "Request", "Why", "Based on"])


def clean_irs(df: pd.DataFrame, prefix: str = "DPU", set_no: int = 1) -> pd.DataFrame:
    """Tidy the edited IR table (empty rows dropped, new rows numbered, format re-applied)."""
    from .irformat import IRSet, clean
    return clean(df, IRSet(prefix=prefix, set_no=set_no))


def irs_to_text(df: pd.DataFrame, header: str = "", meta=None) -> str:
    """Plain-text set in the Department's format (see rag/irformat.py)."""
    from .irformat import IRSet, to_text
    return to_text(df, meta or IRSet())


# --------------------------------------------------------------------------
# 4. grade IR responses
# --------------------------------------------------------------------------
GRADE_TOOL = {
    "name": "record_grades",
    "description": "Record an assessment of each information request response.",
    "input_schema": {
        "type": "object",
        "properties": {"grades": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "ir": {"type": "string"},
                "status": {"type": "string", "enum": ["answered", "partial", "non-responsive",
                                                      "objection", "refers to attachment"]},
                "summary": {"type": "string", "description": "One-sentence summary of the answer"},
                "gaps": {"type": "string", "description": "What was asked but not provided; empty if none"},
                "attachments": {"type": "array", "items": {"type": "string"},
                                "description": "Attachments or exhibits the response relies on"},
                "follow_up": {"type": "string", "description": "Suggested follow-up request, or empty"},
            },
            "required": ["ir", "status", "summary", "gaps", "attachments", "follow_up"]}}},
        "required": ["grades"]},
}

GRADE_SYSTEM = """You review utility responses to DPU information requests. For each IR, compare the \
response with what was requested. "answered" = fully responsive; "partial" = some parts missing or \
vague; "non-responsive" = does not address the request; "objection" = the company objects or declines; \
"refers to attachment" = the substance is in an attachment not included in the text. Be specific about \
gaps and suggest a precise follow-up request when warranted. Base everything only on the text given."""


def ir_groups(chunks: list[Chunk]) -> dict[str, list[Chunk]]:
    groups: dict[str, list[Chunk]] = {}
    for c in chunks:
        if c.ir:
            groups.setdefault(c.ir.upper(), []).append(c)
    return groups


GRADE_COLS = ["IR", "Status", "Summary", "Gaps", "Attachments not uploaded", "Suggested follow-up", "Where"]


def missing_attachments(atts: list[str], uploaded_files: list[str]) -> list[str]:
    """Attachments named in a response whose identifier (e.g. DPU-1-2) is in no uploaded file name."""
    names = " ".join(uploaded_files).lower()
    return [a for a in atts if a and not any(tok in names for tok in
                                             re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)+", a.lower()))]


def ir_batches(groups: dict[str, list[Chunk]], max_chars: int, max_items: int = 12, measure=len) -> list[list]:
    out, cur, n = [], [], 0
    for ir, cs in groups.items():
        size = sum(measure(c.text) for c in cs)
        if cur and (n + size > max_chars or len(cur) >= max_items):
            out.append(cur)
            cur, n = [], 0
        cur.append((ir, cs))
        n += size
    if cur:
        out.append(cur)
    return out


def add_unanswered(rows: list[dict], groups, drafted) -> pd.DataFrame:
    from .irformat import normalize_id
    answered = {normalize_id(g) for g in groups}
    for ir in drafted or []:
        if normalize_id(ir) not in answered:
            rows.append({"IR": ir, "Status": "no response found", "Summary": "", "Gaps": "",
                         "Attachments not uploaded": "", "Suggested follow-up": "", "Where": ""})
    return pd.DataFrame(rows, columns=GRADE_COLS)


def grade_responses(client, model: str, chunks: list[Chunk], uploaded_files: list[str],
                    drafted: list[str] | None = None, max_chars: int = 60_000,
                    max_tokens: int = 4000, measure=len) -> pd.DataFrame:
    groups = ir_groups(chunks)
    rows = []
    for part in ir_batches(groups, max_chars, measure=measure):
        body = "\n\n".join(f'<ir id="{ir}">\n' + "\n".join(c.text for c in cs) + "\n</ir>" for ir, cs in part)
        out = _call_tool(client, model, GRADE_SYSTEM, f"Responses:\n\n{body}", GRADE_TOOL, max_tokens)
        for g in out.get("grades", []) or []:
            if not isinstance(g, dict):
                continue
            ir = str(g.get("ir", "")).upper()
            atts = [str(a) for a in (g.get("attachments") or [])]
            loc = groups.get(ir, [None])[0]
            rows.append({"IR": ir, "Status": g.get("status", ""), "Summary": g.get("summary", ""),
                         "Gaps": g.get("gaps", ""),
                         "Attachments not uploaded": ", ".join(missing_attachments(atts, uploaded_files)),
                         "Suggested follow-up": g.get("follow_up", ""),
                         "Where": loc.citation() if loc else ""})
    return add_unanswered(rows, groups, drafted)


# --------------------------------------------------------------------------
# 5. compare filings
# --------------------------------------------------------------------------
def compare(figs: list[Figure], prior_file: str, current_file: str) -> pd.DataFrame:
    df = figures_frame(figs)
    key = ["metric", "rate_class_k"]
    df["rate_class_k"] = df["rate_class"].str.lower().str.strip()
    # within a filing, the proposed figure is the filing's own number
    rank = {"proposed": 0, "forecast": 1, "actual": 2, "current": 3, "prior": 4, "": 5}
    df["r"] = df["status"].map(rank).fillna(5)
    pick = lambda f: (df[df["file"] == f].sort_values("r").drop_duplicates(key)
                      .set_index(key)[["label", "value", "unit", "location"]])
    a, b = pick(prior_file), pick(current_file)
    out = a.merge(b, how="outer", left_index=True, right_index=True,
                  suffixes=(" (prior)", " (current)")).reset_index()
    out["Change"] = out["value (current)"] - out["value (prior)"]
    out["Change %"] = out["Change"] / out["value (prior)"].abs() * 100
    out["Metric"] = out["metric"].map(METRICS).fillna(out["metric"])
    out["Rate class"] = out["rate_class_k"]
    cols = ["Metric", "Rate class", "label (current)", "value (prior)", "value (current)", "Change",
            "Change %", "unit (current)", "location (prior)", "location (current)"]
    return out[cols].sort_values(["Metric", "Rate class"]).reset_index(drop=True)
