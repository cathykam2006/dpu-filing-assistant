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


def _batches(chunks: list[Chunk], max_chars: int = 60_000) -> list[list[Chunk]]:
    out, cur, n = [], [], 0
    for c in chunks:
        if cur and n + len(c.text) > max_chars:
            out.append(cur)
            cur, n = [], 0
        cur.append(c)
        n += len(c.text)
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
verbatim from the source so it can be checked automatically. Do not compute, infer or round. Skip page \
numbers, dates, docket numbers and line numbers."""


def extract_figures(client, model: str, chunks: list[Chunk]) -> list[Figure]:
    numeric = [c for c in chunks if has_numbers(c)]
    figures: list[Figure] = []
    for batch in _batches(numeric):
        out = _call_tool(client, model, EXTRACT_SYSTEM,
                         f"Filing excerpts:\n\n{_sources_block(batch)}", FIGURE_TOOL)
        for f in out.get("figures", []):
            sid = int(f.get("source_id", 0))
            if not 1 <= sid <= len(batch):
                continue
            src = batch[sid - 1]
            fig = Figure(metric=f.get("metric", "other"), label=f.get("label", ""),
                         value=float(f.get("value", 0)), unit=f.get("unit", ""),
                         rate_class=f.get("rate_class", "") or "", period=f.get("period", "") or "",
                         status=f.get("status", "") or "", quote=f.get("quote", ""),
                         source_id=sid, file=src.file, location=src.location)
            fig.verified = verify(fig, src.text)
            if not fig.verified:
                # the model sometimes cites the neighbouring source; relocate if the quote is elsewhere
                alt = next((c for c in batch if verify(fig, c.text)), None)
                if alt is not None:
                    fig.file, fig.location, fig.verified = alt.file, alt.location, True
            figures.append(fig)
    # the same figure is often in a table and the text; keep one, preferring verified
    seen, uniq = set(), []
    for f in sorted(figures, key=lambda f: not f.verified):
        key = (f.file, f.metric, f.rate_class.lower(), f.status, round(f.value, 6))
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
        if (d["metric"] == "gaf").any():
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
    "description": "Record draft information requests.",
    "input_schema": {
        "type": "object",
        "properties": {"requests": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "topic": {"type": "string"},
                "request": {"type": "string", "description": "The request text in DPU style"},
                "rationale": {"type": "string", "description": "Why staff would ask this (one sentence)"},
                "source_ids": {"type": "array", "items": {"type": "integer"}},
            },
            "required": ["topic", "request", "rationale", "source_ids"]}}},
        "required": ["requests"]},
}

IR_SYSTEM = """You assist Massachusetts DPU staff in drafting information requests (discovery) on \
public utility filings. Write requests in the style used in DPU proceedings: specific, neutral, \
one subject per request, referring to the page, schedule or line that prompts it ("Refer to page 2 \
of the filing. Please explain...", "Please provide the workpapers supporting...", "Please reconcile..."). \
Prioritize: figures that do not tie out, large changes, unexplained drivers, missing support or \
workpapers, assumptions behind forecasts, and bill impacts. Do not ask for information the excerpts \
already provide. Cite the source ids each request relies on."""


def draft_irs(client, model: str, chunks: list[Chunk], issues: list[Issue], prefix: str = "DPU",
              set_no: int = 1, n: int = 10) -> pd.DataFrame:
    flagged = "\n".join(f"- [{i.severity}] {i.description} ({i.file}, {', '.join(i.locations)})"
                        for i in issues) or "- (no automated findings)"
    batch = _batches(chunks, 70_000)[0] if chunks else []
    out = _call_tool(client, model, IR_SYSTEM,
                     f"Automated review findings:\n{flagged}\n\nFiling excerpts:\n\n{_sources_block(batch)}"
                     f"\n\nDraft up to {n} information requests, most important first.", IR_TOOL)
    rows = []
    for k, r in enumerate(out.get("requests", [])[:n], start=1):
        cites = [batch[s - 1].citation() for s in r.get("source_ids", []) if 1 <= s <= len(batch)]
        rows.append({"No.": f"{prefix}-{set_no}-{k}", "Topic": r.get("topic", ""),
                     "Request": r.get("request", ""), "Why": r.get("rationale", ""),
                     "Based on": "; ".join(cites)})
    return pd.DataFrame(rows, columns=["No.", "Topic", "Request", "Why", "Based on"])


def irs_to_text(df: pd.DataFrame, header: str = "") -> str:
    lines = [header] if header else []
    for _, r in df.iterrows():
        lines += [f"Information Request {r['No.']}", r["Request"], ""]
    return "\n".join(lines)


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


def grade_responses(client, model: str, chunks: list[Chunk], uploaded_files: list[str],
                    drafted: list[str] | None = None) -> pd.DataFrame:
    groups = ir_groups(chunks)
    rows = []
    items = list(groups.items())
    for i in range(0, len(items), 12):
        part = items[i:i + 12]
        body = "\n\n".join(f'<ir id="{ir}">\n' + "\n".join(c.text for c in cs) + "\n</ir>" for ir, cs in part)
        out = _call_tool(client, model, GRADE_SYSTEM, f"Responses:\n\n{body}", GRADE_TOOL)
        for g in out.get("grades", []):
            ir = g.get("ir", "").upper()
            atts = g.get("attachments", []) or []
            names = " ".join(uploaded_files).lower()
            missing = [a for a in atts if a and not any(tok in names for tok in
                                                        re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)+", a.lower()))]
            loc = groups.get(ir, [None])[0]
            rows.append({"IR": ir, "Status": g.get("status", ""), "Summary": g.get("summary", ""),
                         "Gaps": g.get("gaps", ""), "Attachments not uploaded": ", ".join(missing),
                         "Suggested follow-up": g.get("follow_up", ""),
                         "Where": loc.citation() if loc else ""})
    for ir in drafted or []:
        if ir.upper() not in groups:
            rows.append({"IR": ir.upper(), "Status": "no response found", "Summary": "", "Gaps": "",
                         "Attachments not uploaded": "", "Suggested follow-up": "", "Where": ""})
    cols = ["IR", "Status", "Summary", "Gaps", "Attachments not uploaded", "Suggested follow-up", "Where"]
    return pd.DataFrame(rows, columns=cols)


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
    out = a.join(b, how="outer", lsuffix=" (prior)", rsuffix=" (current)").reset_index()
    out["Change"] = out["value (current)"] - out["value (prior)"]
    out["Change %"] = out["Change"] / out["value (prior)"].abs() * 100
    out["Metric"] = out["metric"].map(METRICS).fillna(out["metric"])
    out["Rate class"] = out["rate_class_k"]
    cols = ["Metric", "Rate class", "label (current)", "value (prior)", "value (current)", "Change",
            "Change %", "unit (current)", "location (prior)", "location (current)"]
    return out[cols].sort_values(["Metric", "Rate class"]).reset_index(drop=True)
