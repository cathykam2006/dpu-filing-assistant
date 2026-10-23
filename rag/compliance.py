"""Does the filing do what the Department asked?

Drop in a Department order, a prior order, or the filing's own cover letter.
1. Requirements are pulled out: directive sentences ("shall", "is directed to",
   "ORDERED: That ..."), approvals with values, and the documents a cover letter
   says are enclosed. With a model, Claude/Groq/Gemini extract them too; every
   requirement quote is checked to be verbatim in the order.
2. Each requirement's concrete items are checked in the filing by code:
     values    is $0.4310 in the filing? Is there a related figure (same label or
               rate class) with a different value? Caps ("shall not exceed") and
               floors ("at least") are compared.
     documents is Schedule 4 / Attachment A actually provided (a page or sheet
               headed with it, or an uploaded file named for it), not just mentioned?
     dates     is the date in the filing; for deadlines, is a filing date stated?
3. With a model, each requirement is also judged (met / partly met / not met /
   not addressed / contradicts) with a quote from the filing, checked verbatim.
The final status is the worst of the code checks and the model's judgement, so
a model saying "met" never hides a value the code could not find.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from . import schedules as S
from .discovery import _call_tool, _norm, _sources_block, _to_float
from .ingest import Document
from .rules import _sentences
from .verify import NUMBER, _same, _val, page_texts

DIRECTIVE = re.compile(
    r"\b(shall|must|is directed to|are directed to|directs the company|is required to|are required to|"
    r"ordered:|ordered that|approves?|is approved|are approved|no later than|is to file|will be required)\b",
    re.I)
ENCLOSED = re.compile(r"\b(enclosed|attached|submits the following|includes the following|"
                      r"please find the following|consists of the following)\b", re.I)
DOC_REF = re.compile(r"\b(Schedules?|Attachments?|Exhibits?|Appendix|Appendices|Workpapers?|Section)\s+"
                     r"(?:No\.\s*)?([A-Z]{0,4}[-\s]?\d[\w.-]*|[A-Z](?![a-z]))", re.I)
MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
DATE = re.compile(rf"\b(?:{MONTHS})\s+\d{{1,2}},\s+\d{{4}}\b")
CAP = re.compile(r"not (?:to )?exceed|no (?:more|greater|higher) than|at most|cap(?:ped)? at|maximum of|up to", re.I)
FLOOR = re.compile(r"at least|no (?:less|lower|fewer) than|minimum of|not (?:be )?(?:less|lower) than", re.I)
DEADLINE = re.compile(r"no later than|on or before|\bby\s+(?=" + MONTHS + ")|deadline", re.I)
FILED_DATE = re.compile(rf"(?:filed|dated|date|submitted)\s*:?\s*(?:on\s+)?((?:{MONTHS})(?:\s+\d{{1,2}},)?\s+\d{{4}})",
                        re.I)
STOP = set("""the a an of to in on for and or is are be by with as at from that this these its it shall must
company department approved approves approve per therm not exceed than no more less least filing filed
will which each any all include including provide report its their such said other""".split())

STATUS_ORDER = ["✗ Discrepancy", "✗ Not met", "✗ Not provided", "? Not found", "◐ Partly met", "? Review",
                "✓ Met", "✓ Items found"]


@dataclass
class Item:
    kind: str                 # value | document | date | deadline
    text: str
    status: str               # found | missing | conflict | mentioned | review
    where: str = ""
    note: str = ""


@dataclass
class Requirement:
    id: str
    text: str
    source: str               # where in the order / cover letter
    origin: str               # directive | enclosure
    verified: bool = True
    items: list[Item] = field(default_factory=list)
    ai_status: str = ""
    ai_evidence: str = ""
    ai_where: str = ""
    ai_note: str = ""
    status: str = ""
    discrepancy: str = ""


# --------------------------------------------------------------------------
# 1. requirements
# --------------------------------------------------------------------------
def extract_requirements(order_docs: list[Document]) -> list[Requirement]:
    """Directive sentences and cover-letter enclosure lists, by pattern."""
    reqs: list[Requirement] = []
    for d in order_docs:
        for loc, text in page_texts(d):
            # ordering paragraphs: "Accordingly, ... it is ORDERED: That ...; and it is FURTHER ORDERED: ..."
            flat = re.sub(r"(?:Accordingly,[^.:]*?,\s*)?(?:;?\s*and\s+)?it\s+is\s+((?:FURTHER\s+)?ORDERED\s*:)",
                          r". \1", text)
            for s in _sentences(flat):
                s = re.sub(r"\.\s*\.$", ".", s.lstrip(". "))
                if DIRECTIVE.search(s) and len(s) > 25:
                    reqs.append(Requirement("", s.strip(), f"{d.name} · {loc}", "directive"))
            lines = [ln.strip() for ln in text.splitlines()]
            for i, ln in enumerate(lines):
                if ENCLOSED.search(ln):
                    for item in lines[i + 1:i + 40]:
                        m = DOC_REF.match(item)
                        if m:
                            reqs.append(Requirement("", f"Cover letter lists as enclosed: {item}",
                                                    f"{d.name} · {loc}", "enclosure"))
                        elif reqs and item and not item[0].isdigit() and len(item) > 60:
                            break
    seen, out = set(), []
    for r in reqs:
        key = _norm(r.text)
        if key not in seen:
            seen.add(key)
            r.id = f"R{len(out) + 1}"
            out.append(r)
    return out


REQ_TOOL = {
    "name": "record_requirements",
    "description": "Record each thing the Department directs, approves or requires, or each document a "
                   "cover letter says is enclosed.",
    "input_schema": {"type": "object", "properties": {"requirements": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "quote": {"type": "string", "description": "The requirement copied verbatim from the source"},
            "summary": {"type": "string", "description": "What the filing must contain or do, in one line"},
            "source_id": {"type": "integer"}},
        "required": ["quote", "summary", "source_id"]}}}, "required": ["requirements"]},
}
REQ_SYSTEM = """You read Massachusetts DPU orders, prior orders and filing cover letters, and list every \
requirement a later filing must satisfy: directives ("the Company shall...", "is directed to..."), ordering \
paragraphs, approved values or caps, required schedules, attachments, analyses, reports and deadlines, and \
documents a cover letter says are enclosed. Copy each requirement verbatim as the quote. Do not list \
background, procedural history or the Department's reasoning unless it imposes a requirement."""


def extract_requirements_ai(client, model: str, order_docs: list[Document], max_chars: int = 60_000,
                            max_tokens: int = 3000, measure=len) -> list[Requirement]:
    from .discovery import _batches
    chunks = [c for d in order_docs for c in d.chunks]
    out = extract_requirements(order_docs)            # patterns first; the model adds what they miss
    have = {_norm(r.text) for r in out}
    for batch in _batches(chunks, max_chars, measure):
        res = _call_tool(client, model, REQ_SYSTEM, f"Documents:\n\n{_sources_block(batch)}", REQ_TOOL, max_tokens)
        for r in res.get("requirements", []) or []:
            if not isinstance(r, dict) or not r.get("quote"):
                continue
            sid = _to_float(r.get("source_id"))
            src = batch[int(sid) - 1] if sid and 1 <= int(sid) <= len(batch) else None
            quote = str(r["quote"]).strip()
            q = _norm(quote)
            if any(q in h or h in q for h in have):    # already found by pattern
                continue
            verified = any(q.replace(" ", "") in _norm(c.text).replace(" ", "") for c in batch)
            have.add(q)
            out.append(Requirement(f"R{len(out) + 1}", quote, src.citation() if src else "",
                                   "directive", verified))
    return out


# --------------------------------------------------------------------------
# 2. item checks (code)
# --------------------------------------------------------------------------
def _filing_lines(filing: list[Document]) -> list[tuple[str, str]]:
    out = []
    for d in filing:
        for loc, text in page_texts(d):
            out += [(f"{d.name} · {loc}", ln) for ln in text.splitlines() if ln.strip()]
        for c in d.chunks:
            if c.kind in ("table", "spreadsheet"):
                out += [(c.citation(), ln) for ln in c.text.splitlines() if ln.strip()]
    return out


def _numbers(line: str) -> list[tuple[str, float]]:
    out = []
    for m in NUMBER.finditer(line):
        v = _val(m.group(0))
        if v is not None:
            out.append((m.group(0).strip(), v))
    return out


def _decimals(tok: str) -> int:
    m = re.search(r"\.(\d+)", tok)
    return len(m.group(1)) if m else 0


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z][a-z&]+", text.lower()) if w not in STOP and len(w) > 2}


def _filing_schedules(filing: list[Document]) -> list[tuple[str, S.Schedule]]:
    out = []
    for d in filing:
        for loc, text in page_texts(d):
            sch = S.parse(text)
            if sch:
                out.append((f"{d.name} · {loc}", sch))
    return out


def _kind_ok(tok: str, ref: str, x: float, v: float) -> bool:
    """Same kind of number: same decimal places and within a factor of three."""
    return _decimals(tok) == _decimals(ref) and x != 0 and v != 0 and 1 / 3 <= abs(x / v) <= 3


def check_value(req: str, tok: str, pos: int, lines, scheds=()) -> Item:
    v = _val(tok)
    window = req[max(0, pos - 90):pos]
    label = _content_words(window)
    classes = set(S.CLASS.findall(req[max(0, pos - 120):pos + 40]))
    exact = [(w, ln) for w, ln in lines if any(_same(x, v) for _, x in _numbers(ln))]
    related = []
    # schedules: the rate class is a column, so read the named class's cell in rows with the same label
    for w, sch in scheds:
        cols = [c for c in sch.columns if c in classes] if classes else list(sch.columns)
        if classes and not cols:
            continue
        for r in sch.rows.values():
            overlap = len(label & _content_words(r.label))
            if overlap < 2:
                continue
            nums = [(f"{sch.columns[j]}: {r.raw[j]}", r.values[j])
                    for j in range(min(len(r.values), len(sch.columns)))
                    if sch.columns[j] in cols and _kind_ok(r.raw[j], tok, r.values[j], v)]
            if nums:
                related.append((overlap + 1, f"{w}, line {r.line}", r.text, nums))
    sched_rows = {r.text for _, sch in scheds for r in sch.rows.values()}
    for w, ln in lines:
        if ln.strip() in sched_rows:
            continue
        if classes and not all(c in ln for c in classes):
            continue
        overlap = len(label & _content_words(ln))
        if overlap < (1 if classes else 2):
            continue
        nums = [(t, x) for t, x in _numbers(ln) if _kind_ok(t, tok, x, v)]
        if nums:
            related.append((overlap, w, ln, nums))
    related.sort(key=lambda r: -r[0])
    cap, floor = CAP.search(window[-40:]), FLOOR.search(window[-40:])
    if cap or floor:
        bad = [(w, t) for _, w, _, nums in related[:1] for t, x in nums
               if (x > v + 1e-12 if cap else x < v - 1e-12)]
        if bad:
            w, t = bad[0]
            return Item("value", tok, "conflict", w, f"the order sets a {'cap' if cap else 'floor'} of {tok}; "
                        f"the filing shows {t}, which is {'above' if cap else 'below'} it")
        if related:
            return Item("value", tok, "found", related[0][1],
                        f"filing values {', '.join(t for t, _ in related[0][3])} are within the "
                        f"{'cap' if cap else 'floor'}")
    if any(_same(x, v) for _, _, _, nums in related[:1] for _, x in nums):
        return Item("value", tok, "found", related[0][1])
    if related:
        _, w, ln, nums = related[0]
        shown = list(dict.fromkeys(t for t, _ in nums))
        if len(nums) > 2 and len({x for _, x in nums}) == 1:          # same value in every class
            shown = [f"{nums[0][0].split(': ')[-1]} in all {len(nums)} rate classes"]
        return Item("value", tok, "conflict", w, f"the order says {tok}; the filing shows "
                    f"{', '.join(shown[:6])}{' …' if len(shown) > 6 else ''} for the same item: "
                    f"\"{ln.strip()[:120]}\"")
    if exact:
        return Item("value", tok, "found", exact[0][0])
    return Item("value", tok, "missing", "", "not found anywhere in the filing")


def _doc_patterns(ref: str) -> re.Pattern:
    word, ident = re.match(r"(\w+)\s+(?:No\.\s*)?(.+)", ref).groups()
    word = word.rstrip("s").lower()
    return re.compile(rf"\b{word}s?\s+(?:No\.\s*)?{re.escape(ident)}(?![\w.]*\d)", re.I)


def check_document(ref: str, filing: list[Document]) -> Item:
    pat = _doc_patterns(ref)
    ident = re.split(r"\s+", ref, maxsplit=1)[-1]
    for d in filing:
        if ident.lower() in d.name.lower() and len(ident) > 1:
            return Item("document", ref, "found", d.name, "uploaded file named for it")
        for loc, text in page_texts(d):
            lines = [ln for ln in text.splitlines() if ln.strip()]
            if any(pat.search(ln) for ln in lines[:8] + lines[-3:]):
                return Item("document", ref, "found", f"{d.name} · {loc}", "page headed with it")
        for c in d.chunks:
            if c.kind == "spreadsheet" and pat.search(c.location):
                return Item("document", ref, "found", c.citation(), "sheet named for it")
    for d in filing:
        for loc, text in page_texts(d):
            if pat.search(text):
                return Item("document", ref, "mentioned", f"{d.name} · {loc}",
                            "mentioned in the filing, but no page, sheet or file is headed with it")
    return Item("document", ref, "missing", "", "not in the filing")


LABELLED_DATE = re.compile(rf"\b(?:filed|dated|date|submitted)\s*:\s*((?:{MONTHS})(?:\s+\d{{1,2}},)?\s+\d{{4}})",
                           re.I)


def filing_date(filing: list[Document]) -> tuple[str, str] | None:
    """The filing's own date: a labelled "Filed: …" / "Dated: …" first; otherwise "filed …" on a line that is
    not citing another docket (schedules often say "D.P.U. 26-99 filed July 30, 2026")."""
    for pattern, skip_citations in ((LABELLED_DATE, False), (FILED_DATE, True)):
        for d in filing:
            for loc, text in page_texts(d):
                for ln in text.splitlines():
                    if skip_citations and re.search(r"D\.\s?P\.\s?U\.|D\.T\.E\.", ln):
                        continue
                    m = pattern.search(ln)
                    if m:
                        return m.group(1), f"{d.name} · {loc}"
    return None


def check_items(req: Requirement, filing: list[Document]) -> list[Item]:
    lines = _filing_lines(filing)
    scheds = _filing_schedules(filing)
    text = req.text.replace("Cover letter lists as enclosed: ", "")
    items = []
    for m in DOC_REF.finditer(text):
        ref = re.sub(r"\s+", " ", m.group(0)).rstrip(".,;")
        if m.group(1).lower().startswith("section"):
            continue                                  # sections of the order itself
        items.append(check_document(ref, filing))
    deadline = DEADLINE.search(text)
    for m in DATE.finditer(text):
        if deadline:
            fd = filing_date(filing)
            items.append(Item("deadline", m.group(0), "review", fd[1] if fd else "",
                              f"filing is dated {fd[0]}; confirm it is on or before {m.group(0)}" if fd
                              else "no filing date stated in the uploaded documents; check the docket"))
        else:
            found = next((w for w, ln in lines if m.group(0) in ln), None)
            items.append(Item("date", m.group(0), "found" if found else "missing", found or ""))
    for m in NUMBER.finditer(text):
        before = text[max(0, m.start() - 12):m.start()]
        if re.search(r"(D\.P\.U\.|No\.|Line|Page|Section|CMR)\s*$", before):
            continue
        items.append(check_value(text, m.group(0).strip(), m.start(), lines, scheds))
    return items


# --------------------------------------------------------------------------
# 3. model judgement
# --------------------------------------------------------------------------
JUDGE_TOOL = {
    "name": "record_compliance",
    "description": "Record whether the filing satisfies each requirement.",
    "input_schema": {"type": "object", "properties": {"results": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "status": {"type": "string", "enum": ["met", "partly met", "not met", "not addressed", "contradicts"]},
            "evidence_quote": {"type": "string", "description": "Verbatim text from the filing excerpts, or empty"},
            "evidence_source_id": {"type": "integer"},
            "explanation": {"type": "string", "description": "One or two sentences; say exactly what is missing "
                                                              "or different"}},
        "required": ["id", "status", "evidence_quote", "explanation"]}}}, "required": ["results"]},
}
JUDGE_SYSTEM = """You check a utility's filing against requirements set by the Massachusetts DPU (orders, \
prior orders) or listed in the filing's own cover letter. For each requirement decide, using only the filing \
excerpts given: "met" (fully satisfied), "partly met" (some parts missing, e.g. bill impacts for one class \
when all classes were required), "not met" (addressed but not done as required), "not addressed" (nothing \
in the excerpts), or "contradicts" (the filing uses a different value, date or method than required). \
Quote the filing verbatim as evidence. Be strict: if the excerpts do not show it, it is not addressed."""


def judge(client, model: str, reqs: list[Requirement], index, max_chars: int = 30_000, max_tokens: int = 3000,
          measure=len, files: list[str] | None = None) -> None:
    """Fill ai_* fields. Requirements are grouped so each call's excerpts fit the budget."""
    groups, cur, cur_hits, used = [], [], [], 0
    for r in reqs:
        hits = [c for c, _ in index.search(r.text, k=3, files=files, always_rank=True)][:3]
        new = [c for c in hits if c not in cur_hits]
        size = sum(measure(c.text) for c in new) + measure(r.text)
        if cur and used + size > max_chars:
            groups.append((cur, cur_hits))
            cur, cur_hits, used = [], [], 0
            new = hits
            size = sum(measure(c.text) for c in new) + measure(r.text)
        cur.append(r)
        cur_hits += new
        used += size
    if cur:
        groups.append((cur, cur_hits))
    for grp, hits in groups:
        listing = "\n".join(f'<requirement id="{r.id}">{r.text}</requirement>' for r in grp)
        res = _call_tool(client, model, JUDGE_SYSTEM,
                         f"Requirements:\n{listing}\n\nFiling excerpts:\n\n{_sources_block(hits)}", JUDGE_TOOL,
                         max_tokens)
        by_id = {r.id: r for r in grp}
        for x in res.get("results", []) or []:
            if not isinstance(x, dict) or str(x.get("id")) not in by_id:
                continue
            r = by_id[str(x["id"])]
            r.ai_status = str(x.get("status", ""))
            r.ai_note = str(x.get("explanation", ""))
            quote = str(x.get("evidence_quote", "") or "")
            sid = _to_float(x.get("evidence_source_id"))
            src = hits[int(sid) - 1] if sid and 1 <= int(sid) <= len(hits) else None
            if quote:
                ok = next((c for c in ([src] if src else []) + hits
                           if _norm(quote).replace(" ", "") in _norm(c.text).replace(" ", "")), None)
                r.ai_evidence = quote if ok else f"(quote not found in the filing) {quote}"
                r.ai_where = ok.citation() if ok else ""


# --------------------------------------------------------------------------
# combine
# --------------------------------------------------------------------------
AI_MAP = {"met": "✓ Met", "partly met": "◐ Partly met", "not met": "✗ Not met",
          "not addressed": "? Not found", "contradicts": "✗ Discrepancy"}


def decide(r: Requirement) -> None:
    conflicts = [i for i in r.items if i.status == "conflict"]
    missing = [i for i in r.items if i.status == "missing"]
    mentioned = [i for i in r.items if i.status == "mentioned"]
    if conflicts:
        r.status, r.discrepancy = "✗ Discrepancy", "; ".join(f"{i.text}: {i.note}" for i in conflicts)
    elif r.ai_status == "contradicts":
        r.status, r.discrepancy = "✗ Discrepancy", r.ai_note
    elif missing and any(i.kind == "document" for i in missing):
        r.status = "✗ Not provided"
        r.discrepancy = "; ".join(f"{i.text}: {i.note}" for i in missing)
    elif missing:
        r.status = "? Not found"
        r.discrepancy = "; ".join(f"{i.text}: {i.note}" for i in missing)
    elif r.ai_status:
        r.status = AI_MAP.get(r.ai_status, "? Review")
        reviews = [i.note for i in r.items if i.status == "review"]
        if r.status == "✓ Met" and mentioned:
            r.status = "◐ Partly met"
            r.discrepancy = "; ".join(f"{i.text}: {i.note}" for i in mentioned)
        elif r.status == "✓ Met" and reviews:
            r.status, r.discrepancy = "? Review", "; ".join(reviews)
        elif r.status == "✓ Met" and (not r.ai_evidence or r.ai_evidence.startswith("(quote not found")):
            r.status = "? Review"
            r.discrepancy = "The model judged this met but gave no quote from the filing that shows it; check."
        elif r.status != "✓ Met":
            r.discrepancy = r.ai_note
    elif mentioned:
        r.status = "◐ Partly met"
        r.discrepancy = "; ".join(f"{i.text}: {i.note}" for i in mentioned)
    elif any(i.status == "review" for i in r.items):
        r.status = "? Review"
        r.discrepancy = "; ".join(i.note for i in r.items if i.status == "review")
    elif r.items:
        r.status = "✓ Items found"
    else:
        r.status = "? Review"
    if r.ai_status == "met" and (missing or conflicts):
        r.discrepancy = (f"The model judged this met, but the code check disagrees: {r.discrepancy}")
    if not r.verified:
        r.discrepancy = ("(requirement quote not found verbatim in the order; check it) " + r.discrepancy).strip()


def check(order_docs: list[Document], filing: list[Document], client=None, model: str = "", index=None,
          max_chars: int = 30_000, max_tokens: int = 3000, measure=len, files=None) -> list[Requirement]:
    if client is not None:
        reqs = extract_requirements_ai(client, model, order_docs, max_chars, max_tokens, measure)
    else:
        reqs = extract_requirements(order_docs)
    for r in reqs:
        r.items = check_items(r, filing)
    if client is not None and index is not None and reqs:
        judge(client, model, reqs, index, max_chars, max_tokens, measure, files)
    for r in reqs:
        decide(r)
    return reqs


def evidence_for(r: Requirement, index, files=None) -> tuple[str, str]:
    """Best filing passage for a requirement (no-AI mode)."""
    hits = index.search(r.text, k=1, files=files, always_rank=True) if index else []
    if not hits:
        return "", ""
    c = hits[0][0]
    return c.text[:300], c.citation()


def results_frame(reqs: list[Requirement], index=None, files=None) -> pd.DataFrame:
    rows = []
    for r in reqs:
        ev, where = (r.ai_evidence, r.ai_where) if r.ai_status else evidence_for(r, index, files)
        rows.append({"ID": r.id, "Status": r.status, "Requirement": r.text, "From": r.source,
                     "Discrepancy / what to check": r.discrepancy,
                     "Items checked": "; ".join(f"{i.text} → {i.status}" + (f" ({i.where})" if i.where else "")
                                               for i in r.items),
                     "Filing evidence": ev, "Where in filing": where,
                     "Model's view": f"{r.ai_status}: {r.ai_note}" if r.ai_status else ""})
    df = pd.DataFrame(rows, columns=["ID", "Status", "Requirement", "From", "Discrepancy / what to check",
                                     "Items checked", "Filing evidence", "Where in filing", "Model's view"])
    rank = {s: i for i, s in enumerate(STATUS_ORDER)}
    return df.assign(_r=df["Status"].map(rank).fillna(99)).sort_values(["_r", "ID"]).drop(columns="_r")


def order_reference(doc: Document) -> str:
    """How a request cites an order or cover letter: 'the Department’s Order in D.P.U. 98-PGAF-SAMPLE'."""
    text = " ".join(t for _, t in page_texts(doc))[:6000]
    if ENCLOSED.search(text) and not re.search(r"\bORDER\b", text):
        return "the Company’s filing cover letter"
    if doc.dockets and re.search(r"\border\b", text, re.I):
        return f"the Department’s Order in D.P.U. {doc.dockets[0]}"
    return doc.name


def follow_up_irs(df: pd.DataFrame, prefix: str = "DPU", set_no: int = 1, start: int = 1,
                  refs: dict[str, str] | None = None, filing_ref: str = "the Company’s filing") -> pd.DataFrame:
    """Information requests, in the Department's format, for every requirement not shown as met.
    refs maps an order / cover-letter file name to how it is cited."""
    from .irformat import compose, label, pinpoint
    refs = refs or {}
    rows = []
    for _, r in df[~df["Status"].str.startswith("✓")].iterrows():
        file, _, loc = str(r["From"]).partition(" · ")
        where = " ".join(x for x in (refs.get(file, file), pinpoint(loc)) if x)
        enclosure = str(r["Requirement"]).startswith("Cover letter lists as enclosed: ")
        quote = str(r["Requirement"]).replace("Cover letter lists as enclosed: ", "")[:300].rstrip(".")
        what = re.sub(r"^\S+: (?=the order|not in|mentioned)", "", str(r["Discrepancy / what to check"]))[:220]
        if r["Status"] == "✗ Discrepancy":
            ask = (f"Please explain why {filing_ref} differs from this requirement ({what.rstrip('.')}), and "
                   "provide corrected schedules, if necessary.")
        elif enclosure:
            ask = f"Please provide this document, which does not appear in {filing_ref}."
        else:
            ask = (f"Please identify where {filing_ref} complies with this requirement or, if it does not, "
                   "explain why and provide the missing information.")
        cited = (f"Refer to {where}, which lists “{quote}” as enclosed." if enclosure
                 else f"Refer to {where}, which states: “{quote}.”")
        rows.append({"No.": label(prefix, set_no, start + len(rows)), "Topic": f"Compliance ({r['ID']})",
                     "Request": compose("", f"{cited}  {ask}"),
                     "Why": f"Compliance check: {r['Status']}", "Based on": r["From"]})
    return pd.DataFrame(rows, columns=["No.", "Topic", "Request", "Why", "Based on"])
