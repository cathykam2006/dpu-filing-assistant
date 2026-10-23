"""Information requests in the Department's issued format.

The format follows information requests the Department has issued in public dockets, e.g. the
second set of common information requests in the D.P.U. 26-PGAF dockets (October 2, 2026):

    SECOND SET OF INFORMATION REQUESTS OF THE
    DEPARTMENT OF PUBLIC UTILITIES
    TO NSTAR GAS COMPANY
    D.P.U. 26-PGAF-NSTAR
    ____________________________________________________________
    Pursuant to 220 CMR 1.06(5)(c), the Department of Public Utilities ("Department")
    submits to NSTAR Gas Company ("Company") its second set of Information Requests.

                                  Requests

    DPU 2-1     Refer to the Company's September 17, 2026 Peak GAF filing at Section VI.
                Please provide:
                (a)   updated residential bill impacts ...; and
                (b)   an updated ... analysis.

    Dated: October 2, 2026

Requests open with "Refer to <document> at <page/schedule/line>", then "Please ...". Subparts are
(a), (b), ... each starting lowercase, ending ";", with "; and" before the last and "." at the end.
Curly quotes, "percent" rather than "%", full dates. Pages after the first carry a running header
"D.P.U. <docket>  Page N". Drafts are marked DRAFT and carry no seal or letterhead.
"""
from __future__ import annotations

import io
import re
import textwrap
from dataclasses import dataclass
from datetime import date

import pandas as pd

ORDINALS = ["FIRST", "SECOND", "THIRD", "FOURTH", "FIFTH", "SIXTH", "SEVENTH", "EIGHTH", "NINTH", "TENTH",
            "ELEVENTH", "TWELFTH", "THIRTEENTH", "FOURTEENTH", "FIFTEENTH", "SIXTEENTH", "SEVENTEENTH",
            "EIGHTEENTH", "NINETEENTH", "TWENTIETH"]
COLS = ["No.", "Topic", "Request", "Why", "Based on"]
SUB = re.compile(r"^\s*\(([a-z])\)\s*(.*)$")


@dataclass
class IRSet:
    company: str = "the Company"
    docket: str = ""
    set_no: int = 1
    prefix: str = "DPU"
    dated: str = ""                      # "October 2, 2026"
    authority: str = "220 CMR 1.06(5)(c)"

    @property
    def ordinal(self) -> str:
        return ORDINALS[self.set_no - 1] if 1 <= self.set_no <= len(ORDINALS) else f"{self.set_no}TH"

    @property
    def date_text(self) -> str:
        return self.dated or long_date(date.today())


def long_date(d: date) -> str:
    return f"{d.strftime('%B')} {d.day}, {d.year}"


def label(prefix: str, set_no: int, n: int) -> str:
    """'DPU 2-1', as the Department numbers its requests (common sets use e.g. 'DPU-LDC 2-1')."""
    return f"{prefix.strip()} {int(set_no)}-{int(n)}"


def normalize_id(s: str) -> str:
    """'DPU 2-1', 'DPU-2-1', 'dpu 2–1' -> 'DPU-2-1', for matching drafts to responses."""
    return re.sub(r"[\s–-]+", "-", str(s).strip().upper())


def pinpoint(location: str) -> str:
    """App locations to citation pinpoints: 'p. 2 (table)' -> 'at 2'; sheet rows -> 'Schedule 3, rows 5-10'."""
    loc = str(location or "").strip()
    m = re.match(r"p\.\s*(\d+)", loc)
    if m:
        return f"at {m.group(1)}"
    m = re.match(r"Sheet '([^']+)'(?:,\s*(rows? [\d-]+))?", loc)
    if m:
        sheet = m.group(1)
        where = sheet if re.match(r"(Schedule|Attachment|Exhibit|Workpaper)", sheet, re.I) else f"worksheet “{sheet}”"
        return f"at {where}" + (f", {m.group(2)}" if m.group(2) else "")
    if loc.startswith("¶"):
        return f"at ¶¶ {loc[1:].strip()}" if "-" in loc else f"at {loc}"
    return f"at {loc}" if loc else ""


# --------------------------------------------------------------------------
# style
# --------------------------------------------------------------------------
def smart_quotes(t: str) -> str:
    t = re.sub(r'(^|[\s(\[{—–-])"', r"\1“", t)
    t = t.replace('"', "”")
    t = re.sub(r"(\w)'(\w)", r"\1’\2", t)                     # Company's
    t = re.sub(r"(^|[\s(\[{])'", r"\1‘", t)
    return t.replace("'", "’")


ABBREV = re.compile(r"(?:\b(?:[A-Z]\.){2,}|\b(?:Inc|Corp|Co|No|Nos|St|Mr|Ms|Mrs|Dr|Jr|Sr|Ltd|vs|v|Sec|Fig|Sch|Att|Ex|al))\.$")


def _sentence_spacing(t: str) -> str:
    """Two spaces after a sentence, as in the Department's issued requests (not after D.P.U., Inc., No.)."""
    out, last = [], 0
    for m in re.finditer(r"([.?!][”’)]?) +(?=[A-Z(“‘])", t):
        if ABBREV.search(t[max(0, m.start() - 12):m.start() + 1]):
            continue
        out.append(t[last:m.end(1)] + "  ")
        last = m.end()
    return "".join(out) + t[last:]


def style_text(t: str) -> str:
    t = re.sub(r"[ \t]+", " ", str(t or "")).strip()
    t = re.sub(r"(\d)\s?%", r"\1 percent", t)
    t = re.sub(r"\be\.g\.(?!,)", "e.g.,", t)
    t = re.sub(r"\bi\.e\.(?!,)", "i.e.,", t)
    return _sentence_spacing(smart_quotes(t))


def split_request(text: str) -> tuple[str, list[str]]:
    """'body\n(a) x\n(b) y' -> ('body', ['x', 'y']). Inline '(a) ...; (b) ...' is split too."""
    text = str(text or "").strip()
    if "\n(" not in text and re.search(r":\s*\(a\)\s", text):
        text = re.sub(r"\s*\(([a-z])\)\s+", r"\n(\1) ", text)
    body, subs = [], []
    for ln in text.splitlines():
        m = SUB.match(ln)
        if m:
            subs.append(m.group(2).strip())
        elif subs:
            subs[-1] += " " + ln.strip()
        elif ln.strip():
            body.append(ln.strip())
    return " ".join(body), [s for s in subs if s]


def _subpart_punctuation(subs: list[str]) -> list[str]:
    out = []
    for i, s in enumerate(subs):
        s = re.sub(r"[;.,:]?\s*(?:and|or)?\s*$", "", s.strip())
        s = re.sub(r"[;.,]\s*$", "", s)
        if s and s[0].isupper() and not re.match(r"^[A-Z]{2,}|^[A-Z]-\d|^R-|^G-", s):
            s = s[0].lower() + s[1:]
        out.append(s + ("." if i == len(subs) - 1 else "; and" if i == len(subs) - 2 else ";"))
    return out


def compose(refer_to: str, request: str, subparts: list[str] | None = None) -> str:
    """Build one request in the Department's form."""
    refer_to, request = style_text(refer_to).rstrip(". "), style_text(request)
    subs = [style_text(s) for s in (subparts or []) if str(s).strip()]
    if request:
        request = request[0].upper() + request[1:]
    if refer_to and not request.lower().startswith("refer to"):
        ref = refer_to if refer_to.lower().startswith("refer to") else f"Refer to {refer_to}"
        request = f"{ref}. {request}"
    request = _sentence_spacing(request.strip())
    if subs:
        request = request.rstrip(" .;:") + ":"
        return request + "\n" + "\n".join(f"({chr(97 + i)}) {s}" for i, s in enumerate(_subpart_punctuation(subs)))
    return request if request.endswith((".", "?")) else request + "."


def format_request(text: str) -> str:
    """Re-apply the format to an edited request (quotes, percent, subpart punctuation)."""
    body, subs = split_request(text)
    return compose("", body, subs)


def lint(text: str) -> list[str]:
    """Format points a person must fix (the rest is fixed automatically by format_request)."""
    body, subs = split_request(text)
    notes = []
    if not body.lower().startswith("refer to"):
        notes.append("does not begin “Refer to …” with the document and page, schedule or line")
    elif not re.search(r"\bat\b|Section|Schedule|Attachment|Exhibit|Line|page|response to", body, re.I):
        notes.append("“Refer to …” has no pinpoint (page, section, schedule or line)")
    if not re.search(r"\bplease\b", body, re.I):
        notes.append("has no “Please …” request")
    if len(subs) == 1:
        notes.append("has a single subpart; fold it into the request or add the others")
    if len(body.split()) + sum(len(s.split()) for s in subs) > 220:
        notes.append("is long; consider splitting it into separate requests")
    return notes


def clean(df: pd.DataFrame, meta: IRSet) -> pd.DataFrame:
    """Tidy an edited table: drop empty rows, number new rows, and re-apply the format."""
    df = df.reindex(columns=COLS).astype(object).where(df.reindex(columns=COLS).notna(), "")
    df = df.map(lambda v: str(v).strip())
    df = df[df["Request"] != ""].reset_index(drop=True)
    used = {normalize_id(n) for n in df["No."] if n}
    k = 1
    for i in df.index:
        if not df.at[i, "No."]:
            while normalize_id(label(meta.prefix, meta.set_no, k)) in used:
                k += 1
            df.at[i, "No."] = label(meta.prefix, meta.set_no, k)
            used.add(normalize_id(df.at[i, "No."]))
        df.at[i, "Request"] = format_request(df.at[i, "Request"])
    return df


# --------------------------------------------------------------------------
# documents
# --------------------------------------------------------------------------
def title_lines(meta: IRSet) -> list[str]:
    return [f"{meta.ordinal} SET OF INFORMATION REQUESTS OF THE", "DEPARTMENT OF PUBLIC UTILITIES",
            f"TO {meta.company.upper()}", f"D.P.U. {meta.docket}".strip()]


def intro(meta: IRSet) -> str:
    return (f"Pursuant to {meta.authority}, the Department of Public Utilities (“Department”) submits to "
            f"{meta.company} (“Company”) its {meta.ordinal.lower()} set of Information Requests.")


def to_text(df: pd.DataFrame, meta: IRSet, width: int = 92) -> str:
    out = ["DRAFT — not issued", ""] + [ln.center(width).rstrip() for ln in title_lines(meta)]
    out += ["_" * width, textwrap.fill(intro(meta), width), "", "Requests".center(width).rstrip(), ""]
    for _, r in clean(df, meta).iterrows():
        body, subs = split_request(r["Request"])
        lab = r["No."].ljust(14)
        out.append(textwrap.fill(body, width, initial_indent=lab, subsequent_indent=" " * 14))
        for i, s in enumerate(subs):
            out += ["", textwrap.fill(s, width, initial_indent=" " * 14 + f"({chr(97 + i)})".ljust(7),
                                      subsequent_indent=" " * 21)]
        out.append("")
    out.append(f"Dated:  {meta.date_text}")
    return "\n".join(out) + "\n"


def to_docx(df: pd.DataFrame, meta: IRSet) -> bytes:
    import docx
    from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Inches, Pt

    d = docx.Document()
    st = d.styles["Normal"]
    st.font.name, st.font.size = "Times New Roman", Pt(12)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    st.paragraph_format.space_after = Pt(0)
    sec = d.sections[0]
    sec.top_margin = sec.bottom_margin = sec.left_margin = sec.right_margin = Inches(1)
    sec.different_first_page_header_footer = True

    def tabbed_header(p, left: str, page: bool):
        p.style = d.styles["Normal"]               # no inherited center tab: one right tab at the margin
        p.paragraph_format.tab_stops.add_tab_stop(Inches(6.5), WD_TAB_ALIGNMENT.RIGHT)
        p.add_run(left + "\t" + ("Page " if page else "DRAFT — not issued"))
        if page:
            fld = OxmlElement("w:fldSimple")
            fld.set(qn("w:instr"), "PAGE")
            r = OxmlElement("w:r")
            t = OxmlElement("w:t")
            t.text = "2"
            r.append(t)
            fld.append(r)
            p._p.append(fld)

    tabbed_header(sec.first_page_header.paragraphs[0], "", page=False)
    tabbed_header(sec.header.paragraphs[0], f"D.P.U. {meta.docket}".strip(), page=True)
    sec.footer.paragraphs[0].text = "DRAFT — not issued"
    sec.footer.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

    title = d.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_before = Pt(24)
    for i, ln in enumerate(title_lines(meta)):
        run = title.add_run(ln)
        if i < 3:
            run.add_break()
    rule = d.add_paragraph()
    pbdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    for k, v in (("val", "single"), ("sz", "6"), ("space", "1"), ("color", "000000")):
        bottom.set(qn(f"w:{k}"), v)
    pbdr.append(bottom)
    rule._p.get_or_add_pPr().append(pbdr)
    d.add_paragraph(intro(meta)).paragraph_format.space_after = Pt(12)
    head = d.add_paragraph()
    head.alignment = WD_ALIGN_PARAGRAPH.CENTER
    head.paragraph_format.space_after = Pt(12)
    run = head.add_run("Requests")
    run.bold, run.underline = True, True

    for _, r in clean(df, meta).iterrows():
        body, subs = split_request(r["Request"])
        p = d.add_paragraph()
        pf = p.paragraph_format
        pf.left_indent, pf.first_line_indent = Inches(1.0), Inches(-1.0)
        pf.tab_stops.add_tab_stop(Inches(1.0))
        pf.space_after = Pt(12)
        p.add_run(f"{r['No.']}\t{body}")
        for i, s in enumerate(subs):
            q = d.add_paragraph()
            qf = q.paragraph_format
            qf.left_indent, qf.first_line_indent = Inches(1.5), Inches(-0.5)
            qf.tab_stops.add_tab_stop(Inches(1.5))
            qf.space_after = Pt(12)
            q.add_run(f"({chr(97 + i)})\t{s}")
    d.add_paragraph(f"Dated:  {meta.date_text}")
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


# --------------------------------------------------------------------------
# defaults from the filing
# --------------------------------------------------------------------------
COMPANY = re.compile(r"\b((?:[A-Z][\w&.]* ){0,5}(?:Gas|Electric|Water|Energy|Power|Light)(?: (?:and|&) \w+)?"
                     r"(?: [A-Z][\w&.]*){0,3} (?:Company|Corporation|Corp\.|Inc\.))")


def guess_company(texts: list[str]) -> str:
    for t in texts:
        m = COMPANY.search(t[:3000])
        if m:
            return re.sub(r"\s+", " ", m.group(1)).strip()
    return "the Company"


def guess_docket(dockets: list[str], texts: list[str] | None = None) -> str:
    """The filing's own docket: the first one its first page mentions (the header), else a GAF/LDAF docket."""
    if texts:
        first = " ".join(texts)[:4000].upper()
        pos = {d: first.find(d.upper()) for d in dockets if first.find(d.upper()) >= 0}
        if pos:
            return min(pos, key=pos.get)
    ranked = sorted(dockets, key=lambda d: (not re.search(r"GAF|LDAC|LDAF", d), -len(d)))
    return ranked[0] if ranked else ""


def default_reference(name: str, text: str, doc_type: str, filed: str | None = None) -> str:
    """How to cite a filing in a request, e.g. 'the Company’s August 2026 LDAF filing'."""
    low = text[:5000].lower()
    if doc_type == "Information request response":
        m = re.search(r"\b((?:DPU|AG)[-\s]?\d{1,2}[-–]\d{1,3})", text)
        return f"the Company’s response to information request {m.group(1).replace('-', ' ', 1)}" if m \
            else "the Company’s information request responses"
    if "ldac" in low or "local distribution adjustment" in low:
        kind = "LDAF filing"
    elif "off-peak" in low or "ogaf" in low:
        kind = "Off-Peak GAF filing"
    elif doc_type.startswith("GAF"):
        kind = "Peak GAF filing" if "peak" in low else "GAF filing"
    elif doc_type == "Rate case":
        kind = "initial filing"
    else:
        kind = "filing"
    return f"the Company’s {filed + ' ' if filed else ''}{kind}"
