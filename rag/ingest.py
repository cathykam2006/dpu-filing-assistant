"""Turn uploaded filings into searchable chunks with exact citations.

Each chunk remembers its file and location (PDF page, Excel sheet and rows,
Word paragraph range) so every answer can point back to the source.

Regulatory filings need a few special touches:
* Information-request (IR) responses are split per question, so
  "DPU-1-3" retrieves exactly that question and answer.
* PDF tables are extracted as rows (pipe-separated) so dollar figures and
  their labels stay together.
* Excel schedules keep the header row with every block of rows.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .schedules import header_line

MAX_CHARS = 1800        # target chunk size
OVERLAP = 250           # characters carried into the next chunk
ROWS_PER_CHUNK = 40     # Excel/CSV rows per chunk

# DPU-1-3, AG-2-14, DPU 3-1, IR DPU-1-3 ...
IR_PATTERN = re.compile(
    r"\b(?:Information\s+Request\s+)?((?:DPU|AG|DOER|LEI|CLF|NCLC|[A-Z]{2,5})[-\s]?\d{1,2}[-–]\d{1,3})\b")
DOCKET_PATTERN = re.compile(r"\bD\.?\s?P\.?\s?U\.?\s*(\d{2}-[A-Z0-9-]+)", re.I)

DOC_TYPES = [
    ("Information request response", r"information\s+request|response\s+to\s+(dpu|ag)|\bIR\b\s*[A-Z]+-\d"),
    ("GAF / cost of gas filing", r"gas\s+adjustment\s+factor|\bp?gaf\b|\bogaf\b|cost\s+of\s+gas|\bldaf\b"),
    ("Rate case", r"revenue\s+requirement|cost\s+of\s+service|rate\s+base|base\s+distribution\s+rate|"
                  r"pre-?filed\s+(direct\s+)?testimony"),
]


@dataclass
class Chunk:
    text: str
    file: str
    location: str              # "p. 12", "Sheet 'Sch 1', rows 2-41", "¶ 10-25"
    page: int | None = None
    ir: str | None = None      # e.g. "DPU-1-3"
    kind: str = "text"         # text | table | spreadsheet
    meta: dict = field(default_factory=dict)

    def citation(self) -> str:
        bits = [self.file, self.location]
        if self.ir:
            bits.append(f"IR {self.ir}")
        return " · ".join(bits)


@dataclass
class Document:
    name: str
    doc_type: str
    pages: int
    chunks: list[Chunk]
    dockets: list[str]
    warning: str | None = None
    page_texts: list[tuple[str, str]] = field(default_factory=list)   # (location, full page text)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _clean(text: str) -> str:
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\ufffe\uffff]", " ", text.replace("\x02", "-"))
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split(text: str, max_chars: int = MAX_CHARS, overlap: int = OVERLAP) -> list[str]:
    """Split on paragraph boundaries, then sentences, with overlap."""
    if len(text) <= max_chars:
        return [text] if text.strip() else []
    paras = re.split(r"\n\s*\n", text)
    out, cur = [], ""
    for p in paras:
        tabular = p.count("\n") >= 5 and len(p) / (p.count("\n") + 1) < 400
        if len(p) > max_chars and tabular:     # schedules: keep each row whole
            parts = p.split("\n")
        elif len(p) > max_chars:               # long prose: split on sentences
            parts = re.split(r"(?<=[.;:])\s+", p)
        else:
            parts = [p]
        sep = "\n" if len(parts) > 1 and tabular else "\n\n"
        for part in parts:
            if len(cur) + len(part) + 2 > max_chars and cur:
                out.append(cur.strip())
                if tabular:                    # carry the last whole rows, never half a row
                    tail = cur.split("\n")[-2:]
                    cur = "\n".join(tail)[-overlap:].split("\n", 1)[-1] + "\n" + part
                else:
                    cur = cur[-overlap:] + "\n" + part
            else:
                cur = f"{cur}{sep}{part}" if cur else part
    if cur.strip():
        out.append(cur.strip())
    return out


def with_header(pieces: list[str], text: str) -> list[str]:
    """Repeat a schedule's column header (rate classes) at the top of every piece of the page."""
    head = header_line(text)
    return [p if not head or head in p else f"{head}\n{p}" for p in pieces]


def classify(text: str, name: str) -> str:
    sample = (name + " " + text[:20000]).lower()
    for label, pat in DOC_TYPES:
        if re.search(pat, sample, re.I):
            return label
    return "Other filing"


def find_dockets(text: str) -> list[str]:
    return sorted({m.group(1).upper() for m in DOCKET_PATTERN.finditer(text[:50000])})


def split_by_ir(text: str) -> list[tuple[str | None, str]]:
    """Split a response document at each IR heading ("Information Request DPU-1-3")."""
    heads = [m for m in re.finditer(
        r"(?im)^\s*(?:information\s+request|request\s+no\.?|ir)?\s*[:#]?\s*"
        r"((?:DPU|AG|DOER|[A-Z]{2,5})[-\s]?\d{1,2}[-–]\d{1,3})\b", text)]
    if len(heads) < 2:
        m = IR_PATTERN.search(text)
        return [(m.group(1).replace(" ", "-") if m else None, text)]
    parts = []
    if heads[0].start() > 0 and text[:heads[0].start()].strip():
        parts.append((None, text[:heads[0].start()]))
    for i, h in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        parts.append((h.group(1).replace(" ", "-").replace("–", "-").upper(), text[h.start():end]))
    return parts


# --------------------------------------------------------------------------
# readers
# --------------------------------------------------------------------------
NUMERIC_LINE = re.compile(r"(?:\$\s?\(?\d|\b\d[\d,]*\.\d+\b|\b\d{1,3}(?:,\d{3})+\b)")


def needs_layout(text: str, n_paths: int) -> bool:
    """Pages where reading order by position matters: schedules and tables (rows of numbers, a rate-class
    header, or drawn ruling lines). Prose pages can use the fast reader; its text is the same."""
    if n_paths >= 6 or header_line(text):
        return True
    rows = sum(1 for ln in text.splitlines() if len(NUMERIC_LINE.findall(ln)) >= 3)
    return rows >= 2


def _fast_pages(data: bytes):
    """Per page: (text, number of drawn path objects) from pdfium; None if pdfium cannot open the file."""
    try:
        import pypdfium2 as pdfium
        import pypdfium2.raw as raw
        doc = pdfium.PdfDocument(data)
    except Exception:
        return None
    out = []
    for i in range(len(doc)):
        try:
            page = doc[i]
            text = page.get_textpage().get_text_range().replace("\r\n", "\n").replace("\r", "\n")
            paths = sum(1 for _ in page.get_objects(filter=[raw.FPDF_PAGEOBJ_PATH], max_depth=2))
        except Exception:
            text, paths = None, 99                       # force the careful reader for this page
        out.append((text, paths))
    return out


def read_pdf(name: str, data: bytes, progress=None) -> Document:
    """Prose pages are read with pdfium (fast); pages with tables or schedules with pdfplumber, which keeps
    each value in its row and column. progress(done, total) is called after every page."""
    import pdfplumber

    chunks, all_text, pages, empty, page_texts = [], [], 0, 0, []
    current_ir = None
    fast = _fast_pages(data)
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        pages = len(pdf.pages)
        for i, page in enumerate(pdf.pages, start=1):
            quick = fast[i - 1] if fast and i <= len(fast) else (None, 99)
            careful = quick[0] is None or needs_layout(_clean(quick[0]), quick[1])
            if careful:
                try:
                    text = page.extract_text() or ""
                except Exception:
                    text = ""
            else:
                text = quick[0]
            text = _clean(text)
            if not text:
                empty += 1
            all_text.append(text)
            page_texts.append((f"p. {i}", text))
            for ir, part in split_by_ir(text) if text else []:
                current_ir = ir or current_ir
                for piece in with_header(_split(part), text):   # every piece of a schedule keeps its columns
                    chunks.append(Chunk(piece, name, f"p. {i}", page=i, ir=current_ir))
            if careful:
                try:
                    for t in page.extract_tables() or []:
                        rows = [" | ".join(str(c or "").strip() for c in r) for r in t if any(r)]
                        if len(rows) >= 2:
                            body = "\n".join(rows)
                            for piece in _split(body):
                                chunks.append(Chunk(piece, name, f"p. {i} (table)", page=i,
                                                    ir=current_ir, kind="table"))
                except Exception:
                    pass
            if progress:
                progress(i, pages)
    joined = "\n".join(all_text)
    warning = None
    if pages and empty / pages > 0.5:
        warning = (f"{empty} of {pages} pages have no extractable text (likely scanned images). "
                   "Run OCR first (e.g. Adobe 'Recognize Text') for those pages to be searchable.")
    return Document(name, classify(joined, name), pages, chunks, find_dockets(joined), warning, page_texts)


def _frame_chunks(df: pd.DataFrame, name: str, where: str, first_row: int = 2) -> list[Chunk]:
    df = df.dropna(how="all").dropna(axis=1, how="all")
    if df.empty:
        return []
    df = df.astype(str).replace({"nan": "", "None": "", "NaT": ""})
    header = " | ".join(str(c) for c in df.columns)
    out = []
    for start in range(0, len(df), ROWS_PER_CHUNK):
        block = df.iloc[start:start + ROWS_PER_CHUNK]
        lines = [" | ".join(v.strip() for v in row) for row in block.values.tolist()]
        text = f"{where}\nColumns: {header}\n" + "\n".join(lines)
        out.append(Chunk(text[:MAX_CHARS * 2], name,
                         f"{where.split(' (')[0]}, rows {first_row + start}-"
                         f"{first_row + start + len(block) - 1}",
                         kind="spreadsheet"))
    return out


def read_excel(name: str, data: bytes) -> Document:
    xls = pd.ExcelFile(io.BytesIO(data))
    chunks, text = [], []
    for sheet in xls.sheet_names:
        raw = pd.read_excel(xls, sheet, header=None)
        if raw.dropna(how="all").empty:
            continue
        # use the first mostly-filled row as the header (schedules often have titles above it)
        filled = raw.notna().sum(axis=1)
        hdr = int(filled[filled >= max(2, filled.max() * 0.6)].index[0])
        title = " ".join(str(v) for v in raw.iloc[:hdr].stack().dropna().tolist()
                         if str(v).strip() and str(v) != "nan")[:200]
        df = raw.iloc[hdr + 1:].copy()
        df.columns = [str(c) if pd.notna(c) else f"col{j}" for j, c in enumerate(raw.iloc[hdr])]
        where = f"Sheet '{sheet}'" + (f" ({title})" if title else "")
        chunks += _frame_chunks(df, name, where, first_row=hdr + 2)
        text.append(where + " " + " ".join(map(str, raw.iloc[:20].stack().dropna().tolist())))
    joined = "\n".join(text)
    return Document(name, classify(joined, name), len(xls.sheet_names), chunks, find_dockets(joined))


def read_csv(name: str, data: bytes) -> Document:
    df = pd.read_csv(io.BytesIO(data))
    chunks = _frame_chunks(df, name, "CSV")
    return Document(name, classify(" ".join(df.columns), name), 1, chunks, [])


def read_docx(name: str, data: bytes) -> Document:
    import docx

    d = docx.Document(io.BytesIO(data))
    paras = [p.text for p in d.paragraphs]
    chunks, buf, start = [], [], 0
    for i, p in enumerate(paras + ["\n"]):
        buf.append(p)
        if sum(len(x) for x in buf) > MAX_CHARS or i == len(paras):
            text = _clean("\n".join(buf))
            for ir, part in split_by_ir(text) if text else []:
                for piece in _split(part):
                    chunks.append(Chunk(piece, name, f"¶ {start + 1}-{i + 1}", ir=ir))
            buf, start = [], i + 1
    for ti, t in enumerate(d.tables, start=1):
        rows = [" | ".join(c.text.strip() for c in r.cells) for r in t.rows]
        for piece in _split("\n".join(rows)):
            chunks.append(Chunk(piece, name, f"table {ti}", kind="table"))
    joined = "\n".join(paras)
    return Document(name, classify(joined, name), len(paras), chunks, find_dockets(joined))


def read_text(name: str, data: bytes) -> Document:
    text = _clean(data.decode("utf-8", errors="ignore"))
    chunks = [Chunk(p, name, f"part {i}", ir=ir)
              for i, (ir, part) in enumerate(split_by_ir(text), start=1) for p in with_header(_split(part), text)]
    return Document(name, classify(text, name), 1, chunks, find_dockets(text), page_texts=[("text", text)])


READERS = {".pdf": read_pdf, ".xlsx": read_excel, ".xlsm": read_excel, ".xls": read_excel,
           ".csv": read_csv, ".docx": read_docx, ".txt": read_text, ".md": read_text}
SUPPORTED = sorted(READERS)


def load(name: str, data: bytes, progress=None) -> Document:
    ext = Path(name).suffix.lower()
    if ext not in READERS:
        raise ValueError(f"Unsupported file type {ext}. Supported: {', '.join(SUPPORTED)}")
    doc = READERS[ext](name, data, progress) if ext == ".pdf" else READERS[ext](name, data)
    if not doc.chunks and not doc.warning:
        doc.warning = "No text could be extracted from this file."
    return doc
