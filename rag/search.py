"""Keyword search tuned for regulatory filings.

BM25 ranks chunks by the words they share with the question. The tokenizer
keeps the things filings are full of intact: docket numbers (D.P.U. 25-PGAF),
IR numbers (DPU-1-3), rate classes (G-41), dollar amounts ($1,234,567) and
rates (0.9564). An IR number in the question pulls in every chunk of that
response, and when all uploaded documents together are small, everything is
sent to the model instead of a top-k subset.
"""
from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from .ingest import IR_PATTERN, Chunk

STOP = set("""a an the of to in on for and or is are was were be by with as at from that this these those
it its which what who whom how when where why do does did can could should would will shall may might
please tell me show give list find about any all there their them they our we you your i""".split())

WHOLE_CORPUS_CHARS = 120_000   # below this, send every chunk (~30k tokens)


def tokenize(text: str) -> list[str]:
    t = text.lower().replace("–", "-")
    tokens = re.findall(r"[a-z]{1,6}-?\d{1,3}-\d{1,3}"        # ir numbers dpu-1-3
                        r"|\d{2}-[a-z0-9]+(?:-[a-z0-9]+)*"     # dockets 25-pgaf-grid
                        r"|[a-z]-\d{1,3}[a-z]?"                # rate classes g-41, r-3
                        r"|\$?\d[\d,]*\.?\d*%?"                # numbers and money
                        r"|[a-z][a-z'&]+", t)                  # words
    out = []
    for tok in tokens:
        tok = tok.strip("$,").rstrip("%")
        if not tok or tok in STOP:
            continue
        out.append(tok.replace(",", ""))
        if tok.endswith("s") and len(tok) > 4:      # light plural folding
            out.append(tok[:-1])
    return out


class Index:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(c.text + " " + c.file) for c in chunks]) if chunks else None
        self.total_chars = sum(len(c.text) for c in chunks)

    def search(self, query: str, k: int = 12, files: list[str] | None = None) -> list[tuple[Chunk, float]]:
        if not self.chunks:
            return []
        pool = [i for i, c in enumerate(self.chunks) if not files or c.file in files]
        if not pool:
            return []
        if sum(len(self.chunks[i].text) for i in pool) <= WHOLE_CORPUS_CHARS:
            return [(self.chunks[i], 1.0) for i in pool]

        scores = self.bm25.get_scores(tokenize(query))
        wanted_irs = {m.group(1).upper().replace(" ", "-") for m in IR_PATTERN.finditer(query)}
        ranked = sorted(pool, key=lambda i: scores[i], reverse=True)
        picked = [i for i in pool if wanted_irs and (self.chunks[i].ir or "").upper() in wanted_irs]
        for i in ranked:
            if len(picked) >= k + len(wanted_irs) * 3:
                break
            if i not in picked and scores[i] > 0:
                picked.append(i)
        # keep reading order within each file so tables and their headings stay together
        picked.sort(key=lambda i: (self.chunks[i].file, i))
        return [(self.chunks[i], float(scores[i])) for i in picked]
