"""Ask the model a question about the retrieved filing excerpts, with citations."""
from __future__ import annotations

import re
from typing import Iterator

from .ingest import Chunk

# model choices live in rag/llm.py; any client with an Anthropic-style interface works here

SYSTEM = """You are a research assistant for staff reviewing public utility filings submitted to the \
Massachusetts Department of Public Utilities (DPU): gas adjustment factor (GAF/PGAF/OGAF) filings, \
LDAF filings, rate cases, and responses to information requests (IRs).

Answer ONLY from the numbered sources provided. Rules:
- Cite every factual statement with its source number in square brackets, e.g. [S3] or [S2][S5].
- Quote numbers exactly as filed (rates, dollar amounts, volumes, percentages) with their units and \
the period they apply to. Never round or recompute a filed figure unless asked; if you do a \
calculation, show it.
- If the sources do not contain the answer, say so plainly and say what document or schedule would \
be needed. Do not use outside knowledge to fill gaps.
- When sources disagree (e.g. an original filing and a revision), point it out and say which is later.
- Keep answers concise and organized. Use a short table when comparing figures across companies, \
periods or rate classes.
- These are public filings; do not speculate about confidential information."""


def build_context(hits: list[tuple[Chunk, float]]) -> tuple[str, list[Chunk]]:
    parts, sources = [], []
    for n, (c, _) in enumerate(hits, start=1):
        sources.append(c)
        parts.append(f'<source id="S{n}" file="{c.file}" location="{c.location}"'
                     + (f' ir="{c.ir}"' if c.ir else "") + f">\n{c.text}\n</source>")
    return "\n\n".join(parts), sources


def build_messages(question: str, context: str, history: list[dict], turns: int = 6) -> list[dict]:
    msgs = []
    for turn in (history[-turns:] if turns > 0 else []):   # short memory for follow-up questions
        msgs.append({"role": turn["role"], "content": turn["content"]})
    msgs.append({"role": "user", "content":
                 f"Sources from the uploaded filings:\n\n{context}\n\nQuestion: {question}"})
    return msgs


def stream_answer(client, model: str, question: str, hits, history: list[dict],
                  max_tokens: int = 1500, history_turns: int = 6) -> tuple[Iterator[str], list[Chunk]]:
    context, sources = build_context(hits)
    messages = build_messages(question, context, history, history_turns)

    def gen():
        with client.messages.stream(model=model, max_tokens=max_tokens, system=SYSTEM,
                                    messages=messages) as stream:
            for text in stream.text_stream:
                yield text

    return gen(), sources


def cited_ids(answer: str) -> list[int]:
    """Source numbers the answer actually cites, in first-use order."""
    seen = []
    for m in re.finditer(r"\[S(\d+)\]", answer):
        n = int(m.group(1))
        if n not in seen:
            seen.append(n)
    return seen
