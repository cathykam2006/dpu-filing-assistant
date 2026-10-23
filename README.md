# DPU Filing Assistant

Drag in public utility filings submitted to the Massachusetts Department of Public Utilities (gas adjustment factor filings, LDAF filings, rate cases, information-request responses), ask questions in plain English, and run the routine parts of case discovery: pull key figures with automatic checks, draft information requests, track responses and compare filings. Everything cites the exact file, page, sheet or IR it came from, and an evaluation harness measures how often it is right.

**Live app:** [Open the app](https://dpu-filing-assistant-6tsrcudhmux8mpakqcnlux.streamlit.app/) [![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://dpu-filing-assistant-6tsrcudhmux8mpakqcnlux.streamlit.app/) · **Author:** Cathy Kam

> **Public filings only.** Files are processed in memory for the browser session and are not stored. With a model selected, questions and matching excerpts are sent to that provider (Groq, Google or Anthropic). In no-AI mode nothing leaves the app. Do not upload confidential or redacted material.

## Models: free by default, no key required

| Option | Cost | Notes |
|---|---|---|
| **Groq · GPT-OSS 120B / Llama 3.3 70B** | Free tier | Open models. The free tier allows about 8,000 tokens per minute, so the app budgets every request in tokens (number-heavy schedules use about 1 token per 2 characters), sends at most two earlier chat turns, and stops oversized requests before sending them |
| **Gemini Flash** | Free tier | On the free tier, Google may use inputs to improve its models |
| **Claude Sonnet / Haiku / Opus 5.5** | Paid API | Strongest on drafting and extraction |
| **No AI** | Free, no key | Search with highlighted passages, figures from tables and standard wording, the same arithmetic checks, template IRs, response checks, comparison, retrieval scoring |

The app picks the first provider with a key in Secrets (Groq, then Claude, then Gemini). If none has a key, it uses no-AI mode. Visitors can also paste their own key in the sidebar. Groq and Gemini are reached through their OpenAI-compatible endpoints. `rag/llm.py` translates the same calls, including forced tool use for structured output, so every feature works with every provider.

## What it does

- **Drag-and-drop** PDF, Excel (.xlsx/.xls), Word (.docx), CSV and text files, several at once.
- **Understands filing structure:**
  - Information-request responses are split per question, so asking about "DPU-1-3" retrieves exactly that request and response.
  - PDF tables are extracted row by row, so a rate or dollar figure stays next to its label.
  - Excel schedules keep their column headers, and citations give the real sheet and row numbers.
  - Docket numbers (e.g. D.P.U. 25-PGAF-GRID) and document type (GAF filing, rate case, IR response) are detected automatically.
- **Cited answers:** the model answers only from the retrieved excerpts, quotes figures exactly as filed, flags conflicts between an original and a revision, and says so when the filings don't contain the answer. Click "Sources cited" to read each excerpt.
- **Scope and export:** limit questions to selected files, and download the Q&A with citations.
- **Try it without your own files:** one click loads three fictional sample filings, or a real public NSTAR Gas LDAC filing.

## Verification: no unflagged mistakes, no silent misses

No AI system, and no analyst, is right 100% of the time. The app is built so that every mistake or miss is shown to you rather than hidden. None of these checks use AI:

| Check | What it does |
|---|---|
| **Schedule parser** | Numbered-line schedules (e.g. the LDAC summary: 22 lines × 10 rate classes) are read cell by cell by code, so the AI never extracts table cells. The column header is repeated in every excerpt, so the model always knows which rate class a number belongs to. |
| **Formula checks** | Every formula written in a schedule, e.g. "Sum of Line 5 through Line 18" or "Line 3 + Line 19 + Line 20 + Line 21", is recomputed for every rate class. A difference of one unit in the last decimal is reported as rounding; anything larger is a high-severity issue. |
| **Quote verification** | Every extracted figure's quote and number must appear on the cited page, or the figure is marked ✗. |
| **Coverage** | Lists every dollar amount, rate, percentage and large number in the filing that is *not* in the figures table, with page, line and rate class. Nothing the extraction skipped goes unnoticed. |
| **Answer check** | Every number in a chat answer is looked up in the source it cites. If the sentence names a rate class and the source is a schedule, the number must be in that class's column. Otherwise the answer gets a warning: "not in the cited source", "wrong column" or "not in the filings (check the arithmetic)". |

Tested on a real public filing included in `samples/`: NSTAR Gas LDAC, D.P.U. 26-PGAF-NSTAR (corrected). All 220 numbers were captured and verified, and all 20 formula checks tie. Corrupted copies in `tests/test_verify.py` show each check catching its error: a total changed by $0.01, a one-unit rounding difference, a skipped line, an answer quoting the C&I rate as residential, and an answer citing the wrong source.

## Discovery tools

| Tab | What it does | What is AI and what isn't |
|---|---|---|
| **Key figures & checks** | Every GAF, LDAF, cost-of-gas component, reconciliation balance, sales forecast and bill impact in a table, with the sentence it came from | The model extracts via a JSON schema (tool use); in no-AI mode, table rows and keyword patterns do. Code then confirms each quote and number appear verbatim on the cited page (✓/✗), checks that components sum to the total (0.5% tolerance), flags proposed-vs-prior changes over 10% and missing items |
| **Draft IRs** | Numbered requests (e.g. DPU-1-1…) in DPU style, each with a rationale and the page that prompted it; edit in the table, download as Word, text or CSV | The model drafts, starting from the automatic findings; in no-AI mode, templates are filled from the checks and figures |
| **Track responses** | For each IR response: answered / partial / non-responsive / objection / refers to attachment, gaps, attachments not yet uploaded, suggested follow-up; drafted IRs with no response are flagged | The model grades; in no-AI mode, rules flag objections, deferrals and short answers. Attachment and no-response checks are always code |
| **Compare filings** | Last year's vs this year's filing, figure by figure, with change and the page for each number | Join and arithmetic are code |

Draft IRs are a starting point for staff review. This demo is intended for public filings.

## Evaluation

`eval/questions_sample.csv` holds 17 questions on the sample filings with known answers, including two whose correct answer is "the filings don't say". Each question lists the expected figures and where they are.

```bash
python eval/run_eval.py                                  # retrieval only, free
python eval/run_eval.py --answers --baseline             # needs ANTHROPIC_API_KEY in your shell
python eval/run_eval.py --answers --model "Groq · GPT-OSS 120B (free)"   # needs GROQ_API_KEY: compare an open model with Claude
python eval/run_eval.py --questions eval/questions_real.csv --files filings --answers --baseline
```

It scores:
- **Retrieval:** whether the right passage was in the top k.
- **Figures:** whether the answer contains every expected value as filed. `0.912` matches `0.9120`, and `$3412580` matches `3,412,580`.
- **Citations:** whether a cited source is at the expected location and contains the value.
- **Baseline:** the same questions answered from the whole document pasted in with a plain prompt, for comparison.

It writes a Markdown report, a CSV of every answer, and a template for scoring ChatGPT or Claude.ai file upload by hand on the same questions. The **Evaluation** tab in the app runs the same thing.

To build a real set, add rows to a copy of the CSV in Excel: `id, category, question, files, expected_answer, expected_values, expected_location`. Aim for 30–50 questions over 3–5 public filings, and include some where the answer is not in the filing.

## How it works (retrieval-augmented generation)

```
upload → extract text, tables and sheets → split into cited sections
question → keyword search (BM25) tuned for docket/IR numbers, rate classes and dollar figures
        → top sections (or all of them, when the documents are small)
        → Claude, with instructions to answer only from those sources and cite [S1], [S2] …
```

| File | Role |
|---|---|
| `rag/ingest.py` | Readers for PDF (pdfplumber), Excel, Word, CSV; IR splitting; docket and document-type detection |
| `rag/search.py` | BM25 search with a filing-aware tokenizer; IR-number boosting; whole-document mode for small uploads |
| `rag/llm.py` | Providers (Groq, Gemini, Anthropic), size budgets, OpenAI-compatible adapter, error messages |
| `rag/answer.py` | Prompt, streaming answer, citation parsing |
| `rag/schedules.py`, `rag/verify.py` | Schedule parser, formula checks, coverage report, answer number check |
| `rag/rules.py` | No-AI mode: pattern-based figures, template IRs, response checks, passage highlighting |
| `rag/discovery.py` | Figure extraction (tool use + verbatim verification), checks, IR drafting, response grading, comparison |
| `rag/evaluate.py`, `eval/` | Question sets, scoring, baseline, reports |
| `app.py` | Streamlit interface: upload, Ask, Key figures, Draft IRs, Track responses, Compare, Evaluation |
| `samples/` | Fictional sample filings (`make_samples.py` regenerates them) |
| `tests/` | 44 tests on ingestion, search, prompts, discovery, scoring, the Groq/Gemini adapter, no-AI mode and verification (the API is faked, so tests are free) |

Scanned PDFs without a text layer are flagged; run OCR first for those pages to be searchable.

## Run locally

```bash
pip install -r requirements-dev.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # add your Anthropic API key
pytest -q
streamlit run app.py
```

## Deploy (Streamlit Community Cloud)

1. Push this repo to GitHub.
2. On share.streamlit.io, create an app from the repo with main file `app.py`.
3. In **App settings → Secrets**, add any of `GROQ_API_KEY` (free, console.groq.com/keys), `GEMINI_API_KEY` or `ANTHROPIC_API_KEY`. You can also add none, and the app runs in no-AI mode. Optional: `DEFAULT_MODEL`, `APP_PASSWORD`, `MAX_QUESTIONS` (see `.streamlit/secrets.toml.example`). `MAX_QUESTIONS` caps AI requests per session on your keys. Visitors who paste their own key are not capped.
4. For a paid key (Anthropic), set a monthly spend limit in the provider's console.

## Limitations and next steps

- Keyword search can miss paraphrases (e.g. "hedging" vs. "fixed-price purchases"); adding embedding-based search alongside BM25 would help.
- Image-only pages need OCR.
- Answers are only as complete as the uploaded documents; the assistant does not look up other filings.
