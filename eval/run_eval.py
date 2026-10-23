"""Score the assistant on a question set.

    # retrieval only, free, no API key needed
    python eval/run_eval.py --questions eval/questions_sample.csv --files samples

    # full run: answers, citations, and the whole-document baseline
    export ANTHROPIC_API_KEY=...        # never commit this
    python eval/run_eval.py --questions eval/questions_real.csv --files filings --answers --baseline

Writes eval/results/<name>_report.md, <name>_results.csv and
<name>_manual_comparison.csv (for scoring ChatGPT / Claude.ai file upload by hand).
"""
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag import answer, evaluate, ingest  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=str(ROOT / "eval" / "questions_sample.csv"))
    ap.add_argument("--files", default=str(ROOT / "samples"), help="folder holding the filings")
    ap.add_argument("--answers", action="store_true", help="call Claude and score answers (costs API credit)")
    ap.add_argument("--baseline", action="store_true", help="also score the whole-document baseline")
    ap.add_argument("--model", default=answer.MODELS[answer.DEFAULT_MODEL])
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--out", default=str(ROOT / "eval" / "results"))
    a = ap.parse_args()

    qs = evaluate.load_questions(a.questions)
    names = sorted({f for q in qs for f in q.files})
    folder = Path(a.files)
    chunks = []
    for n in names:
        p = folder / n
        if not p.exists():
            print(f"  missing file: {p} (questions that need it will score as misses)")
            continue
        chunks += ingest.load(n, p.read_bytes()).chunks

    client = None
    if a.answers:
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            sys.exit("Set ANTHROPIC_API_KEY to score answers (or drop --answers for retrieval only).")
        import anthropic
        client = anthropic.Anthropic(api_key=key)

    res = evaluate.run(qs, chunks, client=client, model=a.model, k=a.k, baseline=a.baseline,
                       progress=lambda i, n, q: print(f"  [{i}/{n}] {q.id} {q.question[:70]}"))
    df = evaluate.results_frame(res)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(a.questions).stem
    notes = (f"{len(qs)} questions · {len(names)} files · model {a.model if client else '(retrieval only)'} "
             f"· top {a.k} passages · {datetime.now():%Y-%m-%d %H:%M}")
    (out / f"{stem}_report.md").write_text(evaluate.report_markdown(df, f"Evaluation: {stem}", notes))
    df.to_csv(out / f"{stem}_results.csv", index=False)
    (out / f"{stem}_manual_comparison.csv").write_text(evaluate.manual_template(qs))
    print()
    print(evaluate.summary(df).to_string(index=False))
    print(f"\nReport: {out / f'{stem}_report.md'}")


if __name__ == "__main__":
    main()
