"""Formula checks, coverage and answer checks on a real public filing:
NSTAR Gas LDAC summary, D.P.U. 26-PGAF-NSTAR (corrected), 22 lines x 10 rate classes.
Corrupted copies prove each check catches the error it is meant to catch."""
from pathlib import Path

from rag import discovery as D
from rag import evaluate as E
from rag import ingest, rules
from rag import schedules as S
from rag import verify as V

REAL = Path(__file__).resolve().parents[1] / "samples" / "real_NSTAR_Gas_LDAC_DPU_26-PGAF-NSTAR_corrected.pdf"
COLS = ["R-1", "R-2", "R-3", "R-4", "G-41", "G-51", "G-42", "G-52", "G-43", "G-53"]


def real():
    return ingest.load(REAL.name, REAL.read_bytes())


def page_text():
    return real().page_texts[0][1]


def as_text(text, name="copy.txt"):
    return ingest.load(name, text.encode())


def test_schedule_parsed_cell_by_cell():
    sch = S.parse(page_text())
    assert sch.columns == COLS and sorted(sch.rows) == list(range(1, 23))
    assert sch.cell(19, "R-3") == 0.7434 and sch.cell(19, "G-53") == 0.3376
    assert sch.cell(12, "G-53") == -0.0008                 # ($0.0008) is a credit
    assert sch.cell(4, "G-53") == 0.35                      # "$ 0.35" with a space
    assert sch.cell(1, "G-53") == 305.0
    assert sch.rows[19].reference == "Sum of Line 5 through Line 18"


def test_every_chunk_keeps_columns_and_whole_rows():
    d = real()
    assert d.dockets[-1] == "26-PGAF-NSTAR" and len(d.chunks) >= 2
    seen = set()
    for c in d.chunks:
        sch = S.parse(c.text)
        assert sch is not None and sch.columns == COLS
        for r in sch.rows.values():
            assert len(r.values) == 10                      # no row cut in half
        seen |= set(sch.rows)
    assert seen == set(range(1, 23))


def test_formulas_all_tie_on_the_filed_schedule():
    checks = V.formula_checks([real()])
    assert len(checks) == 20                                # lines 19 and 22, ten classes each
    assert set(checks["Result"]) == {"✓ ties"}
    assert set(checks["Line"]) == {19, 22}


def test_corrupted_total_is_caught():
    bad = page_text().replace("LDAF = Total Local Distribution Adjustment Factor (Excluding BDRF) $0.7434",
                              "LDAF = Total Local Distribution Adjustment Factor (Excluding BDRF) $0.7534")
    checks = V.formula_checks([as_text(bad)])
    wrong = checks[checks["Result"] == "✗ does not tie"]
    # line 19 R-1 no longer equals lines 5-18, and line 22 R-1 no longer equals 3+19+20+21
    assert set(zip(wrong["Line"], wrong["Rate class"])) == {(19, "R-1"), (22, "R-1")}
    issues = V.formula_issues(checks)
    assert len(issues) == 2 and all(i.severity == "high" for i in issues)
    assert "filed 0.7534, recomputed 0.7434" in issues[0].description


def test_one_unit_difference_is_reported_as_rounding_not_error():
    near = page_text().replace("Sum of Line 5 through Line 18", "Sum of Line 5 through Line 18", 1).replace(
        "(Excluding BDRF) $0.7434", "(Excluding BDRF) $0.7435")
    res = V.formula_checks([as_text(near)])
    row = res[(res["Line"] == 19) & (res["Rate class"] == "R-1")].iloc[0]
    assert row["Result"] == "≈ rounding"


def test_coverage_lists_what_extraction_skipped():
    d = real()
    figs = V.schedule_figures(d)
    assert len(figs) == 220 and all(f.verified for f in figs)
    full = V.coverage([d], figs)
    assert len(full) == 220 and full["In figures table"].all()
    partial = [f for f in figs if not f.label.startswith("Line 9:")]
    cov = V.coverage([d], partial)
    missed = cov[~cov["In figures table"]]
    assert len(missed) == 10 and set(missed["Line / column"]) == {f"Line 9 · {c}" for c in COLS}


def test_coverage_counts_numbers_outside_schedules():
    d = as_text("Total forecast cost of gas is $142,650,000, of which LNG is $4,817,420. Filed August 2026, "
                "D.P.U. 26-109, page 3.")
    fig = D.Figure("cost_of_gas_total", "Total", 142_650_000, "$", file=d.name)
    cov = V.coverage([d], [fig])
    assert list(cov["Number"]) == ["$142,650,000", "$4,817,420"]          # no dates, dockets or pages
    assert list(cov["In figures table"]) == [True, False]


def test_answer_numbers_checked_against_citation_and_column():
    src = real().chunks
    last = len(src)
    ans = (f"The R-3 LDAF is $0.7434 per therm [S{last}], and G-41 is $0.5085 [S{last}]. "
           "The EES for R-1 is $0.1385 [S1]. The total peak rate for R-1 is $2.5000 "
           f"[S{last}].\n\n| Rate class | LDAF |\n|---|---|\n| G-43 | $0.4254 [S{last}] |")
    got = {(c.number, c.status) for c in V.check_answer(ans, src)}
    assert ("$0.7434", "verified") in got and ("$0.5085", "verified") in got
    assert ("$0.1385", "wrong column") in got                 # the C&I EES, not residential
    assert ("$2.5000", "not found") in got                    # filed figure is $2.9417
    assert ("$0.4254", "wrong column") in got                 # that is G-42/G-52, not G-43
    assert not V.answer_ok(V.check_answer(ans, src))


def test_answer_citing_the_wrong_source_is_flagged():
    chunks = ingest.load("a.txt", b"LNG peaking supply of $4,817,420.\n\n" + b"x " * 1000 +
                         b"\n\nCommodity costs of $98,300,000.").chunks
    assert len(chunks) >= 2
    checks = V.check_answer(f"LNG costs $4,817,420 [S{len(chunks)}].", chunks)
    assert checks[0].status == "other source"


def test_no_ai_mode_reads_schedules_not_prose_patterns():
    d = real()
    assert rules.extract_figures(d.chunks) == []              # schedule chunks are left to the parser
    figs = D.dedupe(V.schedule_figures(d))
    assert len(figs) == 220                                   # lines with equal values are not merged
    ldaf = {f.rate_class: f.value for f in figs if f.metric == "ldaf"}
    assert ldaf["R-1"] == 0.7434 and ldaf["G-43"] == 0.3376


def test_evaluation_scores_traced_numbers():
    class Fake:
        def __init__(self, text):
            self.text = text
            self.messages = self

        def stream(self, **kw):
            text = self.text

            class S_:
                text_stream = iter([text])
                def __enter__(s): return s
                def __exit__(s, *a): return False
            return S_()

    d = real()
    q = E.Question("N1", "What is the R-3 LDAF?", [d.name], "", ["0.7434"], ["p. 1"])
    n = len(d.chunks)                                         # the last chunk holds line 19
    good = E.run([q], d.chunks, client=Fake(f"The R-3 LDAF is $0.7434 [S{n}]."))[0]
    bad = E.run([q], d.chunks, client=Fake(f"The R-3 LDAF is $0.7434 [S{n}], versus $0.9999 last year [S{n}]."))[0]
    assert good.numbers_traced is True and bad.numbers_traced is False
    assert "Every number in the answer found in its cited source" in E.summary(E.results_frame([good, bad]))["Measure"].tolist()
