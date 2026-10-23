"""Create FICTIONAL sample filings for trying the app and for tests.

"Sample Gas Company" and every number here are invented. They mimic the
layout of a seasonal GAF filing, an information-request response set and
an Excel cost schedule.

    python samples/make_samples.py
"""
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

OUT = Path(__file__).parent
ST = getSampleStyleSheet()
NOTE = "SAMPLE DOCUMENT – fictional company and data, for demonstration only."


def table(rows):
    t = Table(rows)
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                           ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey)]))
    return t


def gaf_filing(path):
    doc = SimpleDocTemplate(str(path), pagesize=letter)
    p = [Paragraph(NOTE, ST["Italic"]),
         Paragraph("Sample Gas Company", ST["Title"]),
         Paragraph("D.P.U. 99-PGAF-SAMPLE · Peak Period Gas Adjustment Factor Filing, "
                   "November 1, 2026 – April 30, 2027", ST["Heading2"]),
         Paragraph("Pursuant to 220 CMR 6.00, Sample Gas Company submits its proposed Peak Period Gas "
                   "Adjustment Factor (PGAF). The proposed residential heating GAF is $0.9120 per therm, "
                   "compared with $0.8455 per therm for the 2025-2026 peak period, an increase of $0.0665 "
                   "per therm (7.9%). The increase is driven primarily by higher forecast commodity costs "
                   "at the Algonquin citygate and a prior-period under-collection of $3,412,580.",
                   ST["BodyText"]),
         Spacer(1, 12),
         table([["Rate class", "2025-26 GAF ($/therm)", "Proposed 2026-27 GAF ($/therm)", "Change"],
                ["Residential heating (R-3)", "0.8455", "0.9120", "+0.0665"],
                ["Residential non-heating (R-1)", "0.7710", "0.8302", "+0.0592"],
                ["C&I high load factor (G-41)", "0.8455", "0.9120", "+0.0665"],
                ["C&I low load factor (G-51)", "0.7710", "0.8302", "+0.0592"]]),
         PageBreak(),
         Paragraph("Schedule 2 – Cost of Gas Components", ST["Heading2"]),
         Paragraph("Total forecast peak-period cost of gas is $142,650,000, consisting of commodity "
                   "costs of $98,300,000, pipeline and storage demand charges of $36,120,000, LNG "
                   "peaking supply of $4,817,420, and the reconciliation adjustment of $3,412,580. "
                   "Forecast peak-period sales are 156,410,000 therms.", ST["BodyText"]),
         Spacer(1, 12),
         table([["Component", "Amount ($)"],
                ["Commodity", "98,300,000"], ["Pipeline & storage demand", "36,120,000"],
                ["LNG peaking", "4,817,420"], ["Prior-period reconciliation", "3,412,580"],
                ["Total", "142,650,000"]]),
         Paragraph("The Company also proposes a Local Distribution Adjustment Factor (LDAF) of "
                   "$0.4310 per therm for residential customers effective November 1, 2026.",
                   ST["BodyText"])]
    doc.build(p)


def ir_responses(path):
    doc = SimpleDocTemplate(str(path), pagesize=letter)
    qa = [("DPU-1-1", "Please explain the drivers of the projected increase in commodity costs.",
           "Commodity costs rise because the NYMEX strip for November 2026 through March 2027 averaged "
           "$4.12 per MMBtu at the time of filing, compared with $3.45 per MMBtu a year earlier, and "
           "Algonquin basis is forecast at $3.80 per MMBtu for December–February."),
          ("DPU-1-2", "Provide the calculation of the prior-period under-collection.",
           "The under-collection of $3,412,580 equals actual 2025-26 peak gas costs of $131,904,210 less "
           "GAF revenues of $128,491,630. See Attachment DPU-1-2 for monthly detail."),
          ("DPU-1-3", "Has the Company hedged any portion of its winter supply?",
           "Yes. The Company has fixed the price on 28% of forecast winter purchases through financial "
           "hedges at a weighted average of $3.95 per MMBtu, consistent with its approved hedging plan."),
          ("AG-1-1", "What is the bill impact for a typical residential heating customer?",
           "A typical residential heating customer using 120 therms in January would see a bill increase "
           "of $7.98 (from $283.41 to $291.39), or 2.8%.")]
    p = [Paragraph(NOTE, ST["Italic"]), Paragraph("Sample Gas Company", ST["Title"]),
         Paragraph("D.P.U. 99-PGAF-SAMPLE · Responses to Information Requests", ST["Heading2"])]
    for ir, q, a in qa:
        p += [Spacer(1, 10), Paragraph(f"Information Request {ir}", ST["Heading3"]),
              Paragraph(f"<b>Request:</b> {q}", ST["BodyText"]),
              Paragraph(f"<b>Response:</b> {a}", ST["BodyText"])]
    doc.build(p)


def schedule_xlsx(path):
    months = ["Nov-26", "Dec-26", "Jan-27", "Feb-27", "Mar-27", "Apr-27"]
    therms = [18_420_000, 28_910_000, 33_870_000, 30_120_000, 26_040_000, 19_050_000]
    price = [3.71, 4.05, 4.38, 4.21, 3.89, 3.52]
    df = pd.DataFrame({"Month": months, "Forecast sales (therms)": therms,
                       "Commodity price ($/MMBtu)": price,
                       "Commodity cost ($)": [round(t / 10 * p) for t, p in zip(therms, price)]})
    with pd.ExcelWriter(path) as xw:
        pd.DataFrame([[NOTE], ["Schedule 3 – Monthly Forecast Sales and Commodity Cost"]]).to_excel(
            xw, sheet_name="Schedule 3", header=False, index=False)
        df.to_excel(xw, sheet_name="Schedule 3", startrow=3, index=False)


def sample_order(path):
    """Fictional prior-year order with directives for the next filing (for the Compliance tab)."""
    doc = SimpleDocTemplate(str(path), pagesize=letter)
    p = [Paragraph(NOTE, ST["Italic"]),
         Paragraph("D.P.U. 98-PGAF-SAMPLE", ST["Heading2"]),
         Paragraph("Petition of Sample Gas Company for approval of its Peak Period Gas Adjustment Factor", ST["Heading3"]),
         Paragraph("ORDER ON PEAK PERIOD GAS ADJUSTMENT FACTOR — October 30, 2025", ST["Heading3"]),
         Paragraph("The Department has reviewed the Company's filing and responses to information requests. "
                   "In its next peak period filing, the Company shall include a typical bill impact analysis "
                   "for each rate class, including residential and C&amp;I customers.", ST["BodyText"]),
         Paragraph("The Company shall provide monthly detail supporting any prior-period reconciliation "
                   "balance as Attachment DPU-1-2, in Excel format with formulas intact.", ST["BodyText"]),
         Paragraph("The Company is directed to report the percentage of forecast winter supply that is hedged "
                   "and the weighted average hedge price.", ST["BodyText"]),
         Paragraph("The Department approves a residential Local Distribution Adjustment Factor of $0.4310 per "
                   "therm, effective November 1, 2026.", ST["BodyText"]),
         Paragraph("The Company shall include Schedule 4, a summary of its hedging program, with each annual "
                   "filing.", ST["BodyText"]),
         Paragraph("Accordingly, after due notice, hearing and consideration, it is", ST["BodyText"]),
         Paragraph("ORDERED: That the C&amp;I low load factor (G-51) Gas Adjustment Factor shall not exceed "
                   "$0.8200 per therm for the 2026-2027 peak period; and it is", ST["BodyText"]),
         Paragraph("FURTHER ORDERED: That Sample Gas Company shall file its 2026-2027 peak period filing no "
                   "later than September 1, 2026.", ST["BodyText"]),
         Paragraph("By Order of the Department (fictional).", ST["Italic"])]
    doc.build(p)


def sample_cover_letter(path):
    doc = SimpleDocTemplate(str(path), pagesize=letter)
    p = [Paragraph(NOTE, ST["Italic"]),
         Paragraph("Sample Gas Company — Cover letter", ST["Heading2"]),
         Paragraph("Re: D.P.U. 99-PGAF-SAMPLE, Peak Period Gas Adjustment Factor Filing", ST["BodyText"]),
         Paragraph("Dear Secretary: Enclosed for filing please find the following:", ST["BodyText"])]
    for item in ["Schedule 1 – Summary of Proposed Gas Adjustment Factors",
                 "Schedule 2 – Cost of Gas Components",
                 "Schedule 3 – Monthly Forecast Sales and Commodity Cost",
                 "Schedule 4 – Hedging Program Summary",
                 "Attachment A – Typical Bill Impacts by Rate Class"]:
        p.append(Paragraph(item, ST["BodyText"]))
    p.append(Paragraph("Thank you for your attention to this matter.", ST["BodyText"]))
    doc.build(p)


if __name__ == "__main__":
    gaf_filing(OUT / "sample_pgaf_filing.pdf")
    ir_responses(OUT / "sample_ir_responses.pdf")
    schedule_xlsx(OUT / "sample_schedule3.xlsx")
    sample_order(OUT / "order_sample_98-PGAF-SAMPLE.pdf")
    sample_cover_letter(OUT / "order_sample_cover_letter.pdf")
    print("Samples written to", OUT)
