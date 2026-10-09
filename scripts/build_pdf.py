#!/usr/bin/env python3
"""Render the dataset as a human-readable PDF (requires reportlab).

Usage: python scripts/build_pdf.py [in.jsonl] [out.pdf]
"""
import json
import sys
from collections import Counter
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (HRFlowable, KeepTogether, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

SRC = sys.argv[1] if len(sys.argv) > 1 else "dataset/clarifying_question_dataset.jsonl"
OUT = sys.argv[2] if len(sys.argv) > 2 else "dataset/clarifying_question_dataset.pdf"

rows = [json.loads(line) for line in open(SRC, encoding="utf-8")]
n_exec = sum(r["can_execute"] for r in rows)
type_counts = Counter(r["ambiguity_type"] for r in rows if not r["can_execute"])

GREEN_BG, ORANGE_BG = colors.HexColor("#e6f4ea"), colors.HexColor("#fdf0e2")
GREY, PROMPT_BG = colors.HexColor("#555555"), colors.HexColor("#f4f4f4")
Q_BG, Q_BLUE = colors.HexColor("#eaf1fb"), colors.HexColor("#1a4f8b")

ss = getSampleStyleSheet()


def style(name, **kw):
    return ParagraphStyle(name, parent=ss["Normal"], **kw)


st_title = ParagraphStyle("t", parent=ss["Title"], fontSize=22, spaceAfter=6)
st_sub = style("sub", fontSize=10.5, textColor=GREY, spaceAfter=2)
st_meta = style("meta", fontSize=8.5, textColor=GREY)
st_meta_r = style("metaR", fontSize=8.5, textColor=GREY, alignment=2)
st_prompt = style("prompt", fontName="Courier", fontSize=8.5, leading=11.5,
                  backColor=PROMPT_BG, borderPadding=5, spaceBefore=3, spaceAfter=3)
st_label = style("label", fontSize=8, textColor=GREY, spaceBefore=3)
st_bullet = style("bullet", fontSize=8.5, leading=11, leftIndent=12, bulletIndent=4)
st_q = style("q", fontSize=9, leading=12, textColor=Q_BLUE, backColor=Q_BG,
             borderPadding=5, spaceBefore=3)


def badge(i, r):
    if r["can_execute"]:
        chip, fg, bg, extra = "EXECUTABLE", "#1a7f37", GREEN_BG, ""
    else:
        chip, fg, bg = "NEEDS CLARIFICATION", "#b35900", ORANGE_BG
        extra = f'&nbsp;&nbsp;<font color="#888888">type:</font> {r["ambiguity_type"]}'
    left = Paragraph(f'<b>#{i}</b>&nbsp;&nbsp;<font color="{fg}"><b>{chip}</b></font>{extra}', st_meta)
    right = Paragraph(f'confidence {r["confidence"]:.2f}', st_meta_r)
    t = Table([[left, right]], colWidths=[5.1 * inch, 1.6 * inch])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), bg),
                           ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                           ("LEFTPADDING", (0, 0), (-1, -1), 6),
                           ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                           ("TOPPADDING", (0, 0), (-1, -1), 3),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
    return t


def entry(i, r):
    parts = [badge(i, r), Paragraph(escape(r["prompt"]), st_prompt)]
    if not r["can_execute"]:
        parts.append(Paragraph("<b>Ambiguities</b>", st_label))
        parts += [Paragraph(escape(a), st_bullet, bulletText="•") for a in r["ambiguities"]]
        parts.append(Paragraph("<b>Missing information</b>", st_label))
        parts += [Paragraph(escape(m), st_bullet, bulletText="•") for m in r["missing_information"]]
        parts.append(Paragraph("<b>Clarifying question:</b> " + escape(r["clarifying_question"]), st_q))
    parts.append(Spacer(1, 10))
    return KeepTogether(parts)


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(GREY)
    canvas.drawCentredString(letter[0] / 2, 0.4 * inch, f"Page {doc.page}")
    canvas.drawString(0.75 * inch, 0.4 * inch, "clarifying_question_dataset.jsonl")
    canvas.restoreState()


doc = SimpleDocTemplate(OUT, pagesize=letter, leftMargin=0.75 * inch, rightMargin=0.75 * inch,
                        topMargin=0.7 * inch, bottomMargin=0.7 * inch,
                        title="Clarifying Question Dataset", author="Promptimize")
stats = [["Total examples", str(len(rows))], ["Executable", str(n_exec)],
         ["Needs clarification", str(len(rows) - n_exec)]] + \
        [[f"   {t}", str(c)] for t, c in type_counts.most_common()]
stats_table = Table(stats, colWidths=[2.6 * inch, 1.0 * inch])
stats_table.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 9),
                                 ("TEXTCOLOR", (0, 3), (0, -1), GREY),
                                 ("FONTNAME", (0, 0), (0, 2), "Helvetica-Bold"),
                                 ("LINEBELOW", (0, 2), (-1, 2), 0.25, colors.HexColor("#cccccc")),
                                 ("TOPPADDING", (0, 0), (-1, -1), 1.5),
                                 ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5)]))
story = [Paragraph("Clarifying Question Dataset", st_title),
         Paragraph("Fine-tuning data for judging whether a developer prompt is executable "
                   "and asking the single best clarifying question when it isn't.", st_sub),
         Spacer(1, 8), stats_table, Spacer(1, 6),
         HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#bbbbbb")), Spacer(1, 12)]
story += [entry(i, r) for i, r in enumerate(rows, 1)]
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(f"wrote {OUT} ({len(rows)} examples)")
