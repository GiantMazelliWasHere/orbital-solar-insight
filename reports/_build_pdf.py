"""Gera reports/relatorio.pdf a partir de reports/relatorio.md usando ReportLab.

Uso (raiz do repo)::

    python reports/_build_pdf.py
"""
from __future__ import annotations

import re
from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

REPORTS_DIR = Path(__file__).resolve().parent
MD_PATH = REPORTS_DIR / "relatorio.md"
PDF_PATH = REPORTS_DIR / "relatorio.pdf"

PRIMARY = HexColor("#1f4d8a")
GREY = HexColor("#666666")
LIGHT = HexColor("#e7eef7")
BORDER = HexColor("#aab2bd")


def make_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    styles = {
        "h1": ParagraphStyle(
            "h1", parent=base["Heading1"],
            fontName="Helvetica-Bold", fontSize=18, textColor=PRIMARY,
            spaceBefore=14, spaceAfter=8,
        ),
        "h2": ParagraphStyle(
            "h2", parent=base["Heading2"],
            fontName="Helvetica-Bold", fontSize=14, textColor=PRIMARY,
            spaceBefore=12, spaceAfter=6,
        ),
        "h3": ParagraphStyle(
            "h3", parent=base["Heading3"],
            fontName="Helvetica-Bold", fontSize=11.5, textColor=HexColor("#333333"),
            spaceBefore=8, spaceAfter=4,
        ),
        "h4": ParagraphStyle(
            "h4", parent=base["Heading4"],
            fontName="Helvetica-Bold", fontSize=10.5, textColor=HexColor("#444444"),
            spaceBefore=6, spaceAfter=3,
        ),
        "body": ParagraphStyle(
            "body", parent=base["BodyText"],
            fontName="Helvetica", fontSize=9.5, leading=13,
            alignment=TA_JUSTIFY, spaceAfter=4,
        ),
        "bullet": ParagraphStyle(
            "bullet", parent=base["BodyText"],
            fontName="Helvetica", fontSize=9.5, leading=13,
            leftIndent=18, bulletIndent=6, spaceAfter=2,
        ),
        "code": ParagraphStyle(
            "code", parent=base["BodyText"],
            fontName="Courier", fontSize=8.5, leading=11,
            backColor=HexColor("#f4f6f8"),
            leftIndent=10, rightIndent=10, spaceBefore=2, spaceAfter=6,
        ),
        "quote": ParagraphStyle(
            "quote", parent=base["BodyText"],
            fontName="Helvetica-Oblique", fontSize=9.5, leading=13,
            leftIndent=14, rightIndent=14, textColor=GREY,
            backColor=HexColor("#f4f6f8"), spaceBefore=4, spaceAfter=6,
        ),
    }
    return styles


def md_inline_to_rl(text: str) -> str:
    """Converte negrito/itálico/inline-code de Markdown para tags do ReportLab."""
    # escape básico de tags
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"`([^`]+)`", r"<font face='Courier' size='8.5'>\1</font>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", text)
    # links markdown [t](url) -> simplesmente t
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"<u>\1</u>", text)
    return text


def parse_markdown(md: str, styles: dict[str, ParagraphStyle]) -> list:
    """Tradução simples de Markdown para uma sequência de Flowables."""
    flow: list = []
    lines = md.splitlines()
    i = 0

    def flush_table(rows: list[list[str]]) -> None:
        if not rows:
            return
        rl_rows: list[list[Paragraph]] = []
        header_style = ParagraphStyle(
            "th", parent=styles["body"],
            fontName="Helvetica-Bold", fontSize=9, leading=11,
        )
        cell_style = ParagraphStyle(
            "td", parent=styles["body"], fontSize=9, leading=11,
            spaceAfter=0, alignment=TA_LEFT,
        )
        for r, row in enumerate(rows):
            rl_row = [
                Paragraph(md_inline_to_rl(cell.strip()),
                          header_style if r == 0 else cell_style)
                for cell in row
            ]
            rl_rows.append(rl_row)
        n_cols = max(len(r) for r in rl_rows)
        # iguala n colunas
        rl_rows = [r + [Paragraph("", cell_style)] * (n_cols - len(r)) for r in rl_rows]
        avail = A4[0] - 4 * cm
        col_width = avail / n_cols
        table = Table(rl_rows, colWidths=[col_width] * n_cols, repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), LIGHT),
            ("GRID", (0, 0), (-1, -1), 0.4, BORDER),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        flow.append(table)
        flow.append(Spacer(1, 6))

    while i < len(lines):
        line = lines[i]

        # divisor
        if line.strip() in ("---", "***"):
            flow.append(Spacer(1, 6))
            i += 1
            continue

        # cabeçalhos
        m = re.match(r"^(#{1,4})\s+(.*)", line)
        if m:
            level = len(m.group(1))
            txt = md_inline_to_rl(m.group(2).strip())
            style_key = {1: "h1", 2: "h2", 3: "h3", 4: "h4"}[level]
            flow.append(Paragraph(txt, styles[style_key]))
            i += 1
            continue

        # citação
        if line.startswith("> "):
            txt = md_inline_to_rl(line[2:].strip())
            flow.append(Paragraph(txt, styles["quote"]))
            i += 1
            continue

        # código em bloco
        if line.strip().startswith("```"):
            buf: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1  # fecha o bloco
            for code_line in buf:
                flow.append(Paragraph(
                    code_line.replace(" ", "&nbsp;").replace("<", "&lt;").replace(">", "&gt;"),
                    styles["code"],
                ))
            flow.append(Spacer(1, 4))
            continue

        # tabela
        if "|" in line and i + 1 < len(lines) and re.match(r"^\s*\|?[\s\-:|]+\|?\s*$", lines[i + 1]):
            rows: list[list[str]] = []
            header = [c.strip() for c in line.strip().strip("|").split("|")]
            rows.append(header)
            i += 2  # pula separador
            while i < len(lines) and "|" in lines[i] and lines[i].strip():
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            flush_table(rows)
            continue

        # lista
        if re.match(r"^\s*[-*]\s+", line) or re.match(r"^\s*\d+\.\s+", line):
            content = re.sub(r"^\s*([-*]|\d+\.)\s+", "", line)
            flow.append(Paragraph(
                md_inline_to_rl(content),
                styles["bullet"],
                bulletText="•",
            ))
            i += 1
            continue

        # linha em branco
        if not line.strip():
            flow.append(Spacer(1, 4))
            i += 1
            continue

        # parágrafo regular — agrupa linhas consecutivas
        buf = [line]
        i += 1
        while i < len(lines) and lines[i].strip() and not re.match(
            r"^(#{1,4}\s|>|\s*[-*]\s|\s*\d+\.\s|```|\|)", lines[i]
        ):
            buf.append(lines[i])
            i += 1
        para = " ".join(s.strip() for s in buf)
        flow.append(Paragraph(md_inline_to_rl(para), styles["body"]))

    return flow


def main() -> None:
    md_text = MD_PATH.read_text(encoding="utf-8")
    styles = make_styles()
    flow = parse_markdown(md_text, styles)

    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=1.8 * cm, bottomMargin=1.8 * cm,
        title="Orbital Solar Insight — Relatório",
        author="FIAP — Global Solution 2026",
    )
    doc.build(flow)
    print(f"PDF gerado em {PDF_PATH}")


if __name__ == "__main__":
    main()
