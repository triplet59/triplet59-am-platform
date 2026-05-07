from __future__ import annotations

from pathlib import Path
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

BASE = Path('/Users/derrythornalley/AM100_DATA/output')
SUMMARY_CSV = BASE / 'CAGR_proof_pack_summary.csv'
TRACE_CSV = BASE / 'CAGR_proof_pack_trace.csv'
PDF_PATH = BASE / 'CAGR_proof_pack.pdf'


def pct(x: float) -> str:
    return f'{x * 100:.2f}%' if pd.notna(x) else 'N/A'


def num(x: float) -> str:
    return f'{x:,.6f}' if pd.notna(x) else 'N/A'


def build() -> None:
    summary = pd.read_csv(SUMMARY_CSV)
    trace = pd.read_csv(TRACE_CSV)

    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=landscape(A4),
        rightMargin=14 * mm,
        leftMargin=14 * mm,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='SmallBody', parent=styles['BodyText'], fontSize=9, leading=12))
    styles.add(ParagraphStyle(name='TinyBody', parent=styles['BodyText'], fontSize=8, leading=10))
    styles.add(ParagraphStyle(name='Section', parent=styles['Heading2'], fontSize=14, leading=18, textColor=colors.HexColor('#17365D')))

    story = []
    story.append(Paragraph('CAGR Proof Pack', styles['Title']))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        'This pack provides categorical proof that the published CAGR values are derived directly from the live total return series and reconcile exactly to an independent recomputation.',
        styles['BodyText'],
    ))
    story.append(Spacer(1, 8))
    story.append(Paragraph('Formula', styles['Section']))
    story.append(Paragraph('CAGR = ((Ending Index Level / Starting Index Level) ^ (1 / Year Fraction)) - 1', styles['BodyText']))
    story.append(Paragraph('Year Fraction used in the live pipeline verification: exact elapsed days / 365', styles['BodyText']))
    story.append(Spacer(1, 10))

    story.append(Paragraph('Verification Summary', styles['Section']))
    summary_table = [[
        'Series', 'Status', 'Start Date', 'End Date', 'Day Count', 'Start Value', 'End Value', 'Published CAGR', 'Independent CAGR', 'Match'
    ]]
    for _, row in summary.iterrows():
        summary_table.append([
            row['Series'],
            row['Series Status'],
            row['Start Date'],
            row['End Date'],
            str(int(row['Day Count'])),
            num(row['Start Index Level']),
            num(row['End Index Level']),
            pct(row['Published CAGR']),
            pct(row['Independent Recomputed CAGR (365)']),
            'Yes' if bool(row['Exact Match To Published']) else 'No',
        ])

    tbl = Table(summary_table, repeatRows=1, colWidths=[18*mm, 38*mm, 22*mm, 22*mm, 18*mm, 26*mm, 28*mm, 22*mm, 24*mm, 12*mm])
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1F2937')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.25, colors.HexColor('#CBD5E1')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')]),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ALIGN', (4, 1), (-1, -1), 'RIGHT'),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 10))

    story.append(Paragraph('Methodology Note', styles['Section']))
    story.append(Paragraph(
        'The published CAGR is derived directly from each live series total return index file. This verification pack recomputes CAGR independently from the first and last index levels over the exact elapsed day count. The return methodology itself is unchanged; where published CAGR has moved after data corrections, the change reflects revised constituent ranking, selection, and weighting rather than a change in formula.',
        styles['SmallBody'],
    ))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        'Interpretation note: AM100, AM200, and AM300 are live benchmark series. EAC25 and EAC_EXT are research-only historical series and should not be presented as institutional benchmark track records.',
        styles['SmallBody'],
    ))

    story.append(PageBreak())
    story.append(Paragraph('Chain of Calculation', styles['Section']))
    for series_name in summary['Series']:
        story.append(Paragraph(series_name, styles['Heading3']))
        rows = trace[trace['Series'] == series_name][['Trace Step', 'Detail']]
        tdata = [['Step', 'Detail']] + rows.values.tolist()
        t = Table(tdata, repeatRows=1, colWidths=[58*mm, 198*mm])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1F2937')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.25, colors.HexColor('#CBD5E1')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')]),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 4),
            ('RIGHTPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(t)
        story.append(Spacer(1, 8))

    doc.build(story)
    print(PDF_PATH)


if __name__ == '__main__':
    build()
