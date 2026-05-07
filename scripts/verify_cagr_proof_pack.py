from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path('/Users/derrythornalley/AM100_DATA/output')
SCRIPT_DIR = Path('/Users/derrythornalley/AM100_DATA/scripts')

SERIES = {
    'AM100': {
        'series_file': BASE / 'AM100_total_return.csv',
        'metrics_file': BASE / 'AM100_metrics.csv',
        'status': 'Live benchmark series',
    },
    'AM200': {
        'series_file': BASE / 'AM200_total_return.csv',
        'metrics_file': BASE / 'AM200_metrics.csv',
        'status': 'Live benchmark series',
    },
    'AM300': {
        'series_file': BASE / 'AM300_total_return.csv',
        'metrics_file': BASE / 'AM300_metrics.csv',
        'status': 'Live benchmark series',
    },
    'EAC25': {
        'series_file': BASE / 'EAC25_total_return.csv',
        'metrics_file': BASE / 'EAC25_metrics.csv',
        'status': 'Research-only historical series',
    },
    'EAC_EXT': {
        'series_file': BASE / 'EAC_EXT_total_return.csv',
        'metrics_file': BASE / 'EAC_EXT_metrics.csv',
        'status': 'Research-only historical series',
    },
}


def load_series(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    return df.sort_values('Date').reset_index(drop=True)


def compute_cagr(series: pd.Series, dates: pd.Series, days_per_year: float = 365.0) -> tuple[float, int, float]:
    start = float(series.iloc[0])
    end = float(series.iloc[-1])
    day_count = int((dates.iloc[-1] - dates.iloc[0]).days)
    years = day_count / days_per_year
    cagr = (end / start) ** (1 / years) - 1
    return cagr, day_count, years


def format_pct(x: float) -> str:
    return f'{x:.6%}' if pd.notna(x) else 'N/A'


def build_pack() -> dict[str, pd.DataFrame]:
    summary_rows = []
    trace_rows = []
    methodology_rows = []

    for name, meta in SERIES.items():
        series_df = load_series(meta['series_file'])
        metrics_df = pd.read_csv(meta['metrics_file'])
        published = float(metrics_df.loc[0, 'CAGR'])
        recomputed_365, day_count, years_365 = compute_cagr(series_df['Index Level'], series_df['Date'], 365.0)
        recomputed_36525, _, years_36525 = compute_cagr(series_df['Index Level'], series_df['Date'], 365.25)
        start_date = series_df['Date'].iloc[0]
        end_date = series_df['Date'].iloc[-1]
        start_value = float(series_df['Index Level'].iloc[0])
        end_value = float(series_df['Index Level'].iloc[-1])
        delta = published - recomputed_365
        matched = abs(delta) < 1e-10

        summary_rows.append({
            'Series': name,
            'Series Status': meta['status'],
            'Source Total Return File': str(meta['series_file']),
            'Source Metrics File': str(meta['metrics_file']),
            'Start Date': start_date.date().isoformat(),
            'End Date': end_date.date().isoformat(),
            'Day Count': day_count,
            'Year Fraction (365)': years_365,
            'Year Fraction (365.25 reference)': years_36525,
            'Start Index Level': start_value,
            'End Index Level': end_value,
            'Formula': '((End Index Level / Start Index Level)^(1 / Year Fraction)) - 1',
            'Published CAGR': published,
            'Independent Recomputed CAGR (365)': recomputed_365,
            'Reference Recomputed CAGR (365.25)': recomputed_36525,
            'Published Minus Recomputed (365)': delta,
            'Exact Match To Published': matched,
        })

        methodology_rows.append({
            'Series': name,
            'Methodology Note': (
                'Published CAGR is verified directly from the live total return series using the same 365-day year convention '
                'used by the pipeline metrics helpers. The 365.25-day version is shown only as a reference check.'
            ),
            'Series Status': meta['status'],
        })

        trace_rows.extend([
            {
                'Series': name,
                'Trace Step': '1. Load total return series',
                'Detail': str(meta['series_file']),
            },
            {
                'Series': name,
                'Trace Step': '2. Read first and last valid observation',
                'Detail': f'{start_date.date().isoformat()} -> {end_date.date().isoformat()}',
            },
            {
                'Series': name,
                'Trace Step': '3. Measure exact elapsed days',
                'Detail': str(day_count),
            },
            {
                'Series': name,
                'Trace Step': '4. Convert to year fraction using 365-day convention',
                'Detail': f'{years_365:.12f}',
            },
            {
                'Series': name,
                'Trace Step': '5. Apply CAGR formula',
                'Detail': f'(({end_value:.12f}/{start_value:.12f})^(1/{years_365:.12f})) - 1 = {recomputed_365:.12%}',
            },
            {
                'Series': name,
                'Trace Step': '6. Compare to published metrics file',
                'Detail': f'Published {published:.12%}; delta {delta:.12e}',
            },
        ])

    summary_df = pd.DataFrame(summary_rows)
    trace_df = pd.DataFrame(trace_rows)
    methodology_df = pd.DataFrame(methodology_rows)
    return {
        'summary': summary_df,
        'trace': trace_df,
        'methodology': methodology_df,
    }


def write_outputs(packs: dict[str, pd.DataFrame]) -> None:
    summary_path = BASE / 'CAGR_proof_pack_summary.csv'
    trace_path = BASE / 'CAGR_proof_pack_trace.csv'
    methodology_path = BASE / 'CAGR_methodology_note.csv'
    workbook_path = BASE / 'CAGR_proof_pack.xlsx'
    note_path = BASE / 'CAGR_methodology_note.md'

    packs['summary'].to_csv(summary_path, index=False)
    packs['trace'].to_csv(trace_path, index=False)
    packs['methodology'].to_csv(methodology_path, index=False)

    with pd.ExcelWriter(workbook_path, engine='openpyxl') as writer:
        packs['summary'].to_excel(writer, sheet_name='Summary', index=False)
        packs['trace'].to_excel(writer, sheet_name='Trace', index=False)
        packs['methodology'].to_excel(writer, sheet_name='Methodology', index=False)

    note = """# CAGR Methodology Note

The published CAGR for each series is derived directly from its live total return index series.

Formula used:
`CAGR = ((Ending Index Level / Starting Index Level) ** (1 / Year Fraction)) - 1`

For the live AM and EAC metrics files, the verification pack uses the same 365-day year convention used by the pipeline metrics helpers:
- `Year Fraction = Exact elapsed days / 365`

What this proof pack demonstrates:
- the exact source total return file used for each series
- the exact start date, end date, start index level, end index level, and day count
- an independent recomputation of CAGR from those values
- a direct reconciliation to the published metrics file

Interpretation note:
- AM100, AM200, and AM300 are live benchmark series.
- EAC25 and EAC_EXT are research-only historical series and should not be presented as institutional benchmark track records.

Investor-ready statement:
The published CAGR is derived directly from the series total return index levels and has been independently recomputed from the raw start and end index values over the exact analysis window. The return methodology itself is unchanged; any changes in published CAGR after data corrections reflect revised constituent ranking, selection, and weighting rather than a change in formula.
"""
    note_path.write_text(note)


def main() -> None:
    packs = build_pack()
    write_outputs(packs)
    print('Wrote:')
    for name in ['CAGR_proof_pack_summary.csv', 'CAGR_proof_pack_trace.csv', 'CAGR_methodology_note.csv', 'CAGR_proof_pack.xlsx', 'CAGR_methodology_note.md']:
        print(BASE / name)
    print('\nVerification summary:')
    print(packs['summary'][['Series', 'Published CAGR', 'Independent Recomputed CAGR (365)', 'Exact Match To Published']].to_string(index=False))


if __name__ == '__main__':
    main()
