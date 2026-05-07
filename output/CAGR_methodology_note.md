# CAGR Methodology Note

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
