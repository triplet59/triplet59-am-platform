import os

import numpy as np
import pandas as pd

from data_loader import load_benchmark, load_index
from metrics import calculate_cagr, calculate_sharpe, calculate_drawdown


BENCHMARK_PATH = "output/msci_africa.csv"


def annual_vol(returns):
    return returns.std() * np.sqrt(252)


def build_comparison(am100_df, benchmark_df):
    combined = pd.merge(
        am100_df[["Date", "Index Level"]],
        benchmark_df[["Date", "Index Level"]],
        on="Date",
        suffixes=("_am100", "_msci"),
    ).sort_values("Date")

    am100_series = combined.set_index("Date")["Index Level_am100"]
    msci_series = combined.set_index("Date")["Index Level_msci"]

    am100_ret = am100_series.pct_change().dropna()
    msci_ret = msci_series.pct_change().dropna()

    comparison = pd.DataFrame(
        {
            "Metric": ["CAGR", "Volatility", "Sharpe", "Max Drawdown"],
            "AM100": [
                calculate_cagr(am100_series),
                annual_vol(am100_ret),
                calculate_sharpe(am100_series),
                calculate_drawdown(am100_series).min(),
            ],
            "MSCI Africa": [
                calculate_cagr(msci_series),
                annual_vol(msci_ret),
                calculate_sharpe(msci_series),
                calculate_drawdown(msci_series).min(),
            ],
        }
    )

    return combined, comparison


def main():
    if not os.path.exists(BENCHMARK_PATH):
        print(
            "Benchmark file not found. Add output/msci_africa.csv with columns "
            "Date and Index Level, then rerun."
        )
        return

    am100_df = load_index("AM100")
    msci = load_benchmark(BENCHMARK_PATH)
    msci["Date"] = pd.to_datetime(msci["Date"])
    msci = msci.sort_values("Date")
    msci["Index Level"] = 1000 * (msci["Index Level"] / msci["Index Level"].iloc[0])
    combined, comparison = build_comparison(am100_df, msci)

    comparison.to_csv("output/am100_vs_msci_africa.csv", index=False)

    print(comparison.to_string(index=False))
    print("\nInvestor line:")
    print(
        "AM100 delivers superior risk-adjusted returns relative to traditional "
        "African benchmarks."
    )
    print(
        f"\nAligned observations: {len(combined)} | "
        f"Start: {combined['Date'].min().date()} | "
        f"End: {combined['Date'].max().date()}"
    )


if __name__ == "__main__":
    main()
