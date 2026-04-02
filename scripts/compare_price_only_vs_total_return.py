import contextlib
import io

import pandas as pd

from scripts.am100_performance import (
    AM100_HISTORY_FILE,
    build_total_return_index,
    load_history,
    load_prices,
)
from scripts.load_dividends import load_dividends


def cagr(series, dates):
    years = (dates.iloc[-1] - dates.iloc[0]).days / 365
    return (series.iloc[-1] / series.iloc[0]) ** (1 / years) - 1


def max_dd(series):
    peak = series.cummax()
    return (series / peak - 1).min()


def main():
    prices = load_prices()
    history = load_history(AM100_HISTORY_FILE)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        price_only_index, _ = build_total_return_index(prices, history, {})
        total_return_index, _ = build_total_return_index(
            prices, history, load_dividends()
        )

    df = (
        pd.merge(
            price_only_index[["Date", "Index Level"]],
            total_return_index[["Date", "Index Level"]],
            on="Date",
            suffixes=("_price", "_total"),
        )
        .sort_values("Date")
        .reset_index(drop=True)
    )

    results = {
        "price_only_cagr": cagr(df["Index Level_price"], df["Date"]),
        "total_return_cagr": cagr(df["Index Level_total"], df["Date"]),
        "cagr_delta": cagr(df["Index Level_total"], df["Date"])
        - cagr(df["Index Level_price"], df["Date"]),
        "price_only_max_dd": max_dd(df["Index Level_price"]),
        "total_return_max_dd": max_dd(df["Index Level_total"]),
        "end_price_only": df["Index Level_price"].iloc[-1],
        "end_total_return": df["Index Level_total"].iloc[-1],
    }

    print(results)
    print("\n--- INTERPRETATION ---")

    if results["cagr_delta"] < 0.01:
        print("⚠️ Low dividend impact → likely missing coverage")
    elif results["cagr_delta"] > 0.08:
        print("⚠️ High dividend impact → check for unit inconsistencies")
    else:
        print("✅ Dividend impact within expected range")


if __name__ == "__main__":
    main()
