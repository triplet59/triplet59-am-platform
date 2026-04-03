import contextlib
import io
import os
import sys

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from data_loader import load_index
from am100_performance import (
    AM100_HISTORY_FILE,
    build_total_return_index,
    load_history,
    load_prices,
)
from load_dividends import load_dividends


def cagr(series, dates):
    years = (dates.iloc[-1] - dates.iloc[0]).days / 365
    return (series.iloc[-1] / series.iloc[0]) ** (1 / years) - 1


def max_dd(series):
    peak = series.cummax()
    return (series / peak - 1).min()


def main():
    prices = load_prices()
    am100_history = load_history(AM100_HISTORY_FILE)
    am300_history = load_history("output/AM300_history.xlsx")

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        price_only_index, _ = build_total_return_index(prices, am100_history, {})
        total_return_index, _ = build_total_return_index(
            prices, am100_history, load_dividends()
        )
        am300_price_only_index, _ = build_total_return_index(prices, am300_history, {})
        am300_total_return_index, _ = build_total_return_index(
            prices, am300_history, load_dividends()
        )

    os.makedirs("output", exist_ok=True)
    price_only_index.to_csv("output/AM100_PRICE_ONLY_total_return.csv", index=False)
    total_return_index.to_csv("output/AM100_TOTAL_RETURN_AB_total_return.csv", index=False)
    am300_price_only_index.to_csv("output/AM300_PRICE_ONLY_total_return.csv", index=False)
    am300_total_return_index.to_csv("output/AM300_TOTAL_RETURN_AB_total_return.csv", index=False)

    df = (
        pd.merge(
            price_only_index[["Date", "Index Level"]],
            load_index("AM100")[["Date", "Index Level"]],
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

    print("\nSaved:")
    print("output/AM100_PRICE_ONLY_total_return.csv")
    print("output/AM100_TOTAL_RETURN_AB_total_return.csv")
    print("output/AM300_PRICE_ONLY_total_return.csv")
    print("output/AM300_TOTAL_RETURN_AB_total_return.csv")


if __name__ == "__main__":
    main()
