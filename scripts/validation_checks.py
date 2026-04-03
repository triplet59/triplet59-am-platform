import os
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from data_loader import INDEX_POOL, load_index


def calculate_drawdown(series):
    peak = series.cummax()
    return (series / peak) - 1


def check_dividend_sanity():
    outlier_path = "output/dividend_outliers.csv"
    if not os.path.exists(outlier_path) or os.path.getsize(outlier_path) == 0:
        print("Dividend sanity: OK (no flagged outliers file content)")
        return

    try:
        outliers = pd.read_csv(outlier_path)
    except pd.errors.EmptyDataError:
        print("Dividend sanity: OK (empty outlier file)")
        return

    if outliers.empty:
        print("Dividend sanity: OK (0 flagged outliers)")
        return

    raise AssertionError(f"Dividend sanity failed: {len(outliers)} outlier rows remain")


def check_return_sanity(name):
    df = load_index(name)
    returns = df["Index Level"].pct_change().dropna()
    max_abs = returns.abs().max()
    print(f"{name} max absolute daily return: {max_abs:.2%}")
    assert max_abs < 0.5, f"{name} return sanity failed: max daily move {max_abs:.2%}"
    return df, returns


def check_metric_ranges(name, df, returns):
    index_series = df["Index Level"]
    years = (df["Date"].iloc[-1] - df["Date"].iloc[0]).days / 365
    cagr = (index_series.iloc[-1] / index_series.iloc[0]) ** (1 / years) - 1
    volatility = returns.std() * np.sqrt(252)
    mean_return = returns.mean() * 252
    rf = 0.02
    sharpe = (mean_return - rf) / volatility if volatility != 0 else np.nan
    drawdown = calculate_drawdown(index_series).min()

    print(
        f"{name} metrics -> CAGR: {cagr:.2%}, Vol: {volatility:.2%}, "
        f"Sharpe: {sharpe:.2f}, Max DD: {drawdown:.2%}"
    )

    assert sharpe < 3, f"{name} Sharpe out of range: {sharpe:.2f}"
    assert cagr < 0.5, f"{name} CAGR out of range: {cagr:.2%}"
    assert drawdown > -0.8, f"{name} drawdown out of range: {drawdown:.2%}"


def check_weight_sum(path, label):
    df = pd.read_excel(path)
    sums = df.groupby("Date")["Weight"].sum()
    min_sum = sums.min()
    max_sum = sums.max()
    print(f"{label} weight sums -> min: {min_sum:.8f}, max: {max_sum:.8f}")
    assert np.allclose(sums.values, 1.0), f"{label} weights do not sum to 1"


def check_capacity_sanity():
    path = "output/AM300_capacity.csv"
    capacity = pd.read_csv(path, parse_dates=["Date"])
    required_cols = {"ADV_USD", "InvestableCapacityUSD"}
    missing = required_cols - set(capacity.columns)
    assert not missing, f"AM300 capacity missing columns: {sorted(missing)}"
    valid = capacity.dropna(subset=["ADV_USD", "InvestableCapacityUSD"]).sort_values("Date")
    assert not valid.empty, "AM300 capacity has no valid USD rows"
    latest = valid.iloc[-1]
    assert pd.notna(latest["ADV_USD"]), "Latest AM300 ADV_USD is NaN"
    assert pd.notna(latest["InvestableCapacityUSD"]), "Latest AM300 InvestableCapacityUSD is NaN"
    assert latest["ADV_USD"] > 0, "Latest AM300 ADV_USD must be positive"
    assert latest["InvestableCapacityUSD"] > 0, "Latest AM300 InvestableCapacityUSD must be positive"
    print(
        f"AM300 ADV_USD -> latest: {latest['ADV_USD']:.2f}, "
        f"InvestableCapacityUSD latest: {latest['InvestableCapacityUSD']:.2f}"
    )


def main():
    print("Running validation checks...")
    check_dividend_sanity()

    for name in INDEX_POOL:
        df, returns = check_return_sanity(name)
        check_metric_ranges(name, df, returns)

    check_weight_sum("output/AM100_history.xlsx", "AM100")
    check_weight_sum("output/AM200_history.xlsx", "AM200")
    check_weight_sum("output/AM300_history.xlsx", "AM300")
    check_capacity_sanity()

    print("All validation checks passed.")


if __name__ == "__main__":
    main()
