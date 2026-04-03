import os

import matplotlib.pyplot as plt
import numpy as np
from scripts.data_loader import load_index
from scripts.metrics import calculate_drawdown, calculate_sharpe


def compute_metrics(series):
    returns = series.pct_change().dropna()

    cagr = (series.iloc[-1] / series.iloc[0]) ** (252 / len(returns)) - 1
    vol = returns.std() * np.sqrt(252)
    sharpe = calculate_sharpe(series)

    drawdown = calculate_drawdown(series)
    max_dd = drawdown.min()

    return cagr, vol, sharpe, max_dd

def main():
    am100 = load_index("AM100").set_index("Date")["Index Level"]

    if not os.path.exists("output/AM200_total_return.csv"):
        raise FileNotFoundError(
            "output/AM200_total_return.csv is missing. Build AM200 total return first."
        )

    am200 = load_index("AM200").set_index("Date")["Index Level"]

    cagr100, vol100, sharpe100, dd100 = compute_metrics(am100)
    cagr200, vol200, sharpe200, dd200 = compute_metrics(am200)

    print("\n=== PERFORMANCE SUMMARY ===")
    print("AM100")
    print(
        f"CAGR: {cagr100:.2%}, Vol: {vol100:.2%}, Sharpe: {sharpe100:.2f}, MaxDD: {dd100:.2%}"
    )

    print("\nAM200")
    print(
        f"CAGR: {cagr200:.2%}, Vol: {vol200:.2%}, Sharpe: {sharpe200:.2f}, MaxDD: {dd200:.2%}"
    )

    # Plot 1: Index Levels
    plt.figure()
    plt.plot(am100, label="AM100")
    plt.plot(am200, label="AM200")
    plt.title("AM100 vs AM200 Index Performance")
    plt.xlabel("Date")
    plt.ylabel("Index Level")
    plt.legend()
    plt.grid()

    # Plot 2: Daily Returns Distribution
    returns100 = am100.pct_change().dropna()
    returns200 = am200.pct_change().dropna()

    plt.figure()
    plt.hist(returns100, bins=50, alpha=0.5, label="AM100")
    plt.hist(returns200, bins=50, alpha=0.5, label="AM200")
    plt.legend()
    plt.title("Return Distribution")

    # Plot 3: Drawdown
    dd100_series = am100 / am100.cummax() - 1
    dd200_series = am200 / am200.cummax() - 1

    plt.figure()
    plt.plot(dd100_series, label="AM100")
    plt.plot(dd200_series, label="AM200")
    plt.legend()
    plt.title("Drawdown Comparison")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.grid()
    plt.show()

    rolling_vol100 = am100.pct_change().rolling(30).std() * np.sqrt(252)
    rolling_vol200 = am200.pct_change().rolling(30).std() * np.sqrt(252)

    plt.figure()
    plt.plot(rolling_vol100, label="AM100")
    plt.plot(rolling_vol200, label="AM200")
    plt.title("Rolling Volatility (30D)")
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure()
    plt.plot(np.log(am100), label="AM100")
    plt.plot(np.log(am200), label="AM200")
    plt.title("Log Performance (Relative Growth)")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
