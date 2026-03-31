import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.load_dividends import load_dividends
from scripts.data_validation import DAILY_RETURN_CAP, cap_return_series
from scripts.performance_metrics import compute_metrics

MASTER_FILE = "output/master.xlsx"
AM100_HISTORY_FILE = "output/AM100_history.xlsx"
AM200_HISTORY_FILE = "output/AM200_history.xlsx"
AM100_OUTPUT_XLSX = "output/AM100_index.xlsx"
AM100_OUTPUT_CSV = "output/AM100_total_return.csv"
AM200_OUTPUT_CSV = "output/AM200_total_return.csv"
AM100_METRICS_FILE = "output/AM100_metrics.csv"
AM200_METRICS_FILE = "output/AM200_metrics.csv"
REPORT_FILE = "output/index_report.txt"
RETURN_VALIDATION_FILE = "output/return_validation_report.csv"

BASE_VALUE = 1000.0


def normalize_index(index_df):
    if len(index_df) == 0:
        return index_df

    index_df = index_df.copy()
    index_df["Index Level"] = index_df["Index Level"] / index_df["Index Level"].iloc[0] * BASE_VALUE
    return index_df


def build_total_return_index(prices, history, dividends):
    rebalance_dates = sorted(history["Date"].unique())
    index_levels = []
    current_index = BASE_VALUE
    validation_records = []

    for i in range(len(rebalance_dates) - 1):
        start_date = rebalance_dates[i]
        end_date = rebalance_dates[i + 1]

        weights_df = history[history["Date"] == start_date]
        period_prices = prices[
            (prices["Date"] >= start_date) & (prices["Date"] <= end_date)
        ].copy()

        if len(period_prices) == 0:
            continue

        segment_returns = []

        for _, row in weights_df.iterrows():
            company = row["Company"]
            weight = row["Weight"]

            if pd.isna(weight) or weight <= 0:
                continue

            price_col = f"{company} Price"
            if price_col not in period_prices.columns:
                continue

            series = period_prices[["Date", price_col]].dropna()
            if len(series) < 2:
                continue

            series = series.sort_values("Date").copy()
            series["Dividend"] = 0.0

            div_series = dividends.get(company)
            if div_series is not None:
                aligned_dividends = div_series.reindex(series["Date"]).fillna(0.0)
                series["Dividend"] = aligned_dividends.to_numpy()

            series["PrevPrice"] = series[price_col].shift(1)
            series["DailyReturn"] = (series[price_col] / series["PrevPrice"]) - 1

            dividend_mask = (
                series["Dividend"].ne(0)
                & series["PrevPrice"].notna()
                & series["PrevPrice"].ne(0)
            )
            series.loc[dividend_mask, "DailyReturn"] += (
                series.loc[dividend_mask, "Dividend"]
                / series.loc[dividend_mask, "PrevPrice"]
            )

            daily_returns = series.set_index("Date")["DailyReturn"].dropna()
            if len(daily_returns) == 0:
                continue

            capped_returns, outlier_count = cap_return_series(daily_returns, DAILY_RETURN_CAP)
            if outlier_count > 0:
                validation_records.append(
                    {
                        "RebalanceDate": start_date,
                        "Company": company,
                        "OutlierReturnCount": outlier_count,
                        "MaxAbsDailyReturn": float(daily_returns.abs().max()),
                        "ReturnCap": DAILY_RETURN_CAP,
                    }
                )

            segment_returns.append(capped_returns.mul(weight))

        if len(segment_returns) == 0:
            index_levels.append({"Date": end_date, "Index Level": current_index})
            continue

        period_returns = pd.concat(segment_returns, axis=1).fillna(0.0).sum(axis=1)
        new_series = (1 + period_returns).cumprod()
        stitched_series = current_index * (new_series / new_series.iloc[0])

        for date, level in stitched_series.items():
            index_levels.append({"Date": date, "Index Level": level})

        current_index = stitched_series.iloc[-1]

    index_df = pd.DataFrame(index_levels)
    validation_df = pd.DataFrame(validation_records)
    return normalize_index(index_df), validation_df


def load_prices():
    prices = pd.read_excel(MASTER_FILE)
    prices["Date"] = pd.to_datetime(prices["Date"], dayfirst=True)
    return prices.sort_values("Date")


def load_history(path):
    history = pd.read_excel(path)
    history["Date"] = pd.to_datetime(history["Date"])
    return history


def format_metric(value):
    return f"{value:.1%}"


def print_index_debug(label, index_series):
    print(f"\n=== {label} INDEX DEBUG ===")
    print(index_series.head())
    print(index_series.tail())
    print(index_series.describe())


def write_report(am100_index, am100_metrics, am200_index=None, am200_metrics=None):
    start_year = am100_index["Date"].min().year
    end_year = am100_index["Date"].max().year

    with open(REPORT_FILE, "w", encoding="utf-8") as report:
        report.write("AM100 vs AM200 Performance Report\n\n")
        report.write(f"Period: {start_year}–{end_year}\n\n")

        report.write("AM100:\n")
        report.write(f"CAGR: {format_metric(am100_metrics['CAGR'])}\n")
        report.write(f"Volatility: {format_metric(am100_metrics['Volatility'])}\n")
        report.write(f"Sharpe: {am100_metrics['Sharpe']:.2f}\n")
        report.write(f"Max Drawdown: {format_metric(am100_metrics['Max Drawdown'])}\n\n")

        if am200_metrics is not None and am200_index is not None and len(am200_index) > 0:
            report.write("AM200:\n")
            report.write(f"CAGR: {format_metric(am200_metrics['CAGR'])}\n")
            report.write(f"Volatility: {format_metric(am200_metrics['Volatility'])}\n")
            report.write(f"Sharpe: {am200_metrics['Sharpe']:.2f}\n")
            report.write(f"Max Drawdown: {format_metric(am200_metrics['Max Drawdown'])}\n\n")

            report.write("Observations:\n")
            if am200_metrics["CAGR"] > am100_metrics["CAGR"]:
                report.write("- AM200 delivers higher returns with higher volatility\n")
            else:
                report.write("- AM100 delivers higher returns with lower volatility\n")
            report.write("- AM100 provides stability and concentration\n")
            report.write("- Frontier exposure enhances growth but increases drawdown\n")
        else:
            report.write("AM200:\n")
            report.write("Not available yet\n\n")
            report.write("Observations:\n")
            report.write("- AM100 total return metrics are available\n")
            report.write("- AM200 report will populate once AM200 history is built\n")


def main():
    prices = load_prices()
    dividends = load_dividends()

    am100_history = load_history(AM100_HISTORY_FILE)
    am100_index, am100_validation = build_total_return_index(prices, am100_history, dividends)

    if len(am100_index) > 0:
        am100_index.to_excel(AM100_OUTPUT_XLSX, index=False)
        am100_index.to_csv(AM100_OUTPUT_CSV, index=False)
        am100_tr = am100_index.set_index("Date")["Index Level"]
        print_index_debug("AM100", am100_tr)
        am100_metrics = compute_metrics(am100_tr)
        pd.DataFrame([am100_metrics]).to_csv(AM100_METRICS_FILE, index=False)
        am100_validation.to_csv(RETURN_VALIDATION_FILE, index=False)
        print(f"✅ AM100 total return created → {AM100_OUTPUT_CSV}")
    else:
        print("⚠️ No AM100 total return data generated")

    if os.path.exists(AM200_HISTORY_FILE):
        am200_history = load_history(AM200_HISTORY_FILE)
        am200_index, am200_validation = build_total_return_index(prices, am200_history, dividends)

        if len(am200_index) > 0:
            am200_index.to_csv(AM200_OUTPUT_CSV, index=False)
            am200_tr = am200_index.set_index("Date")["Index Level"]
            print_index_debug("AM200", am200_tr)
            am200_metrics = compute_metrics(am200_tr)
            pd.DataFrame([am200_metrics]).to_csv(AM200_METRICS_FILE, index=False)
            if len(am200_validation) > 0:
                am200_validation["Index"] = "AM200"
                if len(am100_validation) > 0:
                    am100_validation["Index"] = "AM100"
                    pd.concat([am100_validation, am200_validation], ignore_index=True).to_csv(
                        RETURN_VALIDATION_FILE, index=False
                    )
                else:
                    am200_validation.to_csv(RETURN_VALIDATION_FILE, index=False)
            print(f"✅ AM200 total return created → {AM200_OUTPUT_CSV}")
        else:
            print("⚠️ No AM200 total return data generated")
    else:
        print("⚠️ AM200 history not found; skipped AM200 total return export")

    if len(am100_index) > 0:
        write_report(
            am100_index,
            am100_metrics,
            am200_index if "am200_index" in locals() else None,
            am200_metrics if "am200_metrics" in locals() else None,
        )
        print(f"✅ Performance report created → {REPORT_FILE}")


if __name__ == "__main__":
    main()
