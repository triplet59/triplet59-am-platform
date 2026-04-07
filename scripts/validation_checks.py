import os
import sys
from pathlib import Path
import contextlib
import io

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
PROJECT_ROOT = Path(SCRIPT_DIR).parent

from am100_performance import (
    AM100_HISTORY_FILE,
    build_total_return_index,
    load_history,
)
from data_loader import INDEX_POOL, load_index
from load_dividends import load_dividends
from metrics import calculate_cagr, calculate_drawdown, calculate_sharpe

OUTPUT_DIR = PROJECT_ROOT / "output"
RETURN_SANITY_FILE = OUTPUT_DIR / "validation_return_sanity.csv"
PRICE_SPIKE_FILE = OUTPUT_DIR / "validation_price_spikes.csv"
DIVIDEND_IMPACT_FILE = OUTPUT_DIR / "validation_dividend_impacts.csv"
REBALANCE_JUMP_FILE = OUTPUT_DIR / "validation_rebalance_jumps.csv"
COVERAGE_FILE = OUTPUT_DIR / "validation_coverage_gaps.csv"
COUNTRY_CONCENTRATION_FILE = OUTPUT_DIR / "validation_country_concentration.csv"
CONTROL_TEST_FILE = OUTPUT_DIR / "validation_control_test.csv"
SUMMARY_FILE = OUTPUT_DIR / "validation_summary.csv"
REPORT_FILE = OUTPUT_DIR / "VALIDATION_REPORT.md"

RETURN_OUTLIER_THRESHOLD = 0.20
PRICE_SPIKE_THRESHOLD = 0.30
DIVIDEND_IMPACT_THRESHOLD = 0.10
REBALANCE_JUMP_THRESHOLD = 0.10


def load_price_panel():
    master_csv = PROJECT_ROOT / "output" / "master.csv"
    master_xlsx = PROJECT_ROOT / "output" / "master.xlsx"
    if master_csv.exists():
        prices = pd.read_csv(master_csv)
    elif master_xlsx.exists():
        prices = pd.read_excel(master_xlsx)
    else:
        raise FileNotFoundError("Neither output/master.csv nor output/master.xlsx exists")
    prices["Date"] = pd.to_datetime(prices["Date"], dayfirst=True, errors="coerce")
    return prices.sort_values("Date").reset_index(drop=True)


def annual_vol(returns):
    return returns.std() * np.sqrt(252)


def safe_float(value):
    return float(value) if pd.notna(value) else np.nan


def price_columns(prices):
    return [col for col in prices.columns if col.endswith(" Price")]


def company_from_price_col(col):
    return col[:-6]


def return_sanity_check(name):
    df = load_index(name)
    index_series = df.set_index("Date")["Index Level"].sort_index()
    returns = index_series.pct_change().dropna()
    outliers = returns[(returns > RETURN_OUTLIER_THRESHOLD) | (returns < -RETURN_OUTLIER_THRESHOLD)]
    rolling_vol = returns.rolling(252).std() * np.sqrt(252)

    summary = {
        "Index": name,
        "StartDate": index_series.index.min(),
        "EndDate": index_series.index.max(),
        "AnnualReturn": safe_float(returns.mean() * 252),
        "Volatility": safe_float(annual_vol(returns)),
        "RollingVolMin": safe_float(rolling_vol.min()),
        "RollingVolMedian": safe_float(rolling_vol.median()),
        "RollingVolMax": safe_float(rolling_vol.max()),
        "Sharpe": safe_float(calculate_sharpe(index_series)),
        "CAGR": safe_float(calculate_cagr(index_series)),
        "MaxDrawdown": safe_float(calculate_drawdown(index_series).min()),
        "MaxDailyReturn": safe_float(returns.max()),
        "MinDailyReturn": safe_float(returns.min()),
        "ExtremeReturnCount": int(len(outliers)),
    }

    print(
        f"{name} -> Max daily return: {summary['MaxDailyReturn']:.2%}, "
        f"Min daily return: {summary['MinDailyReturn']:.2%}, "
        f"Extreme return count: {summary['ExtremeReturnCount']}"
    )
    assert summary["Sharpe"] < 3, f"{name} Sharpe out of range: {summary['Sharpe']:.2f}"
    assert summary["CAGR"] < 0.5, f"{name} CAGR out of range: {summary['CAGR']:.2%}"
    assert summary["MaxDrawdown"] > -0.8, f"{name} drawdown out of range: {summary['MaxDrawdown']:.2%}"

    detail = outliers.rename("Return").reset_index().rename(columns={"index": "Date"})
    detail["Index"] = name
    detail = detail[["Index", "Date", "Return"]]
    return summary, detail, index_series, returns


def dividend_impact_check(prices, dividends):
    records = []
    for company, dividend_series in dividends.items():
        price_col = f"{company} Price"
        if price_col not in prices.columns:
            continue

        price_series = pd.to_numeric(prices.set_index("Date")[price_col], errors="coerce").sort_index()
        prev_price = price_series.shift(1)
        aligned_dividend = dividend_series.reindex(price_series.index).fillna(0.0)
        div_impact = (aligned_dividend / prev_price).replace([np.inf, -np.inf], np.nan).dropna()
        nonzero = div_impact[div_impact.ne(0)]
        if nonzero.empty:
            continue

        max_idx = nonzero.abs().idxmax()
        records.append(
            {
                "Company": company,
                "MaxDividendImpact": safe_float(nonzero.max()),
                "MinDividendImpact": safe_float(nonzero.min()),
                "MaxAbsDividendImpact": safe_float(nonzero.abs().max()),
                "ImpactDate": max_idx,
                "DividendValue": safe_float(aligned_dividend.loc[max_idx]),
                "PrevPrice": safe_float(prev_price.loc[max_idx]),
                "Flagged": bool(nonzero.abs().max() > DIVIDEND_IMPACT_THRESHOLD),
            }
        )

    impact_df = pd.DataFrame(records).sort_values("MaxAbsDividendImpact", ascending=False)
    print(f"Dividend impact max: {impact_df['MaxAbsDividendImpact'].max():.2%}" if not impact_df.empty else "Dividend impact max: n/a")
    return impact_df


def price_continuity_check(prices):
    records = []
    for price_col in price_columns(prices):
        company = company_from_price_col(price_col)
        series = pd.to_numeric(prices[price_col], errors="coerce")
        returns = series.pct_change()
        spikes = returns[(returns > PRICE_SPIKE_THRESHOLD) | (returns < -PRICE_SPIKE_THRESHOLD)]
        for idx, value in spikes.items():
            records.append(
                {
                    "Company": company,
                    "Date": prices.loc[idx, "Date"],
                    "PriceReturn": safe_float(value),
                }
            )
    spike_df = pd.DataFrame(records).sort_values(["Date", "Company"]) if records else pd.DataFrame(columns=["Company", "Date", "PriceReturn"])
    print(f"Price spike count: {len(spike_df)}")
    return spike_df


def weight_sum_check(path, label):
    df = pd.read_excel(path)
    sums = df.groupby("Date")["Weight"].sum().sort_index()
    assert np.allclose(sums.values, 1.0), f"{label} weights do not sum to 1"
    return pd.DataFrame(
        {
            "Index": label,
            "Date": sums.index,
            "WeightSum": sums.values,
        }
    )


def rebalance_jump_check(index_name, history_path):
    history = load_history(history_path)
    rebalance_dates = pd.to_datetime(sorted(history["Date"].unique()))
    index_series = load_index(index_name).set_index("Date")["Index Level"].sort_index()
    daily_returns = index_series.pct_change()
    records = []
    for date in rebalance_dates:
        window = daily_returns.loc[date - pd.Timedelta(days=2): date + pd.Timedelta(days=2)]
        if window.empty:
            continue
        records.append(
            {
                "Index": index_name,
                "RebalanceDate": date,
                "WindowMaxAbsReturn": safe_float(window.abs().max()),
                "Flagged": bool(window.abs().max() > REBALANCE_JUMP_THRESHOLD),
            }
        )
    return pd.DataFrame(records)


def coverage_impact_check(histories, dividends):
    records = []
    dividend_companies = set(dividends.keys())
    for index_name, history in histories.items():
        latest_date = history["Date"].max()
        latest = history[history["Date"] == latest_date].copy()
        latest["MissingDividend"] = ~latest["Company"].isin(dividend_companies)
        records.append(
            {
                "Index": index_name,
                "Date": latest_date,
                "MissingDividendCompanies": int(latest["MissingDividend"].sum()),
                "MissingDividendWeight": safe_float(latest.loc[latest["MissingDividend"], "Weight"].sum()),
                "TotalCompanies": int(len(latest)),
            }
        )
    return pd.DataFrame(records)


def country_concentration_check(prices, history_path, index_name):
    history = load_history(history_path)
    latest_date = history["Date"].max()
    latest = history[history["Date"] == latest_date].copy()
    trailing = prices.copy()
    trailing = trailing[trailing["Date"] <= latest_date].sort_values("Date").tail(252)
    records = []

    for country, group in latest.groupby("Country"):
        company_returns = []
        for _, row in group.iterrows():
            price_col = f"{row['Company']} Price"
            if price_col not in trailing.columns:
                continue
            series = pd.to_numeric(trailing[price_col], errors="coerce")
            ret = series.pct_change()
            company_returns.append(ret.rename(row["Company"]))
        if not company_returns:
            continue
        country_returns = pd.concat(company_returns, axis=1).mean(axis=1, skipna=True).dropna()
        records.append(
            {
                "Index": index_name,
                "Country": country,
                "Weight": safe_float(group["Weight"].sum()),
                "CountryVolatility": safe_float(annual_vol(country_returns)),
            }
        )
    return pd.DataFrame(records).sort_values(["Index", "Weight"], ascending=[True, False])


def control_test(prices, dividends):
    price_only_path = PROJECT_ROOT / "output" / "AM100_PRICE_ONLY_total_return.csv"
    total_return_path = PROJECT_ROOT / "output" / "AM100_total_return.csv"
    no_dividend_path = PROJECT_ROOT / "output" / "AM100_TOTAL_RETURN_AB_total_return.csv"

    if price_only_path.exists() and total_return_path.exists() and no_dividend_path.exists():
        price_only_index = pd.read_csv(price_only_path, parse_dates=["Date"])
        total_return_index = pd.read_csv(total_return_path, parse_dates=["Date"])
        no_dividend_index = pd.read_csv(no_dividend_path, parse_dates=["Date"])
    else:
        history = load_history(AM100_HISTORY_FILE)
        zero_dividends = {company: series * 0 for company, series in dividends.items()}
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink):
            price_only_index, _ = build_total_return_index(prices, history, {})
            total_return_index, _ = build_total_return_index(prices, history, dividends)
            no_dividend_index, _ = build_total_return_index(prices, history, zero_dividends)

    def block(label, df):
        index_series = df.set_index("Date")["Index Level"].sort_index()
        returns = index_series.pct_change().dropna()
        return {
            "Scenario": label,
            "CAGR": safe_float(calculate_cagr(index_series)),
            "Volatility": safe_float(annual_vol(returns)),
            "Sharpe": safe_float(calculate_sharpe(index_series)),
            "MaxDrawdown": safe_float(calculate_drawdown(index_series).min()),
            "LatestLevel": safe_float(index_series.iloc[-1]),
        }

    return pd.DataFrame(
        [
            block("Price-only", price_only_index),
            block("Total return", total_return_index),
            block("No-dividend", no_dividend_index),
        ]
    )


def write_report(summary_df, dividend_df, price_spikes_df, coverage_df, rebalance_df, country_df, control_df):
    lines = []
    lines.append("# Validation Report")
    lines.append("")
    lines.append("## 1. Return Distribution")
    for _, row in summary_df.iterrows():
        lines.append(
            f"- {row['Index']}: max daily return {row['MaxDailyReturn']:.2%}, "
            f"min daily return {row['MinDailyReturn']:.2%}, "
            f"extreme moves {int(row['ExtremeReturnCount'])}"
        )

    lines.append("")
    lines.append("## 2. Dividend Impact Summary")
    if dividend_df.empty:
        lines.append("- No dividend impact records found.")
    else:
        top = dividend_df.head(10)
        for _, row in top.iterrows():
            flag = " FLAG" if row["Flagged"] else ""
            lines.append(
                f"- {row['Company']}: max dividend impact {row['MaxAbsDividendImpact']:.2%} on {pd.to_datetime(row['ImpactDate']).date()}{flag}"
            )

    lines.append("")
    lines.append("## 3. Outlier Lists")
    lines.append(f"- Price spike count: {len(price_spikes_df)}")
    lines.append(f"- Dividend impact flags (>10%): {int(dividend_df['Flagged'].sum()) if not dividend_df.empty else 0}")

    lines.append("")
    lines.append("## 4. Volatility Breakdown")
    for _, row in summary_df.iterrows():
        lines.append(
            f"- {row['Index']}: annual vol {row['Volatility']:.2%}, "
            f"rolling vol range {row['RollingVolMin']:.2%} to {row['RollingVolMax']:.2%}"
        )

    lines.append("")
    lines.append("## 5. Sharpe Validation")
    for _, row in summary_df.iterrows():
        lines.append(
            f"- {row['Index']}: CAGR {row['CAGR']:.2%}, Sharpe {row['Sharpe']:.2f}, Max Drawdown {row['MaxDrawdown']:.2%}"
        )

    lines.append("")
    lines.append("## 6. Coverage Gaps")
    for _, row in coverage_df.iterrows():
        lines.append(
            f"- {row['Index']}: missing dividend companies {int(row['MissingDividendCompanies'])}, "
            f"missing dividend weight {row['MissingDividendWeight']:.2%}"
        )

    lines.append("")
    lines.append("## 7. Rebalance Continuity Check")
    flagged = rebalance_df[rebalance_df["Flagged"]]
    if flagged.empty:
        lines.append("- No flagged rebalance jump windows above 10%.")
    else:
        for _, row in flagged.head(20).iterrows():
            lines.append(
                f"- {row['Index']} {pd.to_datetime(row['RebalanceDate']).date()}: window max move {row['WindowMaxAbsReturn']:.2%}"
            )

    lines.append("")
    lines.append("## 8. Country Concentration")
    for index_name, group in country_df.groupby("Index"):
        top = group.head(3)
        pieces = [f"{row['Country']} ({row['Weight']:.2%} weight, {row['CountryVolatility']:.2%} vol)" for _, row in top.iterrows()]
        lines.append(f"- {index_name}: " + ", ".join(pieces))

    lines.append("")
    lines.append("## 9. Control Test")
    for _, row in control_df.iterrows():
        lines.append(
            f"- {row['Scenario']}: CAGR {row['CAGR']:.2%}, Vol {row['Volatility']:.2%}, Sharpe {row['Sharpe']:.2f}"
        )

    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")


def main():
    print("Running validation checks...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    os.chdir(PROJECT_ROOT)

    prices = load_price_panel()
    dividends = load_dividends()

    summaries = []
    outlier_frames = []
    index_series_map = {}
    returns_map = {}
    for name in INDEX_POOL:
        summary, outlier_df, index_series, returns = return_sanity_check(name)
        summaries.append(summary)
        if not outlier_df.empty:
            outlier_frames.append(outlier_df)
        index_series_map[name] = index_series
        returns_map[name] = returns

    summary_df = pd.DataFrame(summaries)
    outlier_df = pd.concat(outlier_frames, ignore_index=True) if outlier_frames else pd.DataFrame(columns=["Index", "Date", "Return"])
    dividend_df = dividend_impact_check(prices, dividends)
    price_spikes_df = price_continuity_check(prices)

    weight_sum_frames = [
        weight_sum_check(str(PROJECT_ROOT / "output" / "AM100_history.xlsx"), "AM100"),
        weight_sum_check(str(PROJECT_ROOT / "output" / "AM200_history.xlsx"), "AM200"),
        weight_sum_check(str(PROJECT_ROOT / "output" / "AM300_history.xlsx"), "AM300"),
    ]
    weight_sum_df = pd.concat(weight_sum_frames, ignore_index=True)

    rebalance_df = pd.concat(
        [
            rebalance_jump_check("AM100", str(PROJECT_ROOT / "output" / "AM100_history.xlsx")),
            rebalance_jump_check("AM200", str(PROJECT_ROOT / "output" / "AM200_history.xlsx")),
            rebalance_jump_check("AM300", str(PROJECT_ROOT / "output" / "AM300_history.xlsx")),
        ],
        ignore_index=True,
    )

    histories = {
        "AM100": load_history(str(PROJECT_ROOT / "output" / "AM100_history.xlsx")),
        "AM200": load_history(str(PROJECT_ROOT / "output" / "AM200_history.xlsx")),
        "AM300": load_history(str(PROJECT_ROOT / "output" / "AM300_history.xlsx")),
    }
    coverage_df = coverage_impact_check(histories, dividends)

    country_df = pd.concat(
        [
            country_concentration_check(prices, str(PROJECT_ROOT / "output" / "AM100_history.xlsx"), "AM100"),
            country_concentration_check(prices, str(PROJECT_ROOT / "output" / "AM200_history.xlsx"), "AM200"),
            country_concentration_check(prices, str(PROJECT_ROOT / "output" / "AM300_history.xlsx"), "AM300"),
        ],
        ignore_index=True,
    )

    control_df = control_test(prices, dividends)

    summary_df.to_csv(SUMMARY_FILE, index=False)
    outlier_df.to_csv(RETURN_SANITY_FILE, index=False)
    dividend_df.to_csv(DIVIDEND_IMPACT_FILE, index=False)
    price_spikes_df.to_csv(PRICE_SPIKE_FILE, index=False)
    rebalance_df.to_csv(REBALANCE_JUMP_FILE, index=False)
    coverage_df.to_csv(COVERAGE_FILE, index=False)
    country_df.to_csv(COUNTRY_CONCENTRATION_FILE, index=False)
    control_df.to_csv(CONTROL_TEST_FILE, index=False)

    write_report(summary_df, dividend_df, price_spikes_df, coverage_df, rebalance_df, country_df, control_df)

    print("Validation artifacts written:")
    for path in [
        SUMMARY_FILE,
        RETURN_SANITY_FILE,
        DIVIDEND_IMPACT_FILE,
        PRICE_SPIKE_FILE,
        REBALANCE_JUMP_FILE,
        COVERAGE_FILE,
        COUNTRY_CONCENTRATION_FILE,
        CONTROL_TEST_FILE,
        REPORT_FILE,
    ]:
        print(f"- {path}")

    assert weight_sum_df["WeightSum"].sub(1).abs().max() < 1e-8, "Weight sums drift from 1.0"
    assert summary_df["Sharpe"].max() < 3, "Sharpe out of range"
    assert summary_df["CAGR"].max() < 0.5, "CAGR out of range"
    assert summary_df["MaxDrawdown"].min() > -0.8, "Drawdown out of range"

    print("All validation checks passed.")


if __name__ == "__main__":
    main()
