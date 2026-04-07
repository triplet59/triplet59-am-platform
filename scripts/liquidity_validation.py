from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
PROJECT_ROOT = Path(SCRIPT_DIR).parent
OUTPUT_DIR = PROJECT_ROOT / "output"

from liquidity_tools import sanitize_traded_value_series

MASTER_CSV = OUTPUT_DIR / "master.csv"
MASTER_XLSX = OUTPUT_DIR / "master.xlsx"

SUMMARY_FILE = OUTPUT_DIR / "liquidity_validation_summary.csv"
STOCK_FILE = OUTPUT_DIR / "liquidity_stock_summary.csv"
OUTLIER_FILE = OUTPUT_DIR / "liquidity_outlier_days.csv"
COUNTRY_FILE = OUTPUT_DIR / "liquidity_country_breakdown.csv"
ANOMALY_FILE = OUTPUT_DIR / "liquidity_flagged_anomalies.csv"
REPORT_FILE = OUTPUT_DIR / "LIQUIDITY_AUDIT_REPORT.md"


def load_master():
    if MASTER_CSV.exists():
        df = pd.read_csv(MASTER_CSV)
    elif MASTER_XLSX.exists():
        df = pd.read_excel(MASTER_XLSX)
    else:
        raise FileNotFoundError("Missing master dataset")
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    return df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)


def company_columns(master):
    cols = []
    for col in master.columns:
        if not col.endswith(" Volume"):
            continue
        company = col[:-7]
        price_usd_col = f"{company} Price USD"
        price_col = f"{company} Price"
        if price_usd_col in master.columns:
            cols.append((company, price_usd_col, col))
        elif price_col in master.columns:
            cols.append((company, price_col, col))
    return cols


def annualized_adv(series: pd.Series) -> pd.Series:
    valid = series.notna()
    return series[valid].rolling(30).mean().reindex(series.index).ffill()


def audit_company(master, company, price_col, volume_col):
    dates = master["Date"]
    price = pd.to_numeric(master[price_col], errors="coerce")
    volume = pd.to_numeric(master[volume_col], errors="coerce")
    clean_tv, diagnostics = sanitize_traded_value_series(price, volume)
    positive_tv = clean_tv[clean_tv.gt(0)]
    repeats = volume.diff().abs().eq(0) & volume.gt(0)

    if positive_tv.empty:
        return None, None, None

    outlier_threshold = positive_tv.quantile(0.999)
    outlier_mask = clean_tv.gt(outlier_threshold)
    outlier_rows = pd.DataFrame(
        {
            "Company": company,
            "Date": dates[outlier_mask],
            "TradedValueUSD": clean_tv[outlier_mask],
            "Price": price[outlier_mask],
            "Volume": volume[outlier_mask],
        }
    ).dropna(subset=["Date"])

    country = company.rsplit("(", 1)[-1].rstrip(")")
    adv30 = annualized_adv(clean_tv)
    adv_non_null = adv30.dropna()
    latest_adv = float(adv_non_null.iloc[-1]) if not adv_non_null.empty else np.nan

    stock_summary = {
        "Company": company,
        "Country": country,
        "MedianTradedValueUSD": float(positive_tv.median()),
        "MeanTradedValueUSD": float(positive_tv.mean()),
        "P75TradedValueUSD": float(positive_tv.quantile(0.75)),
        "P90TradedValueUSD": float(positive_tv.quantile(0.90)),
        "P99TradedValueUSD": float(positive_tv.quantile(0.99)),
        "MaxTradedValueUSD": float(positive_tv.max()),
        "LatestADV30USD": latest_adv,
        "LatestCapacity20USD": float(latest_adv * 0.20) if pd.notna(latest_adv) else np.nan,
        "ZeroOrMissingVolumeDays": diagnostics["zero_or_missing_volume_days"],
        "StaleVolumeDays": diagnostics["stale_volume_days"],
        "ClippedValueDays": diagnostics["clipped_value_days"],
        "OutlierDays": int(outlier_mask.sum()),
        "RepeatedVolumeDays": int(repeats.sum()),
        "MaxVolume": float(volume.max()) if volume.notna().any() else np.nan,
        "MedianVolume": float(volume.median()) if volume.notna().any() else np.nan,
    }

    anomalies = []
    if stock_summary["MedianTradedValueUSD"] > 50_000_000:
        anomalies.append("median_value_above_50m")
    if stock_summary["MaxTradedValueUSD"] > 100_000_000:
        anomalies.append("max_value_above_100m")
    if stock_summary["StaleVolumeDays"] > 20:
        anomalies.append("stale_volume")
    if stock_summary["RepeatedVolumeDays"] > 50:
        anomalies.append("repeated_volume")
    if stock_summary["OutlierDays"] > 0:
        anomalies.append("outlier_days")

    anomaly_rows = pd.DataFrame(
        [{"Company": company, "Country": country, "Issue": issue} for issue in anomalies]
    )

    return stock_summary, outlier_rows, anomaly_rows


def build_report(summary_df, stock_df, country_df, anomaly_df, outlier_df):
    top20 = stock_df.nlargest(20, "LatestADV30USD")[
        ["Company", "Country", "LatestADV30USD", "LatestCapacity20USD"]
    ]
    def plain_table(df: pd.DataFrame) -> str:
        return df.to_string(index=False)

    lines = [
        "# Liquidity Audit Report",
        "",
        f"- Stocks audited: {len(stock_df)}",
        f"- Flagged anomalies: {len(anomaly_df)}",
        f"- Outlier days: {len(outlier_df)}",
        "",
        "## Summary",
        "",
        "```text",
        plain_table(summary_df),
        "```",
        "",
        "## Top 20 Most Liquid Stocks",
        "",
        "```text",
        plain_table(top20),
        "```",
        "",
        "## Country Breakdown",
        "",
        "```text",
        plain_table(country_df),
        "```",
    ]
    REPORT_FILE.write_text("\n".join(lines))


def main():
    master = load_master()
    stock_rows = []
    outlier_frames = []
    anomaly_frames = []

    for company, price_col, volume_col in company_columns(master):
        stock_summary, outlier_rows, anomaly_rows = audit_company(master, company, price_col, volume_col)
        if stock_summary is None:
            continue
        stock_rows.append(stock_summary)
        if outlier_rows is not None and not outlier_rows.empty:
            outlier_frames.append(outlier_rows)
        if anomaly_rows is not None and not anomaly_rows.empty:
            anomaly_frames.append(anomaly_rows)

    stock_df = pd.DataFrame(stock_rows).sort_values("LatestADV30USD", ascending=False)
    outlier_df = pd.concat(outlier_frames, ignore_index=True) if outlier_frames else pd.DataFrame(columns=["Company", "Date", "TradedValueUSD", "Price", "Volume"])
    anomaly_df = pd.concat(anomaly_frames, ignore_index=True) if anomaly_frames else pd.DataFrame(columns=["Company", "Country", "Issue"])

    country_df = (
        stock_df.groupby("Country")[["LatestADV30USD", "LatestCapacity20USD"]]
        .sum(min_count=1)
        .reset_index()
        .sort_values("LatestADV30USD", ascending=False)
    )

    summary_df = pd.DataFrame(
        [
            {
                "Metric": "Median stock ADV (USD)",
                "Value": float(stock_df["LatestADV30USD"].median()),
            },
            {
                "Metric": "Mean stock ADV (USD)",
                "Value": float(stock_df["LatestADV30USD"].mean()),
            },
            {
                "Metric": "Stocks with stale volume",
                "Value": int((stock_df["StaleVolumeDays"] > 0).sum()),
            },
            {
                "Metric": "Stocks with outlier days",
                "Value": int((stock_df["OutlierDays"] > 0).sum()),
            },
            {
                "Metric": "Aggregate AM100/200/300-style capacity ceiling proxy (USD)",
                "Value": float(stock_df["LatestCapacity20USD"].sum()),
            },
        ]
    )

    summary_df.to_csv(SUMMARY_FILE, index=False)
    stock_df.to_csv(STOCK_FILE, index=False)
    outlier_df.to_csv(OUTLIER_FILE, index=False)
    country_df.to_csv(COUNTRY_FILE, index=False)
    anomaly_df.to_csv(ANOMALY_FILE, index=False)
    build_report(summary_df, stock_df, country_df, anomaly_df, outlier_df)

    print("Liquidity validation artifacts written:")
    print(f"- {SUMMARY_FILE}")
    print(f"- {STOCK_FILE}")
    print(f"- {OUTLIER_FILE}")
    print(f"- {COUNTRY_FILE}")
    print(f"- {ANOMALY_FILE}")
    print(f"- {REPORT_FILE}")


if __name__ == "__main__":
    main()
