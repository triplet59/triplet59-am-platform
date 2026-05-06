from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.liquidity_tools import sanitize_traded_value_series

MASTER_FILE = PROJECT_ROOT / "output" / "master.csv"
AUDIT_FILE = PROJECT_ROOT / "output" / "company_qualification_audit.csv"
FOCUS_FILE = PROJECT_ROOT / "output" / "company_qualification_focus.csv"

AM100_HISTORY = PROJECT_ROOT / "output" / "AM100_history.xlsx"
AM200_HISTORY = PROJECT_ROOT / "output" / "AM200_history.xlsx"
AM300_HISTORY = PROJECT_ROOT / "output" / "AM300_history.xlsx"

DIVIDEND_RESOLUTION_FILE = PROJECT_ROOT / "output" / "dividend_resolution.csv"

AM100_MIN_ADV = 1_000_000
AM200_MIN_ADV = 500_000
AM300_MIN_ADV = 200_000
AM300_MIN_COVERAGE = 0.60
GAP_THRESHOLD_DAYS = 30
REFERENCE_START = pd.Timestamp("2016-01-01")
EXCLUDED_COLLECTIVE_INVESTMENT_SECURITIES = {
    "CORONATION (NAMIBIA)",
}


def load_master() -> pd.DataFrame:
    if not MASTER_FILE.exists():
        raise FileNotFoundError(f"Missing master file: {MASTER_FILE}")
    master = pd.read_csv(MASTER_FILE)
    master["Date"] = pd.to_datetime(master["Date"], dayfirst=True, errors="coerce")
    return master.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)


def company_names(master: pd.DataFrame) -> list[str]:
    return sorted(col[:-6] for col in master.columns if col.endswith(" Price"))


def load_latest_members(path: Path) -> set[str]:
    if not path.exists():
        return set()
    history = pd.read_excel(path)
    if history.empty:
        return set()
    latest_date = pd.to_datetime(history["Date"], errors="coerce").max()
    return set(history.loc[pd.to_datetime(history["Date"], errors="coerce") == latest_date, "Company"].astype(str))


def load_dividend_status() -> dict[str, str]:
    if not DIVIDEND_RESOLUTION_FILE.exists():
        return {}
    resolution = pd.read_csv(DIVIDEND_RESOLUTION_FILE)
    resolution["Company"] = resolution["Company"].astype(str).str.strip()
    resolution["Dividend_Status"] = resolution["Dividend_Status"].astype(str).str.strip()
    return dict(zip(resolution["Company"], resolution["Dividend_Status"]))


def compute_company_metrics(master: pd.DataFrame, company: str) -> dict:
    price_col = f"{company} Price"
    volume_col = f"{company} Volume"
    if price_col not in master.columns or volume_col not in master.columns:
        return {}

    dates = master["Date"]
    price = pd.to_numeric(master[price_col], errors="coerce")
    volume = pd.to_numeric(master[volume_col], errors="coerce")

    valid_price = price.notna()
    valid_dates = dates[valid_price]

    if valid_dates.empty:
        return {
            "Company": company,
            "CoveragePct": 0.0,
            "FirstValidDate": pd.NaT,
            "LastValidDate": pd.NaT,
            "LargeGapCount": 0,
            "MaxGapDays": np.nan,
            "LatestAvgDailyUSDLiquidity": np.nan,
            "MeanAvgDailyUSDLiquidity": np.nan,
        }

    gaps = valid_dates.sort_values().diff().dt.days.dropna()
    large_gap_count = int((gaps > GAP_THRESHOLD_DAYS).sum())
    max_gap_days = int(gaps.max()) if not gaps.empty else np.nan

    traded_value_usd, _ = sanitize_traded_value_series(price, volume)
    adv30 = traded_value_usd.where(traded_value_usd.gt(0)).rolling(30, min_periods=1).mean()
    latest_adv = float(adv30.dropna().iloc[-1]) if adv30.dropna().any() else np.nan
    mean_adv = float(adv30.dropna().mean()) if adv30.dropna().any() else np.nan

    return {
        "Company": company,
        "CoveragePct": float(valid_price.mean()),
        "FirstValidDate": valid_dates.min(),
        "LastValidDate": valid_dates.max(),
        "LargeGapCount": large_gap_count,
        "MaxGapDays": max_gap_days,
        "LatestAvgDailyUSDLiquidity": latest_adv,
        "MeanAvgDailyUSDLiquidity": mean_adv,
    }


def liquidity_tier(latest_adv: float) -> str:
    if pd.isna(latest_adv):
        return "No Liquidity Data"
    if latest_adv >= AM100_MIN_ADV:
        return "AM100-scale"
    if latest_adv >= AM200_MIN_ADV:
        return "AM200-scale"
    if latest_adv >= AM300_MIN_ADV:
        return "AM300-scale"
    return "Below AM300-scale"


def build_reason(row: pd.Series) -> str:
    reasons: list[str] = []
    if row["Company"] in EXCLUDED_COLLECTIVE_INVESTMENT_SECURITIES:
        reasons.append("excluded_collective_investment_security")
    if row["CoveragePct"] < 0.95:
        reasons.append("coverage_below_95pct")
    if pd.notna(row["FirstValidDate"]) and pd.Timestamp(row["FirstValidDate"]) > REFERENCE_START:
        reasons.append("starts_after_2016")
    if pd.notna(row["LargeGapCount"]) and row["LargeGapCount"] > 0:
        reasons.append("large_data_gaps")
    if pd.isna(row["LatestAvgDailyUSDLiquidity"]) or row["LatestAvgDailyUSDLiquidity"] < AM300_MIN_ADV:
        reasons.append("below_am300_scale")
    if row["CoveragePct"] < AM300_MIN_COVERAGE:
        reasons.append("below_am300_coverage")
    return "; ".join(dict.fromkeys(reasons))


def build_status(row: pd.Series) -> str:
    if row["AM100"]:
        return "Included AM100"
    if row["AM200"]:
        return "Included AM200"
    if row["AM300"]:
        return "Included AM300"
    return "Excluded"


def main():
    master = load_master()
    dividend_status = load_dividend_status()

    am100_members = load_latest_members(AM100_HISTORY)
    am200_members = load_latest_members(AM200_HISTORY)
    am300_members = load_latest_members(AM300_HISTORY)

    metrics = [compute_company_metrics(master, company) for company in company_names(master)]
    audit = pd.DataFrame(metrics)
    audit = audit.dropna(subset=["Company"]).copy()

    audit["LiquidityTier"] = audit["LatestAvgDailyUSDLiquidity"].apply(liquidity_tier)
    audit["LatestRank"] = audit["LatestAvgDailyUSDLiquidity"].rank(method="min", ascending=False)
    audit["AM100"] = audit["Company"].isin(am100_members)
    audit["AM200"] = audit["Company"].isin(am200_members)
    audit["AM300"] = audit["Company"].isin(am300_members)
    audit["Status"] = audit.apply(build_status, axis=1)
    audit["Reason"] = audit.apply(build_reason, axis=1)
    audit["Dividend_Status"] = audit["Company"].map(dividend_status)

    audit = audit.sort_values(
        ["AM100", "AM200", "AM300", "LatestAvgDailyUSDLiquidity"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    output_cols = [
        "Company",
        "CoveragePct",
        "FirstValidDate",
        "LastValidDate",
        "LargeGapCount",
        "MaxGapDays",
        "LatestAvgDailyUSDLiquidity",
        "MeanAvgDailyUSDLiquidity",
        "LiquidityTier",
        "LatestRank",
        "AM100",
        "AM200",
        "AM300",
        "Status",
        "Reason",
    ]

    audit[output_cols].to_csv(AUDIT_FILE, index=False)
    audit[output_cols].to_csv(FOCUS_FILE, index=False)
    print(f"✅ Qualification audit refreshed -> {AUDIT_FILE}")
    print(f"✅ Qualification focus refreshed -> {FOCUS_FILE}")


if __name__ == "__main__":
    main()
