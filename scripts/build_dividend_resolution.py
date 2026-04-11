from pathlib import Path
import re
import unicodedata
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.load_dividends import load_dividends


MASTER_FILE = PROJECT_ROOT / "output" / "master.xlsx"
DIVIDEND_DIR = PROJECT_ROOT / "data" / "dividends"
WORKBOOK_FILE = Path("/Users/derrythornalley/Downloads/Africa Dividend Book.xls")
OUTPUT_FILE = PROJECT_ROOT / "output" / "dividend_resolution.csv"
EXCEPTION_FILE = PROJECT_ROOT / "output" / "dividend_exception_list.csv"
SUMMARY_FILE = PROJECT_ROOT / "output" / "dividend_resolution_summary.csv"
COMPLETE_FILE = PROJECT_ROOT / "output" / "COVERAGE_COMPLETE.csv"
MISSING_DIVIDENDS_FILE = PROJECT_ROOT / "output" / "COVERAGE_MISSING_DIVIDENDS.csv"
MISSING_PRICES_FILE = PROJECT_ROOT / "output" / "COVERAGE_MISSING_PRICES.csv"
AM100_HISTORY_FILE = PROJECT_ROOT / "output" / "AM100_history.xlsx"
AM200_HISTORY_FILE = PROJECT_ROOT / "output" / "AM200_history.xlsx"
AM300_HISTORY_FILE = PROJECT_ROOT / "output" / "AM300_history.xlsx"

KNOWN_ZERO_DIVIDEND = {
    "BELTONE HOLDING EGYPT (EGYPT)",
    "EMAAR MISR EGYPT (EGYPT)",
    "FAWRY EGYPT (EGYPT)",
    "MARIDIVE & OIL SERVICE (EGYPT) (EGYPT)",
    "OANDO PLC NIGERIA (NIGERIA)",
    "SUEZ CANAL BANK (EGYPT)",
    "ZAMBEEF PRODUCTS (ZAMBIA)",
}

MANUAL_HEADER_MAP = {
    "COPPERBELT ENERGY CORPORATION (ZAMBIA)": "COPPERBELT ENERGY CORP ZAMBIA (ZAMBIA)",
    "STANDARD CHARTERED BANK KENYA (KENYA)": None,
}

TOKEN_NORMALIZATIONS = {
    " CORPORATION ": " CORP ",
    " HOLDINGS ": " HOLDING ",
    " INVESTMENTS ": " INVESTMENT ",
    " BREWERIES ": " BREWERY ",
    " TELECOMMUNICATIONS ": " TELECOMS ",
}


def ascii_upper(text):
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    return " ".join(text.upper().split())


def extract_country(name):
    match = re.search(r"\(([^()]+)\)\s*$", str(name))
    return match.group(1).upper() if match else ""


def normalize_key(name):
    text = ascii_upper(name)
    for raw, clean in TOKEN_NORMALIZATIONS.items():
        text = text.replace(raw, clean)
    text = re.sub(r"\(([^()]+)\)\s+\(\1\)$", r"(\1)", text)
    country = extract_country(text)
    if country:
        text = text.replace(f" {country}", " ")
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    return " ".join(text.split())


def company_stem(company, country):
    suffix = f" ({country})"
    return company[: -len(suffix)] if company.endswith(suffix) else company


def ensure_zero_dividend_file(company):
    country = extract_country(company).lower().replace(" ", "_")
    stem = company_stem(company, extract_country(company))
    target_dir = DIVIDEND_DIR / country
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{stem}.csv"
    if not target.exists():
        pd.DataFrame(columns=["Date", "Dividend"]).to_csv(target, index=False)
    return target


def workbook_header_map(master_companies):
    if not WORKBOOK_FILE.exists():
        return {}

    df = pd.read_excel(WORKBOOK_FILE)
    headers = [ascii_upper(c) for c in df.columns if not str(c).startswith("Unnamed:")]
    mapped = {}
    index = {}
    for company in master_companies:
        index.setdefault((extract_country(company), normalize_key(company)), []).append(company)

    for header in headers:
        manual = MANUAL_HEADER_MAP.get(header, "__manual_missing__")
        if manual != "__manual_missing__":
            if manual:
                mapped[manual] = header
            continue

        country = extract_country(header)
        candidates = index.get((country, normalize_key(header)), [])
        if len(candidates) == 1:
            mapped[candidates[0]] = header

    return mapped


def load_latest_history(path):
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_excel(path)
    if "Date" not in df.columns or "Company" not in df.columns:
        return pd.DataFrame()
    df["Date"] = pd.to_datetime(df["Date"])
    latest = df["Date"].max()
    return df[df["Date"] == latest].copy()


def build_materiality_map():
    latest_am100 = load_latest_history(AM100_HISTORY_FILE)
    latest_am200 = load_latest_history(AM200_HISTORY_FILE)
    latest_am300 = load_latest_history(AM300_HISTORY_FILE)

    material = set(latest_am100["Company"]) if not latest_am100.empty else set()

    combined = pd.concat(
        [df for df in (latest_am100, latest_am200, latest_am300) if not df.empty],
        ignore_index=True,
    )
    if combined.empty:
        return {}, set()

    if "Weight" in combined.columns:
        weight_material = set(combined.loc[combined["Weight"].fillna(0) > 0.01, "Company"])
        material.update(weight_material)

    if "Rank" in combined.columns:
        rank_frame = combined.groupby("Company", as_index=False)["Rank"].min()
        rank_material = set(rank_frame.loc[rank_frame["Rank"] <= 50, "Company"])
        material.update(rank_material)

    meta = {}
    for company, group in combined.groupby("Company"):
        meta[company] = {
            "LatestMaxWeight": float(group["Weight"].fillna(0).max()) if "Weight" in group else 0.0,
            "BestLiquidityRank": int(group["Rank"].min()) if "Rank" in group else None,
            "InAM100": bool((not latest_am100.empty) and company in set(latest_am100["Company"])),
        }

    return meta, material


def main():
    master = pd.read_excel(MASTER_FILE, nrows=1)
    master_companies = sorted(col[:-6] for col in master.columns if col.endswith(" Price"))
    dividends = load_dividends()
    header_lookup = workbook_header_map(master_companies)
    materiality_meta, material_companies = build_materiality_map()

    records = []
    all_companies = sorted(set(master_companies) | set(dividends.keys()))
    for company in all_companies:
        has_price_data = company in master_companies
        series = dividends.get(company)
        has_dividend_series = company in dividends
        events = int(series.dropna().shape[0]) if has_dividend_series else 0
        meta = materiality_meta.get(company, {})

        if events > 0:
            status = "OK"
            dividend_status = "OK"
            action = "None"
        elif company in KNOWN_ZERO_DIVIDEND or (
            company in header_lookup and has_dividend_series and events == 0
        ):
            target = ensure_zero_dividend_file(company)
            status = "Zero Dividend"
            dividend_status = "ZERO_CONFIRMED"
            action = f"Zero file ensured: {target}"
        elif company in header_lookup:
            status = "Mapping Required"
            dividend_status = "MAPPING_REQUIRED"
            action = f"Header present in workbook: {header_lookup[company]}"
        else:
            status = "Missing Source"
            dividend_status = "MISSING_SOURCE_ASSUMED_ZERO"
            action = "Assume zero; no mapped header or dividend file"

        materiality = "MATERIAL" if company in material_companies else "NON_MATERIAL"

        records.append(
            {
                "Company": company,
                "HasPriceData": has_price_data,
                "Status": status,
                "Dividend_Status": dividend_status,
                "Materiality": materiality,
                "Action": action,
                "ActionRequired": (
                    "SOURCE_DIVIDENDS"
                    if has_price_data and dividend_status != "OK"
                    else "SOURCE_PRICES"
                    if (not has_price_data) and dividend_status == "OK"
                    else "NONE"
                ),
                "DividendEvents": events,
                "WorkbookHeader": header_lookup.get(company),
                "HasDividendSeries": has_dividend_series,
                "LatestMaxWeight": meta.get("LatestMaxWeight"),
                "BestLiquidityRank": meta.get("BestLiquidityRank"),
                "InAM100": meta.get("InAM100", False),
            }
        )

    resolution = pd.DataFrame(records).sort_values(["Status", "Company"]).reset_index(drop=True)
    resolution.to_csv(OUTPUT_FILE, index=False)
    resolution.to_csv(SUMMARY_FILE, index=False)
    exceptions = resolution[resolution["Dividend_Status"] != "OK"].copy()
    exceptions.to_csv(EXCEPTION_FILE, index=False)

    complete = resolution[(resolution["HasPriceData"] == True) & (resolution["Dividend_Status"] == "OK")].copy()
    trade_only = resolution[(resolution["HasPriceData"] == True) & (resolution["Dividend_Status"] != "OK")].copy()
    div_only = resolution[(resolution["HasPriceData"] == False) & (resolution["Dividend_Status"] == "OK")].copy()

    complete.to_csv(COMPLETE_FILE, index=False)
    trade_only.to_csv(MISSING_DIVIDENDS_FILE, index=False)
    div_only.to_csv(MISSING_PRICES_FILE, index=False)

    print(f"Dividend resolution written to {OUTPUT_FILE}")
    print(f"Dividend resolution summary written to {SUMMARY_FILE}")
    print(f"Dividend exception list written to {EXCEPTION_FILE}")
    print("Coverage files generated:")
    print(f" - {COMPLETE_FILE}")
    print(f" - {MISSING_DIVIDENDS_FILE}")
    print(f" - {MISSING_PRICES_FILE}")
    print(resolution["Status"].value_counts().to_string())


if __name__ == "__main__":
    main()
