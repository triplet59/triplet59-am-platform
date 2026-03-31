import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data_validation import build_master_validation_report, validate_company_frame

BASE_DIR = "data"
BASE_FILE = "output/master.xlsx"
TEMP_BASE_FILE = "output/master_temp.xlsx"
CSV_FILE = "output/master.csv"
AUDIT_FILE = "output/audit_log.csv"
COUNTRY_SUMMARY_FILE = "output/country_summary.csv"
VALIDATION_FILE = "output/data_validation_report.csv"
START_DATE = "2016-01-01"
MIN_TRADING_DAYS = 252
RECENT_WINDOW_DAYS = 90
MIN_RECENT_TRADING_DAYS = 30
TOP_N_COMPANIES = 100
VALID_COUNTRIES = [
    "south_africa",
    "nigeria",
    "kenya",
    "tanzania",
    "malawi",
    "uganda",
    "zambia",
    "zimbabwe",
    "ghana",
    "rwanda",
    "namibia",
    "morocco",
    "tunisia",
    "senegal",
    "togo",
    "niger",
    "egypt",
    "mauritius",
]
audit_log = []
COLUMN_MAP = {
    "price": ["Close", "Price", "Last", "Last Price", "Closing Price"],
    "volume": ["Volume", "Shares", "Shares Traded", "Qty"],
}
NAME_MAP = {
    "BAT": "BRITISH AMERICAN TOBACCO",
    "ZANACO": "ZAMBIA NATIONAL COMMERCIAL BANK",
    "STANCHART": "STANDARD CHARTERED BANK",
    "MTN": "MTN GROUP",
    "IM BANK RWANDA": "I&M BANK",
    "MTN RWANDACELL RWANDA": "MTN RWANDACELL",
    "MTN RWANDACELL": "MTN RWANDA",
}

def load_or_create_master():
    if os.path.exists(BASE_FILE):
        print("📂 Loading existing master dataset...")
        df = pd.read_excel(BASE_FILE)
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    else:
        print("🆕 Creating new master dataset...")
        dates = pd.date_range(start=START_DATE, end=pd.Timestamp.today().normalize(), freq="D")
        df = pd.DataFrame({"Date": dates}).sort_values("Date", ascending=False)
    return df


def clean_company_name(name, country):
    name = name.upper().strip()
    name = " ".join(name.split())
    return f"{name} ({country.upper()})"


def clean_sa_name(raw_name):
    name = raw_name.upper()
    name = name.replace("STOCK PRICE HISTORY", "")
    name = name.replace("(1)", "")
    name = name.replace("(2)", "")
    name = name.replace(" LTD", "")
    name = name.replace(" GROUP", "")
    name = name.replace(" GRP", " GROUP")
    name = " ".join(name.split())
    return name


def company_exists(df, name):
    return any(col.startswith(name + " ") for col in df.columns)


def reorder_columns_by_latest_price(df):
    company_names = []
    for col in df.columns:
        if col.endswith(" Price"):
            company_names.append(col[:-6])

    def latest_price(company_name):
        price_col = f"{company_name} Price"
        series = pd.to_numeric(df[price_col], errors="coerce").dropna()
        if series.empty:
            return float("-inf")
        return series.iloc[0]

    sorted_companies = sorted(company_names, key=latest_price, reverse=True)[:TOP_N_COMPANIES]

    ordered_columns = ["Date"]
    for company_name in sorted_companies:
        for suffix in ("Price", "Change", "Volume"):
            col = f"{company_name} {suffix}"
            if col in df.columns:
                ordered_columns.append(col)

    remaining_columns = [col for col in df.columns if col not in ordered_columns]
    return df[ordered_columns + remaining_columns]


def parse_volume_string(value):
    if pd.isna(value):
        return None

    text = str(value).strip().upper().replace(",", "")
    if text in {"", "-", "—", "NA", "N/A"}:
        return None

    multiplier = 1
    if text.endswith("K"):
        multiplier = 1_000
        text = text[:-1]
    elif text.endswith("M"):
        multiplier = 1_000_000
        text = text[:-1]
    elif text.endswith("B"):
        multiplier = 1_000_000_000
        text = text[:-1]

    number = pd.to_numeric(text, errors="coerce")
    if pd.isna(number):
        return None
    return float(number) * multiplier


def load_excel_standard(filepath):
    excel_file = pd.ExcelFile(filepath)
    data = None

    for sheet_name in excel_file.sheet_names:
        candidate = pd.read_excel(filepath, sheet_name=sheet_name)
        if candidate.empty:
            continue

        candidate.columns = [str(c).strip().title() for c in candidate.columns]
        if "Date" in candidate.columns or "Trading Date" in candidate.columns:
            data = candidate
            break

    if data is None:
        data = pd.read_excel(filepath)
    print(f"      📊 Rows loaded: {len(data)}")

    data.columns = [str(c).strip().title() for c in data.columns]

    if "Trading Date" in data.columns and "Date" not in data.columns:
        data["Date"] = data["Trading Date"]
    if "Last Price" in data.columns and "Price" not in data.columns:
        data["Price"] = data["Last Price"]
    if "Turnover" in data.columns and "Value" not in data.columns:
        data["Value"] = data["Turnover"]

    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data = data.sort_values("Date")

    price_candidates = [col.title() for col in COLUMN_MAP["price"]]
    price_col = next((col for col in price_candidates if col in data.columns), None)
    if price_col is None:
        data["Price"] = pd.Series(pd.NA, index=data.index)
    else:
        data["Price"] = (
            data[price_col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("Rs", "", regex=False)
            .str.strip()
        )
        data["Price"] = pd.to_numeric(data["Price"], errors="coerce")

    volume_candidates = [col.title() for col in COLUMN_MAP["volume"]] + ["Turnover"]
    for col in volume_candidates:
        if col in data.columns:
            data["Volume"] = data[col]
            break

    if "Volume" in data.columns:
        data["Volume"] = (
            data["Volume"]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.strip()
            .replace(["-", "—", "NA", "N/A", ""], None)
        )
        data["Volume"] = pd.to_numeric(data["Volume"], errors="coerce")
    else:
        data["Volume"] = None

    if data["Volume"].isna().all():
        value_col = next((col for col in ["Value", "Turnover"] if col in data.columns), None)
        if value_col is not None:
            value_series = (
                data[value_col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.strip()
                .replace(["-", "—", "NA", "N/A", ""], None)
            )
            value_series = pd.to_numeric(value_series, errors="coerce")
            data["Volume"] = value_series.div(data["Price"].replace(0, pd.NA))
            data["Volume"] = data["Volume"].replace([np.inf, -np.inf], np.nan)
            data["Volume"] = data["Volume"].fillna(0)

    if "Price_USD" not in data.columns:
        data["Price_USD"] = data["Price"]

    return data[["Date", "Price", "Price_USD", "Volume"]]


def load_sa_csv(filepath):
    data = pd.read_csv(filepath)
    print(f"      📊 Rows loaded: {len(data)}")

    # --- Clean Date ---
    data["Date"] = pd.to_datetime(data["Date"], format="%m/%d/%Y", errors="coerce")

    # --- Clean Price ---
    data["Price"] = (
        data["Price"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    data["Price"] = pd.to_numeric(data["Price"], errors="coerce")

    # --- Detect & fix scale (VERY IMPORTANT) ---
    if data["Price"].dropna().median() > 1000:
        data["Price"] = data["Price"] / 100

    # --- Clean Volume ---
    volume_col = "Vol." if "Vol." in data.columns else "Volume"
    if volume_col in data.columns:
        data["Volume"] = data[volume_col].apply(parse_volume_string)
    else:
        data["Volume"] = None

    # --- Keep only required fields ---
    data["Price_USD"] = data["Price"]
    data = data[["Date", "Price", "Price_USD", "Volume"]]

    # --- Drop bad rows ---
    data = data.dropna(subset=["Date"])

    # --- Sort ascending ---
    data = data.sort_values("Date")
    return data


def load_company_data(filepath, country_name):
    if country_name == "MAURITIUS":
        data = pd.read_excel(filepath)
        file = os.path.basename(filepath)
        print(f"[MAURITIUS] Loading file: {file}")
        print(data.head())
        print(f"[MAURITIUS] Columns: {list(data.columns)}")
        return load_excel_standard(filepath)
    if country_name in {"SOUTH_AFRICA", "NAMIBIA"}:
        return load_sa_csv(filepath)
    return load_excel_standard(filepath)


def process_company(file_path, name, master_dates, country_name):
    data = load_company_data(file_path, country_name)
    data, validation = validate_company_frame(data, name)
    data["Price_USD"] = pd.to_numeric(
        data["Price_USD"] if "Price_USD" in data.columns else data["Price"],
        errors="coerce",
    )

    valid_ratio = data["Volume"].notna().mean()
    print(f"      {name} volume coverage: {valid_ratio:.2%}")
    if validation["HasMissingDataFlag"]:
        print(
            f"      ⚠️ Missing data flag: "
            f"prices={validation['MissingPriceCount']}, volumes={validation['MissingVolumeCount']}"
        )
    if validation["HasOutlierReturnFlag"]:
        print(
            f"      ⚠️ Outlier return flag: "
            f"{validation['OutlierReturnCount']} daily moves beyond 20%"
        )
    if valid_ratio == 0:
        audit_log.append({
            "company": name,
            "country": country_name,
            "status": "EXCLUDED",
            "reason": "no valid volume",
        })
        print(f"⏭ Skipping (no valid volume): {name}")
        print(f"[REJECTED] {name} → no valid volume")
        return None

    if pd.to_numeric(data["Volume"], errors="coerce").fillna(0).sum() == 0:
        audit_log.append({
            "company": name,
            "country": country_name,
            "status": "EXCLUDED",
            "reason": "derived volume invalid",
        })
        print(f"[REJECTED] {name} → derived volume invalid")
        return None

    data["Change"] = data["Price_USD"].pct_change()

    data = data[["Date", "Price_USD", "Change", "Volume"]]

    print("      🔗 Aligning with master dates...")
    merged = master_dates.merge(data, on="Date", how="left")

    merged.rename(columns={
        "Price_USD": f"{name} Price",
        "Change": f"{name} Change",
        "Volume": f"{name} Volume"
    }, inplace=True)

    return merged


def has_min_trading_days(file_path, country_name, min_days=MIN_TRADING_DAYS):
    data = load_company_data(file_path, country_name)[["Date"]].copy()
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    trading_days = data["Date"].dropna().nunique()
    return trading_days >= min_days, trading_days


def has_recent_trading_activity(
    file_path,
    country_name,
    window_days=RECENT_WINDOW_DAYS,
    min_days=MIN_RECENT_TRADING_DAYS,
):
    data = load_company_data(file_path, country_name)[["Date"]].copy()
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    dates = data["Date"].dropna()
    if dates.empty:
        return False, 0

    cutoff_date = pd.Timestamp.today().normalize() - pd.Timedelta(days=window_days)
    recent_trading_days = dates[dates >= cutoff_date].nunique()
    return recent_trading_days >= min_days, recent_trading_days


def has_valid_recent_prices(file_path, country_name, window_days=RECENT_WINDOW_DAYS):
    data = load_company_data(file_path, country_name).copy()
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    price_col = "Price_USD" if "Price_USD" in data.columns else "Price"
    data["Price"] = pd.to_numeric(data[price_col], errors="coerce")
    data = data.dropna(subset=["Date"])

    cutoff_date = pd.Timestamp.today().normalize() - pd.Timedelta(days=window_days)
    recent_data = data[data["Date"] >= cutoff_date]
    if recent_data.empty:
        return False

    recent_prices = recent_data["Price"]
    return not recent_prices.isna().any() and not (recent_prices == 0).any()


def append_country(master_dates, folder_path, country_name):
    company_dfs = []

    for file in sorted(os.listdir(folder_path)):
        valid_suffixes = (".csv",) if country_name in {"SOUTH_AFRICA", "NAMIBIA"} else (".xlsx",)
        if file.endswith(valid_suffixes):
            print(f"\nProcessing file: {file}")
            if country_name == "MAURITIUS":
                print(f"\n[MAURITIUS] Found file: {file}")
            raw_name = os.path.splitext(file)[0]
            country_clean = country_name.replace("_", " ").upper()
            if country_name in {"SOUTH_AFRICA", "NAMIBIA"}:
                raw_name = clean_sa_name(raw_name)
            clean_name = clean_company_name(raw_name, country_clean)
            standard_name = NAME_MAP.get(clean_name.replace(f" ({country_clean})", ""), clean_name)
            company_name = (
                standard_name
                if standard_name.endswith(f" ({country_clean})")
                else f"{standard_name} ({country_clean})"
            )
            file_path = os.path.join(folder_path, file)

            print(f"➕ Adding: {company_name}")
            print(f"      🔍 Loading file: {file_path}")
            file_start = time.time()
            flag_low_history = False
            flag_illiquid = False

            step_start = time.time()
            has_enough_days, trading_days = has_min_trading_days(file_path, country_name)
            print(f"   history check: {round(time.time() - step_start, 2)}s")
            if not has_enough_days:
                flag_low_history = True
                print(f"⚠️ Low history flag ({trading_days} trading days): {company_name}")

            step_start = time.time()
            has_recent_days, recent_trading_days = has_recent_trading_activity(
                file_path, country_name
            )
            print(f"   recent activity check: {round(time.time() - step_start, 2)}s")
            if not has_recent_days:
                flag_illiquid = True
                print(
                    f"⚠️ Illiquid flag ({recent_trading_days} trading days in last "
                    f"{RECENT_WINDOW_DAYS} days): {company_name}"
                )

            step_start = time.time()
            if not has_valid_recent_prices(file_path, country_name):
                print(f"   recent price check: {round(time.time() - step_start, 2)}s")
                print(
                    f"⏭ Skipping (price is 0 or NaN in last {RECENT_WINDOW_DAYS} days): "
                    f"{company_name}"
                )
                print(
                    f"[REJECTED] {company_name} → price is 0 or NaN in last "
                    f"{RECENT_WINDOW_DAYS} days"
                )
                audit_log.append({
                    "company": company_name,
                    "country": country_name,
                    "status": "EXCLUDED",
                    "reason": f"price is 0 or NaN in last {RECENT_WINDOW_DAYS} days",
                    "flag_low_history": flag_low_history,
                    "flag_illiquid": flag_illiquid,
                })
                continue
            print(f"   recent price check: {round(time.time() - step_start, 2)}s")

            step_start = time.time()
            comp = process_company(file_path, company_name, master_dates, country_name)
            print(f"   process company: {round(time.time() - step_start, 2)}s")
            print(f"   total file time: {round(time.time() - file_start, 2)}s")
            if comp is not None:
                audit_log.append({
                    "company": company_name,
                    "country": country_name,
                    "status": "INCLUDED",
                    "reason": "passed filters",
                    "flag_low_history": flag_low_history,
                    "flag_illiquid": flag_illiquid,
                    "flag_missing_data": validation["HasMissingDataFlag"] if "validation" in locals() else False,
                    "flag_outlier_return": validation["HasOutlierReturnFlag"] if "validation" in locals() else False,
                })
                company_dfs.append(comp)

    print("\n🚧 STARTING FINAL JOIN STAGE")
    date_df = master_dates.copy()
    date_df["Date"] = pd.to_datetime(date_df["Date"], errors="coerce")
    date_df = (
        date_df
        .dropna()
        .drop_duplicates(subset="Date")
        .sort_values("Date")
        .set_index("Date")
    )
    date_index = date_df.index
    dfs = [date_df]

    for company_df in company_dfs:
        company_df["Date"] = pd.to_datetime(company_df["Date"], errors="coerce")
        company_df = (
            company_df
            .dropna(subset=["Date"])
            .sort_values("Date")
            .drop_duplicates(subset="Date", keep="last")
            .set_index("Date")
            .reindex(date_index)
        )
        dfs.append(company_df)

    print("STEP 1: concat starting")
    country_master = pd.concat(dfs, axis=1)
    print("STEP 2: concat done")
    print("Post-concat shape:", country_master.shape)

    dupes = country_master.columns[country_master.columns.duplicated()]
    print("Duplicate columns post-concat:", dupes.tolist()[:10])
    if len(dupes) > 0:
        print("WARNING: Duplicate columns:", dupes.tolist())
        country_master = country_master.loc[:, ~country_master.columns.duplicated()]

    country_master.index = pd.to_datetime(country_master.index, errors="coerce")
    if not country_master.index.is_unique:
        print("ERROR: MASTER INDEX NOT UNIQUE")
        country_master = country_master[~country_master.index.duplicated(keep="first")]

    print("⚡ Combined company frames")
    print("STEP 3: checking index type", type(country_master.index))
    print("STEP 4: sort starting")
    print("STEP 5: sort done")
    print("STEP 6: reset_index starting")
    country_master = country_master.reset_index()
    print("STEP 7: reset_index done")

    price_cols = [c for c in country_master.columns if "Price" in c]
    print(f"Total companies in master: {len(price_cols)}")
    print(f"Final dataset shape: {country_master.shape}")
    print("✅ JOIN COMPLETE")
    return country_master


def build_dataset():
    audit_log.clear()
    all_country_dfs = []
    country_summary = []
    base_master = load_or_create_master()
    countries = sorted(
        d for d in os.listdir(BASE_DIR)
        if os.path.isdir(os.path.join(BASE_DIR, d))
    )

    for country in countries:
        print(f"[DISCOVERY] Raw folder detected: '{country}'")
        country_clean = country.strip().lower()

        if country_clean == "dividends":
            continue

        if country_clean not in VALID_COUNTRIES:
            continue

        country_upper = country_clean.upper()
        folder_path = os.path.join(BASE_DIR, country)
        files = os.listdir(folder_path)

        print(f"Scanning country folder: {country_upper}")
        print(f"[{country_upper}] File count: {len(files)}")

        country_df = append_country(base_master[["Date"]].copy(), folder_path, country_upper)

        if country_df is not None:
            price_cols = [c for c in country_df.columns if c.endswith(" Price")]
            country_summary.append({
                "country": country_upper,
                "companies": len(price_cols),
            })
            all_country_dfs.append(country_df.set_index("Date"))

    if not all_country_dfs:
        print("⚠️ No country data generated")
        return

    master = pd.concat(all_country_dfs, axis=1)

    dupes = master.columns[master.columns.duplicated()]
    if len(dupes) > 0:
        print("WARNING: Duplicate columns detected:", dupes.tolist())
        master = master.loc[:, ~master.columns.duplicated()]

    if not master.index.is_unique:
        print("WARNING: Duplicate dates in final master — fixing")
        master = master[~master.index.duplicated(keep="first")]

    master = master.sort_index(ascending=False).reset_index()
    master["Date"] = pd.to_datetime(master["Date"], errors="coerce").dt.strftime("%d/%m/%Y")

    os.makedirs("output", exist_ok=True)
    master.to_csv(CSV_FILE, index=False)
    pd.DataFrame(audit_log).to_csv(AUDIT_FILE, index=False)
    pd.DataFrame(country_summary).to_csv(COUNTRY_SUMMARY_FILE, index=False)
    build_master_validation_report(master).to_csv(VALIDATION_FILE, index=False)
    try:
        if os.path.exists(TEMP_BASE_FILE):
            os.remove(TEMP_BASE_FILE)

        master.to_excel(TEMP_BASE_FILE, index=False)
        os.replace(TEMP_BASE_FILE, BASE_FILE)
    except Exception as exc:
        if os.path.exists(TEMP_BASE_FILE):
            os.remove(TEMP_BASE_FILE)
        raise exc

    print("✅ Master updated across all countries")


if __name__ == "__main__":
    build_dataset()
