from pathlib import Path
import re

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DIVIDEND_SOURCE = Path("/Users/derrythornalley/Downloads/African Companies Dividend Data.csv")
ISIN_SOURCE = Path("/Users/derrythornalley/Downloads/African Companies ISIN List - List.csv")
OUTPUT_DIR = PROJECT_ROOT / "data" / "dividends"
DIVIDEND_OUTLIER_FILE = PROJECT_ROOT / "output" / "dividend_outliers.csv"
DIVIDEND_RULES = {
    "SOUTH AFRICA": {"type": "cents", "factor": 0.01},
    "KENYA": {"type": "normal"},
    "NIGERIA": {"type": "normal"},
    "EGYPT": {"type": "normal"},
    "GHANA": {"type": "normal"},
    "ZAMBIA": {"type": "normal"},
    "NAMIBIA": {"type": "cents", "factor": 0.01},
}


def slugify_country(country):
    return country.strip().lower().replace(" ", "_")


def clean_dividend_value(value):
    if pd.isna(value):
        return None

    text = str(value).strip()
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None

    try:
        return float(match.group(0))
    except ValueError:
        return None


def parse_dividend_date(value):
    if pd.isna(value):
        return pd.NaT

    text = str(value).strip()
    if not text:
        return pd.NaT

    if re.fullmatch(r"\d{1,2}/\d{1,2}/\d{4}", text):
        return pd.to_datetime(text, errors="coerce", dayfirst=True)

    return pd.to_datetime(text, errors="coerce")


def normalize_dividend_by_country(value, country):
    if value is None or pd.isna(value):
        return None

    rule = DIVIDEND_RULES.get(country.upper(), {"type": "normal"})
    if rule["type"] == "cents":
        return value * rule.get("factor", 1.0)
    return value


def detect_outliers(series):
    series = series.dropna()
    if series.empty:
        return series

    median = series.median()
    if pd.isna(median) or median <= 0:
        return series.iloc[0:0]

    return series[series > (median * 50)]


def parse_company_header(header):
    match = re.match(r"^(?P<company>.+?)\s+\((?P<country>.+)\)$", str(header).strip())
    if not match:
        return None, None
    return match.group("company").strip(), match.group("country").strip()


def build_company_lookup(isin_map):
    lookup = {}
    for _, row in isin_map.iterrows():
        full_name = str(row["Company"]).strip()
        company, country = parse_company_header(full_name)
        if company and country:
            lookup[full_name.upper()] = (company, country)
    return lookup


def convert_master_dividends(dividend_source=DIVIDEND_SOURCE, isin_source=ISIN_SOURCE, output_dir=OUTPUT_DIR):
    div_raw = pd.read_csv(dividend_source)
    isin_map = pd.read_csv(isin_source)
    company_lookup = build_company_lookup(isin_map)
    outlier_records = []

    records_written = 0
    companies_written = 0

    for col_idx in range(0, len(div_raw.columns), 2):
        if col_idx + 1 >= len(div_raw.columns):
            continue

        date_col = div_raw.columns[col_idx]
        dividend_col = div_raw.columns[col_idx + 1]
        lookup_key = str(dividend_col).strip().upper()

        company_info = company_lookup.get(lookup_key)
        if company_info is None:
            company, country = parse_company_header(dividend_col)
            if not company or not country:
                continue
        else:
            company, country = company_info

        pair_df = div_raw.iloc[1:, [col_idx, col_idx + 1]].copy()
        pair_df.columns = ["Date", "Dividend"]
        pair_df["Date"] = pair_df["Date"].apply(parse_dividend_date)
        pair_df["Dividend"] = pair_df["Dividend"].apply(clean_dividend_value)
        pair_df["Dividend"] = pair_df["Dividend"].apply(
            lambda x: normalize_dividend_by_country(x, country)
        )
        pair_df = pair_df.dropna(subset=["Date", "Dividend"])
        pair_df = pair_df.sort_values("Date").drop_duplicates(subset=["Date"], keep="last")

        if pair_df.empty:
            continue

        outliers = detect_outliers(pair_df["Dividend"])
        if not outliers.empty:
            median_dividend = pair_df["Dividend"].median()
            for idx, value in outliers.items():
                outlier_records.append(
                    {
                        "Company": company.upper(),
                        "Country": country.upper(),
                        "Date": pair_df.loc[idx, "Date"],
                        "Dividend": value,
                        "MedianDividend": median_dividend,
                        "OutlierMultiple": value / median_dividend if median_dividend else None,
                    }
                )

        country_dir = output_dir / slugify_country(country)
        country_dir.mkdir(parents=True, exist_ok=True)
        output_file = country_dir / f"{company.upper()}.csv"
        pair_df.to_csv(output_file, index=False)

        companies_written += 1
        records_written += len(pair_df)

    outlier_df = pd.DataFrame(outlier_records)
    outlier_df.to_csv(DIVIDEND_OUTLIER_FILE, index=False)

    return companies_written, records_written


def main():
    companies_written, records_written = convert_master_dividends()
    print(f"Companies written: {companies_written}")
    print(f"Dividend records written: {records_written}")


if __name__ == "__main__":
    main()
