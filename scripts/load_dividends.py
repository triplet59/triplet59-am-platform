import re
import unicodedata
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DIVIDEND_BASE_DIR = PROJECT_ROOT / "data" / "dividends"
MASTER_FILE = PROJECT_ROOT / "output" / "master.xlsx"
DIVIDEND_NAME_ALIASES = {
    "ABU QIR FERTILIZERS AND CHEMICAL INDUSTRIES (EGYPT)": "ABU QIR FERTILIZERS EGYPT (EGYPT)",
    "BID CORPORATION LTD (SOUTH AFRICA)": "BID CORPORATION (SOUTH AFRICA)",
    "CAPRICORN GROUP LIMITED (NAMIBIA)": "CAPRICORN INVESTMENT (NAMIBIA)",
    "CEC AFRICA INVESTMENT ZAMBIA (ZAMBIA)": "CEC AFRICA INVESTMENTS ZAMBIA (ZAMBIA)",
    "COPPERBELT ENERGY CORPORATION ZAMBIA (ZAMBIA)": "COPPERBELT ENERGY CORP ZAMBIA (ZAMBIA)",
    "HF GROUP PLC (KENYA)": "HF GROUP (KENYA)",
    "INTRAVENOUS INFUSIONS PLC (GHANA)": "INTRAVENOUS INFUSIONS (GHANA)",
    "JUHAYNA FOOD INDUSTRIES S.A.E (EGYPT)": "JUHAYNA FOOD INDUSTRIES (EGYPT)",
    "REDEFINE PROPERTIES LTD (SOUTH AFRICA)": "REDEFINE PROPERTIES (SOUTH AFRICA)",
    "STANDARD CHARTERED BANK ZAMBIA PLC (ZAMBIA)": "STANDARD CHARTERED BANK ZAMBIA (ZAMBIA)",
    "TRANSPACO LTD (SOUTH AFRICA)": "TRANSPACO (SOUTH AFRICA)",
}

SUFFIX_PATTERNS = (
    " LIMITED",
    " LTD",
    " PLC",
    " PLC.",
    " S.A.E",
    " S.A.",
    " SOCIETE ANONYME",
    " COMPANY",
)
TOKEN_NORMALIZATIONS = {
    " CORPORATION ": " CORP ",
    " HOLDINGS ": " HOLDING ",
    " INVESTMENTS ": " INVESTMENT ",
    " BREWERIES ": " BREWERY ",
    " TELECOMMUNICATIONS ": " TELECOMS ",
}


def _ascii_upper(text):
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    return " ".join(text.upper().split())


def _extract_country(name):
    match = re.search(r"\(([^()]+)\)\s*$", name)
    return match.group(1).upper() if match else ""


def _normalize_company_key(name):
    text = _ascii_upper(name)
    for raw, clean in TOKEN_NORMALIZATIONS.items():
        text = text.replace(raw, clean)
    for suffix in SUFFIX_PATTERNS:
        if text.endswith(suffix):
            text = text[: -len(suffix)]
    text = re.sub(r"\(([^()]+)\)\s+\(\1\)$", r"(\1)", text)
    country = _extract_country(text)
    if country:
        country_token = f" {country}"
        text = text.replace(country_token, " ")
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    return " ".join(text.split())


def load_master_company_universe(master_file=MASTER_FILE):
    master_file = Path(master_file)
    if not master_file.exists():
        return set()

    master = pd.read_excel(master_file, nrows=1)
    return {col[:-6] for col in master.columns if col.endswith(" Price")}


def resolve_dividend_name(full_name, master_companies):
    aliased = DIVIDEND_NAME_ALIASES.get(full_name, full_name)
    if aliased in master_companies or not master_companies:
        return aliased

    target_country = _extract_country(aliased)
    normalized = _normalize_company_key(aliased)
    candidates = []
    for company in master_companies:
        if target_country and _extract_country(company) != target_country:
            continue
        if _normalize_company_key(company) == normalized:
            candidates.append(company)

    if len(candidates) == 1:
        return candidates[0]

    return aliased


def load_dividend_file(filepath):
    df = pd.read_csv(filepath)

    required_cols = {"Date", "Dividend"}
    missing = required_cols - set(df.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns: {missing_text}")

    df = df[["Date", "Dividend"]].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Dividend"] = pd.to_numeric(df["Dividend"], errors="coerce")

    df = df.dropna(subset=["Date", "Dividend"])
    # Some vendor files can repeat the same ex-date; consolidate them before indexing.
    df = df.groupby("Date", as_index=False, sort=True)["Dividend"].sum()
    df = df.sort_values("Date").reset_index(drop=True)

    return df


def load_dividends(base_dir=DIVIDEND_BASE_DIR):
    base_dir = Path(base_dir)
    dividend_data = {}
    master_companies = load_master_company_universe()

    if not base_dir.exists():
        return dividend_data

    for country_dir in sorted(p for p in base_dir.iterdir() if p.is_dir()):
        country = country_dir.name.replace("_", " ").upper()

        for filepath in sorted(country_dir.glob("*.csv")):
            company = filepath.stem.upper().strip()
            company = " ".join(company.split())
            full_name = f"{company} ({country})"
            full_name = resolve_dividend_name(full_name, master_companies)

            df = load_dividend_file(filepath)
            dividend_data[full_name] = df.set_index("Date")["Dividend"].sort_index()

    return dividend_data


def build_dividend_matrix(dividends_dict, index):
    if not dividends_dict:
        return pd.DataFrame(index=index)

    matrix = {
        ticker: series.reindex(index)
        for ticker, series in dividends_dict.items()
    }

    return pd.DataFrame(matrix, index=index).fillna(0)


def main():
    dividend_data = load_dividends()
    print(f"Companies: {len(dividend_data)}")
    print(f"Dividend records: {sum(len(series) for series in dividend_data.values())}")
    countries = {name.rsplit('(', 1)[-1].rstrip(')') for name in dividend_data}
    print(f"Countries: {len(countries)}")


if __name__ == "__main__":
    main()
