from pathlib import Path

import pandas as pd


DIVIDEND_BASE_DIR = Path("data/dividends")


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
    df = df.sort_values("Date").reset_index(drop=True)

    return df


def load_dividends(base_dir=DIVIDEND_BASE_DIR):
    base_dir = Path(base_dir)
    dividend_data = {}

    if not base_dir.exists():
        return dividend_data

    for country_dir in sorted(p for p in base_dir.iterdir() if p.is_dir()):
        country = country_dir.name.upper()

        for filepath in sorted(country_dir.glob("*.csv")):
            company = filepath.stem.upper().strip()
            company = " ".join(company.split())
            full_name = f"{company} ({country})"

            df = load_dividend_file(filepath)
            dividend_data[full_name] = df.set_index("Date")["Dividend"].sort_index()

    return dividend_data


def main():
    dividend_data = load_dividends()
    print(f"Companies: {len(dividend_data)}")
    print(f"Dividend records: {sum(len(series) for series in dividend_data.values())}")
    countries = {name.rsplit('(', 1)[-1].rstrip(')') for name in dividend_data}
    print(f"Countries: {len(countries)}")


if __name__ == "__main__":
    main()
