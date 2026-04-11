import os

import pandas as pd


FX_BASE_PATH = "data/fx"
CLEAN_PRICE_PATH = "data/prices"
USD_PRICE_PATH = "data/prices_usd"
CURRENCY_MAP = {
    "EGYPT": "EGP",
    "GHANA": "GHS",
    "KENYA": "KES",
    "MALAWI": "MWK",
    "MAURITIUS": "MUR",
    "MOROCCO": "MAD",
    "NAMIBIA": "NAD",
    "NIGER": "XOF",
    "NIGERIA": "NGN",
    "RWANDA": "RWF",
    "SENEGAL": "XOF",
    "SOUTH AFRICA": "ZAR",
    "TANZANIA": "TZS",
    "TOGO": "XOF",
    "TUNISIA": "TND",
    "UGANDA": "UGX",
    "ZAMBIA": "ZMW",
    "ZIMBABWE": "ZWG",
}


def load_fx(currency_code):
    fx = pd.read_csv(
        f"{FX_BASE_PATH}/USD_{currency_code}.csv",
        parse_dates=["Date"],
    )
    fx["Date"] = pd.to_datetime(fx["Date"], errors="coerce").dt.normalize()
    fx = fx.dropna(subset=["Date"])
    fx = fx.sort_values("Date")
    fx.set_index("Date", inplace=True)
    return fx["FX"]


def convert_local_to_usd(price_series, currency_code):
    fx_series = load_fx(currency_code)
    aligned_fx = fx_series.reindex(price_series.index).ffill().bfill()
    usd_price = price_series / aligned_fx
    return usd_price


def get_currency_code(country):
    return CURRENCY_MAP.get(country)


def convert_to_usd(price_df, fx_series):
    df = price_df.copy()
    df.index = pd.to_datetime(df.index, errors="coerce").normalize()
    df = df[~df.index.isna()]
    df = df.sort_index()
    fx_series = fx_series.copy()
    fx_series.index = pd.to_datetime(fx_series.index, errors="coerce").normalize()
    fx_series = fx_series[~fx_series.index.isna()]
    fx_series = fx_series.sort_index()
    df = df.join(fx_series.rename("FX"), how="left")
    df["FX"] = df["FX"].ffill()
    df["Price_USD"] = df["Price"] / df["FX"]
    return df[["Price_USD"]]


def process_all():
    os.makedirs(USD_PRICE_PATH, exist_ok=True)

    for file in os.listdir(CLEAN_PRICE_PATH):
        if not file.endswith(".csv"):
            continue

        company = file.replace(".csv", "")
        country = company.split("(")[-1].replace(")", "").strip()
        currency = CURRENCY_MAP.get(country)

        if currency is None:
            print(f"⚠️ No currency mapping for {company} — skipping")
            continue

        price_df = pd.read_csv(
            os.path.join(CLEAN_PRICE_PATH, file),
            parse_dates=["Date"],
        )
        price_df.set_index("Date", inplace=True)
        price_df.index = pd.to_datetime(price_df.index, errors="coerce").normalize()
        price_df = price_df[~price_df.index.isna()]
        price_df = price_df.sort_index()

        fx_series = load_fx(currency)
        usd_df = convert_to_usd(price_df, fx_series)
        usd_df.to_csv(os.path.join(USD_PRICE_PATH, file))


if __name__ == "__main__":
    process_all()
