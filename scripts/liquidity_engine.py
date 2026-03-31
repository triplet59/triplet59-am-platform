import numpy as np


COUNTRY_SCALING = {
    "SOUTH AFRICA": 1,
    "NIGERIA": 1,
    "KENYA": 1,
    "EGYPT": 1,
    "IVORY COAST": 1,
    "SENEGAL": 1,
    "TANZANIA": 0.001,
    "UGANDA": 0.001,
    "MALAWI": 0.0001,
    "ZAMBIA": 0.001,
    "RWANDA": 0.001,
}


def compute_liquidity(df, company_name):
    country = company_name.split("(")[-1].replace(")", "").strip()

    scale = COUNTRY_SCALING.get(country, 1)

    df = df.copy()
    df["AdjValue"] = df["Price"] * df["Volume"] * scale

    rolling_value = df["AdjValue"].rolling(30).mean()
    participation = df["Volume"] / df["Volume"].rolling(30).mean()

    liquidity = rolling_value * (participation ** 2)

    return liquidity
