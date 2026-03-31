import pandas as pd

DAILY_RETURN_CAP = 0.20


def sanitize_company_frame(df):
    cleaned = df.copy()
    cleaned["Date"] = pd.to_datetime(cleaned["Date"], errors="coerce")
    cleaned["Price"] = pd.to_numeric(cleaned["Price"], errors="coerce")

    if "Volume" in cleaned.columns:
        cleaned["Volume"] = pd.to_numeric(cleaned["Volume"], errors="coerce")

    cleaned = cleaned.dropna(subset=["Date"]).sort_values("Date")
    return cleaned


def validate_company_frame(df, company, return_cap=DAILY_RETURN_CAP):
    cleaned = sanitize_company_frame(df)
    price = cleaned["Price"]
    returns = price.pct_change()
    outlier_mask = returns.abs() > return_cap

    volume = cleaned["Volume"] if "Volume" in cleaned.columns else pd.Series(dtype=float)

    summary = {
        "Company": company,
        "Rows": int(len(cleaned)),
        "MissingPriceCount": int(price.isna().sum()),
        "MissingVolumeCount": int(volume.isna().sum()) if len(volume) else 0,
        "MissingPriceRatio": float(price.isna().mean()) if len(price) else 0.0,
        "MissingVolumeRatio": float(volume.isna().mean()) if len(volume) else 0.0,
        "ZeroPriceCount": int(price.fillna(0).eq(0).sum()),
        "OutlierReturnCount": int(outlier_mask.fillna(False).sum()),
        "MaxAbsDailyReturn": float(returns.abs().max()) if returns.notna().any() else 0.0,
        "HasMissingDataFlag": bool(price.isna().any() or (len(volume) and volume.isna().any())),
        "HasOutlierReturnFlag": bool(outlier_mask.fillna(False).any()),
    }

    return cleaned, summary


def cap_return_series(return_series, return_cap=DAILY_RETURN_CAP):
    capped = return_series.clip(lower=-return_cap, upper=return_cap)
    outlier_count = int((return_series.abs() > return_cap).fillna(False).sum())
    return capped, outlier_count


def build_master_validation_report(prices_df, return_cap=DAILY_RETURN_CAP):
    records = []

    for col in prices_df.columns:
        if not col.endswith(" Price"):
            continue

        company = col.replace(" Price", "")
        series = pd.to_numeric(prices_df[col], errors="coerce")
        returns = series.pct_change()
        outlier_mask = returns.abs() > return_cap

        records.append(
            {
                "Company": company,
                "MissingPriceCount": int(series.isna().sum()),
                "MissingPriceRatio": float(series.isna().mean()),
                "ZeroPriceCount": int(series.fillna(0).eq(0).sum()),
                "OutlierReturnCount": int(outlier_mask.fillna(False).sum()),
                "MaxAbsDailyReturn": float(returns.abs().max()) if returns.notna().any() else 0.0,
                "HasMissingDataFlag": bool(series.isna().any()),
                "HasOutlierReturnFlag": bool(outlier_mask.fillna(False).any()),
            }
        )

    return pd.DataFrame(records)
