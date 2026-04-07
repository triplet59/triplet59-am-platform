from __future__ import annotations

import numpy as np
import pandas as pd

PRICE_SPIKE_MULTIPLE_CAP = 50.0
TV_QUANTILE_CAP = 0.995
STALE_VOLUME_RUN_LENGTH = 10


def _stale_run_mask(volume: pd.Series) -> pd.Series:
    series = pd.to_numeric(volume, errors="coerce")
    same_as_prev = series.eq(series.shift(1)) & series.gt(0)
    run_id = same_as_prev.ne(same_as_prev.shift(fill_value=False)).cumsum()
    run_length = same_as_prev.groupby(run_id).transform("sum")
    return same_as_prev & run_length.ge(STALE_VOLUME_RUN_LENGTH - 1)


def sanitize_volume_series(volume: pd.Series) -> tuple[pd.Series, pd.Series]:
    clean = pd.to_numeric(volume, errors="coerce")
    clean = clean.where(clean.gt(0))
    stale_mask = _stale_run_mask(clean)
    clean = clean.mask(stale_mask)
    return clean, stale_mask.fillna(False)


def sanitize_traded_value_series(
    price: pd.Series,
    volume: pd.Series,
) -> tuple[pd.Series, dict]:
    price_series = pd.to_numeric(price, errors="coerce").where(lambda s: s.gt(0))
    volume_series, stale_mask = sanitize_volume_series(volume)
    traded_value = price_series * volume_series
    positive = traded_value[traded_value.gt(0)]

    cap = np.nan
    clipped_mask = pd.Series(False, index=traded_value.index)
    if not positive.empty:
        median_tv = positive.median()
        quantile_cap = positive.quantile(TV_QUANTILE_CAP)
        if pd.notna(median_tv) and median_tv > 0:
            cap = min(quantile_cap, median_tv * PRICE_SPIKE_MULTIPLE_CAP)
        else:
            cap = quantile_cap

        if pd.notna(cap):
            clipped_mask = traded_value.gt(cap)
            traded_value = traded_value.clip(upper=cap)

    diagnostics = {
        "zero_or_missing_volume_days": int(volume_series.isna().sum()),
        "stale_volume_days": int(stale_mask.sum()),
        "clipped_value_days": int(clipped_mask.sum()),
        "traded_value_cap_usd": float(cap) if pd.notna(cap) else np.nan,
        "median_traded_value_usd": float(positive.median()) if not positive.empty else np.nan,
        "max_traded_value_usd": float(positive.max()) if not positive.empty else np.nan,
    }
    return traded_value, diagnostics


def compute_liquidity(price: pd.Series, volume: pd.Series, window: int = 30):
    traded_value, diagnostics = sanitize_traded_value_series(price, volume)
    valid = traded_value.notna()

    rolling_value = traded_value[valid].rolling(window).mean().reindex(traded_value.index).ffill()
    trade_count = valid.rolling(window).sum()
    participation = trade_count / window
    liquidity = rolling_value * (participation ** 2)

    return (
        liquidity.ffill(),
        participation.ffill(),
        rolling_value.ffill(),
        trade_count.ffill(),
        diagnostics,
    )
