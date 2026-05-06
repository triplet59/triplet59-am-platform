import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate
from scripts.data_loader import INDEX_POOL, load_benchmark, load_index
from scripts.portfolio_allocator import (
    MODEL_METADATA,
    build_index,
    build_model_metrics_table,
    build_model_weight_system,
    build_portfolio,
    build_model_series,
    metrics as portfolio_metrics,
    optimize_high_risk_portfolio,
    prepare_returns_frame,
    simulate_random_frontier,
)

def help_text(text):
    return f"<span title='{text}' style='cursor: help;'>ⓘ</span>"


def tooltip(label, text):
    return (
        f"<span style=\"border-bottom:1px dotted #999;\" title=\"{text}\">"
        f"{label}"
        f"</span>"
    )


def render_metric_card(label, value, help_copy=None):
    title_attr = f" title=\"{help_copy}\"" if help_copy else ""
    st.markdown(
        f"""
        <div class="kpi-card"{title_attr}>
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def configure_app():
    st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
    # Temporary deployment cache-buster for Streamlit Cloud refreshes.
    st.cache_data.clear()


def get_selected_series():
    return st.session_state.get("global_series_selector", "AM Series")


def render_static_header():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    logo_path = os.path.join(base_dir, "assets", "veri_logo.png")
    col1, col2, col3 = st.columns([1, 6, 1])

    with col1:
        if os.path.exists(logo_path):
            st.image(logo_path, width=120)

    with col2:
        st.markdown(
            """
            ### Veri African Indices
            AM100 / AM200 / AM300 Total Return Indices
            """,
        )
        st.markdown(
            """
            <div style="font-size:15px; font-weight:500; margin-bottom:6px;">
            AM100 is the institutional core of the Veri African Indices suite.
            </div>

            <div style="font-size:13px; color:#c9c9c9; max-width:900px;">
            The framework applies strict USD liquidity, trading consistency, and total return standards to define the
            investable African equity universe. AM200 and AM300 extend this framework through controlled expansion
            of the investable universe, while preserving liquidity integrity and execution realism.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        if st.button("Logout", key="global_logout"):
            st.session_state.auth_mode = None
            st.rerun()

    st.markdown("<hr style='margin: 16px 0; opacity: 0.2;'>", unsafe_allow_html=True)


def render_data_header(validated, eligible):
    st.markdown(
        f"""
        <div style="
            background-color:#111827;
            padding:14px 18px;
            border-radius:8px;
            margin-top:14px;
            margin-bottom:10px;
        ">
            <div style="font-size:12px; color:#9aa0a6;">
                Market Coverage
            </div>
            <div style="font-size:20px; font-weight:600;">
                {validated} Validated Equities | {eligible} Eligible (&ge;80% Trading Coverage)
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def safe_display(value, fallback="N/A"):
    return value if value is not None else fallback


def safe_metric(result):
    if isinstance(result, dict):
        value = result.get("cagr")
    else:
        value = result
    return f"{value:.2%}" if value is not None and pd.notnull(value) else "N/A"


def safe_period_range(period):
    if not period:
        return "N/A"

    try:
        # Case 1: dict payload (current structure)
        if isinstance(period, dict):
            start = period.get("start")
            end = period.get("end")
            if start is None or end is None:
                return "N/A"

        # Case 2: tuple payload (legacy support)
        elif isinstance(period, tuple) and len(period) == 2:
            start, end = period
        else:
            return "N/A"

        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
        return f"{start.strftime('%d %b %Y')} -> {end.strftime('%d %b %Y')}"
    except Exception:
        return "N/A"


def validate_df(df, name, required_columns=None):
    if df is None:
        raise ValueError(f"{name} is None")
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"{name} is not a DataFrame")
    if df.empty:
        raise ValueError(f"{name} is empty")
    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"{name} is missing required columns: {', '.join(missing)}")
    return df


def validate_series(series, name):
    if series is None:
        raise ValueError(f"{name} is None")
    if not isinstance(series, pd.Series):
        raise ValueError(f"{name} is not a Series")
    if series.empty:
        raise ValueError(f"{name} is empty")
    return series.sort_index()


def standardize_index_level_df(df, index_name):
    frame = validate_df(df.copy(), f"{index_name} performance_df", ["Date", "Index Level"])
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame["Index_Level"] = pd.to_numeric(frame["Index Level"], errors="coerce")
    frame = frame[["Date", "Index_Level"]].dropna().copy()
    frame["Index"] = index_name
    return validate_df(
        frame,
        f"{index_name} performance_df standardized",
        ["Date", "Index", "Index_Level"],
    )


def series_metric_frame(series, index_name, metric_name):
    frame = pd.DataFrame({"Date": series.index, "Value": pd.to_numeric(series, errors="coerce")})
    frame["Index"] = index_name
    frame["Metric"] = metric_name
    frame = frame.dropna(subset=["Date", "Value"])
    return validate_df(frame, f"{index_name} {metric_name}", ["Date", "Index", "Metric", "Value"])


def build_dashboard_outputs(
    am100_df,
    am200_df,
    am300_df,
    rolling_perf_1y_100,
    rolling_perf_1y_200,
    rolling_perf_1y_300,
    rolling_vol_100,
    rolling_vol_200,
    rolling_vol_300,
    drawdown_100,
    drawdown_200,
    drawdown_300,
    am100_latest,
    am200_latest,
    am300_latest,
    stats_dict,
):
    performance_df = pd.concat(
        [
            standardize_index_level_df(am100_df, "AM100"),
            standardize_index_level_df(am200_df, "AM200"),
            standardize_index_level_df(am300_df, "AM300"),
        ],
        ignore_index=True,
    )
    rolling_df = pd.concat(
        [
            series_metric_frame(rolling_perf_1y_100, "AM100", "Rolling_1Y"),
            series_metric_frame(rolling_perf_1y_200, "AM200", "Rolling_1Y"),
            series_metric_frame(rolling_perf_1y_300, "AM300", "Rolling_1Y"),
            series_metric_frame(rolling_vol_100, "AM100", "Rolling_30D_Volatility"),
            series_metric_frame(rolling_vol_200, "AM200", "Rolling_30D_Volatility"),
            series_metric_frame(rolling_vol_300, "AM300", "Rolling_30D_Volatility"),
        ],
        ignore_index=True,
    )
    drawdown_df = pd.concat(
        [
            series_metric_frame(drawdown_100, "AM100", "Drawdown"),
            series_metric_frame(drawdown_200, "AM200", "Drawdown"),
            series_metric_frame(drawdown_300, "AM300", "Drawdown"),
        ],
        ignore_index=True,
    )
    constituents_df = pd.concat(
        [
            validate_df(am100_latest.copy(), "AM100 constituents", ["Company", "Country", "Weight"]).assign(Index="AM100"),
            validate_df(am200_latest.copy(), "AM200 constituents", ["Company", "Country", "Weight"]).assign(Index="AM200"),
            validate_df(am300_latest.copy(), "AM300 constituents", ["Company", "Country", "Weight"]).assign(Index="AM300"),
        ],
        ignore_index=True,
    )
    return {
        "performance": validate_df(
            performance_df, "performance_df", ["Date", "Index", "Index_Level"]
        ),
        "rolling": validate_df(
            rolling_df, "rolling_df", ["Date", "Index", "Metric", "Value"]
        ),
        "drawdown": validate_df(
            drawdown_df, "drawdown_df", ["Date", "Index", "Metric", "Value"]
        ),
        "constituents": validate_df(
            constituents_df, "constituents_df", ["Index", "Company", "Country", "Weight"]
        ),
        "stats": stats_dict,
    }


def validate_final_weights(df, label="Index"):
    if df is None or df.empty or "Weight" not in df.columns:
        return
    total_weight = float(df["Weight"].fillna(0).sum())
    if abs(total_weight - 1.0) >= 0.001:
        st.warning(f"{label} weights sum to {total_weight:.4f}, not 1.0000.")
    ordered = df.sort_values(["Weight", "Company"], ascending=[False, True])["Weight"]
    if not ordered.is_monotonic_decreasing:
        st.warning(f"{label} weight ordering inconsistency detected.")


def get_top_country(df):
    if df is None or df.empty or "Country" not in df.columns or "Weight" not in df.columns:
        return None
    grouped = df.groupby("Country")["Weight"].sum()
    return grouped.idxmax() if not grouped.empty else None


def get_country_weights(df):
    if df is None or df.empty or "Country" not in df.columns or "Weight" not in df.columns:
        return pd.Series(dtype=float)
    return df.groupby("Country")["Weight"].sum().sort_values(ascending=False)


def compute_country_weights(df):
    if df is None or df.empty or "Country" not in df.columns or "Weight" not in df.columns:
        return pd.DataFrame(columns=["Country", "Weight"])

    return (
        df.groupby("Country")["Weight"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )


def external_capacity_metrics(snapshot):
    if "AvgDailyValue30dUSD" in snapshot.columns:
        adv_usd = float(snapshot["AvgDailyValue30dUSD"].fillna(snapshot["AvgDailyValue30d"]).sum())
    else:
        adv_usd = float(snapshot["AvgDailyValue30d"].sum())

    if "InvestableCapacity20USD" in snapshot.columns:
        investable_usd = float(
            snapshot["InvestableCapacity20USD"]
            .fillna(snapshot.get("AvgDailyValue30dUSD", pd.Series(index=snapshot.index, dtype=float)) * 0.20)
            .fillna(snapshot["AvgDailyValue30d"] * 0.20)
            .sum()
        )
    else:
        investable_usd = adv_usd * 0.20

    return adv_usd, investable_usd


def audit_volume_quality(df):
    results = []
    for col in df.columns:
        if not str(col).endswith(" Volume"):
            continue
        security = str(col).replace(" Volume", "")
        price_col = f"{security} Price USD" if f"{security} Price USD" in df.columns else f"{security} Price"
        series = df[col]
        if price_col in df.columns:
            observed_mask = pd.to_numeric(df[price_col], errors="coerce").gt(0)
        else:
            observed_mask = pd.Series(True, index=df.index)
        observed_series = pd.to_numeric(series.where(observed_mask), errors="coerce")
        total_days = int(observed_mask.sum())
        missing_days = int(observed_series.isna().sum())
        zero_days = int(observed_series.fillna(0).eq(0).sum() - observed_series.isna().sum())
        positive_days = int(observed_series.gt(0).sum())
        positive_coverage = positive_days / total_days if total_days else 0
        results.append(
            {
                "Security": security,
                "Total Days": total_days,
                "Missing Days": missing_days,
                "Zero Days": zero_days,
                "Positive Days": positive_days,
                "Positive Coverage %": round(positive_coverage * 100, 2),
            }
        )

    if not results:
        return pd.DataFrame(
            columns=[
                "Security",
                "Total Days",
                "Missing Days",
                "Zero Days",
                "Positive Days",
                "Positive Coverage %",
            ]
        )

    return pd.DataFrame(results).sort_values(
        ["Positive Coverage %", "Positive Days", "Security"], ascending=[True, True, True]
    ).reset_index(drop=True)


def get_volume_audit(df):
    audit = audit_volume_quality(df)
    validated = len(audit)
    eligible = int((audit["Positive Coverage %"] >= 80).sum())

    assert validated > 0, "No validated securities"
    assert eligible > 0, "No eligible securities"

    return audit, validated, eligible


def prepare_constituent_table(df, include_rank=False, limit=None, search=None):
    if df is None or df.empty:
        cols = ["Company", "Country", "Weight"]
        if include_rank:
            cols.append("Rank")
        return pd.DataFrame(columns=cols)

    validate_final_weights(df)
    display_df = df.copy()
    display_df = display_df.sort_values(["Weight", "Company"], ascending=[False, True])
    display_df["Rank"] = np.arange(1, len(display_df) + 1)

    if search:
        display_df = display_df[
            display_df["Company"].astype(str).str.contains(search, case=False, na=False)
        ]

    cols = ["Company", "Country", "Weight"]
    if include_rank:
        cols.append("Rank")
    if limit is not None:
        display_df = display_df.head(limit)
    display_df = display_df[cols].copy()
    display_df["Weight"] = display_df["Weight"].map("{:.2%}".format)
    return display_df.reset_index(drop=True)


def calculate_cagr(df, start_date, end_date=None):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    if end_date is None:
        end_date = df["Date"].iloc[-1]

    df_period = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]

    if len(df_period) < 2:
        return None

    start_val = df_period["Index Level"].iloc[0]
    end_val = df_period["Index Level"].iloc[-1]

    years = (df_period["Date"].iloc[-1] - df_period["Date"].iloc[0]).days / 365.25

    return (end_val / start_val) ** (1 / years) - 1


def calculate_cagr_series(series):
    series = series.dropna().sort_index()
    if len(series) < 2:
        return None

    start = series.iloc[0]
    end = series.iloc[-1]
    years = (series.index[-1] - series.index[0]).days / 365.25
    if years <= 0:
        return None
    return (end / start) ** (1 / years) - 1


def get_rolling_window(index, years, tolerance_days=30):
    end_date = index.max()
    target_start = end_date - pd.DateOffset(years=years)

    candidates = index[
        (index >= target_start - pd.Timedelta(days=tolerance_days))
        & (index <= target_start + pd.Timedelta(days=tolerance_days))
    ]

    if len(candidates) == 0:
        return None, None, "INSUFFICIENT_DATA"

    start_date = candidates.min()
    actual_days = (end_date - start_date).days

    if actual_days < (365 * years - tolerance_days):
        return None, None, "WINDOW_TOO_SHORT"

    return start_date, end_date, "OK"


def get_fixed_window(index, start_date, end_date):
    available = index[(index >= start_date) & (index <= end_date)]

    if len(available) == 0:
        return None, None, "NO_DATA"

    return available.min(), available.max(), "OK"


def calculate_cagr_window(series, start_date, end_date):
    window = series.loc[start_date:end_date].dropna()
    if len(window) < 2:
        return None

    start_val = window.iloc[0]
    end_val = window.iloc[-1]
    days = (window.index[-1] - window.index[0]).days
    years = days / 365.25

    if years <= 0 or start_val == 0 or pd.isna(start_val) or pd.isna(end_val):
        return None

    return (end_val / start_val) ** (1 / years) - 1


def compute_performance(series, window_type, name=None, **kwargs):
    if series is None:
        raise ValueError(f"{name or 'compute_performance'}: series is None")

    if isinstance(series, pd.DataFrame):
        if series.shape[1] == 1:
            series = series.iloc[:, 0]
        else:
            raise ValueError(
                f"{name or 'compute_performance'}: expected 1 column, got {series.shape[1]}"
            )

    if not isinstance(series, pd.Series):
        raise TypeError(
            f"{name or 'compute_performance'}: expected pd.Series, got {type(series)}"
        )

    if not isinstance(series.index, pd.DatetimeIndex):
        try:
            series.index = pd.to_datetime(series.index)
        except Exception as exc:
            raise ValueError(
                f"{name or 'compute_performance'}: index is not datetime-compatible"
            ) from exc

    series = series.dropna()
    if series.empty:
        return {
            "status": "NO_DATA",
            "cagr": None,
            "vol": None,
            "sharpe": None,
            "drawdown": None,
            "start": None,
            "end": None,
        }

    index = series.index.sort_values()
    if len(index) < 2:
        return {
            "status": "NO_DATA",
            "cagr": None,
            "vol": None,
            "sharpe": None,
            "drawdown": None,
            "start": index.min() if len(index) else None,
            "end": index.max() if len(index) else None,
        }

    if window_type == "rolling":
        start, end, status = get_rolling_window(
            index, kwargs["years"], kwargs.get("tolerance_days", 30)
        )
    elif window_type == "fixed":
        start, end, status = get_fixed_window(
            index, kwargs["start_date"], kwargs["end_date"]
        )
    else:
        return {"status": "INVALID_TYPE", "cagr": None, "start": None, "end": None}

    if status != "OK":
        return {"status": status, "cagr": None, "start": None, "end": None}

    cagr = calculate_cagr_window(series, start, end)
    if cagr is None:
        return {"status": "INVALID_SERIES", "cagr": None, "start": start, "end": end}

    return {"status": "OK", "cagr": cagr, "start": start, "end": end}


def compute_cagr_for_periods(index_series, periods):
    empty_result = {"status": "NO_DATA", "cagr": None, "start": None, "end": None}
    results = {
        "common": None,
        "fixed_10y": None,
        "fixed_5y": None,
        "since_2016": None,
        "rolling_10y": empty_result.copy(),
        "rolling_5y": empty_result.copy(),
        "rolling_3y": empty_result.copy(),
        "rolling_1y": empty_result.copy(),
    }

    for name, period in periods.items():
        if name == "latest_valid" or not isinstance(period, dict):
            continue
        if period.get("type") == "rolling":
            results[name] = compute_performance(
                index_series,
                "rolling",
                name=name,
                years=period["years"],
                tolerance_days=period.get("tolerance_days", 30),
            )
        else:
            results[name] = compute_performance(
                index_series,
                "fixed",
                name=name,
                start_date=period["start"],
                end_date=period["end"],
            )

    return results


def get_latest_valid_date(price_df, constituents):
    price_columns = [f"{company} Price" for company in constituents if f"{company} Price" in price_df.columns]
    if not price_columns:
        return None

    valid_rows = price_df[price_columns].dropna(how="any")
    if valid_rows.empty:
        return None

    return valid_rows.index.max()


def get_periods(price_df, constituents, index_series):
    start_2016 = pd.Timestamp("2016-01-01")
    end_2020 = pd.Timestamp("2020-12-31")
    end_2025 = pd.Timestamp("2025-12-31")
    latest_valid = get_latest_valid_date(price_df, constituents)
    available_index = index_series.dropna().index.sort_values()
    series_latest = available_index.max()
    latest = min(latest_valid, series_latest) if latest_valid is not None else series_latest

    if latest is None:
        latest = start_2016

    def resolve_start(requested_start, tolerance_days=30):
        lower_bound = requested_start - pd.Timedelta(days=tolerance_days)
        upper_bound = requested_start + pd.Timedelta(days=tolerance_days)
        candidate_dates = available_index[
            (available_index >= lower_bound) & (available_index <= upper_bound)
        ]
        if len(candidate_dates) == 0:
            return None
        deltas = (candidate_dates - requested_start).days
        closest_idx = np.abs(deltas).argmin()
        return candidate_dates[closest_idx]

    def rolling_window(years):
        requested_start = latest - pd.DateOffset(years=years)
        actual_start = resolve_start(requested_start)
        sufficient_history = False
        approximate = False
        if actual_start is not None:
            window_days = (latest - actual_start).days
            sufficient_history = window_days >= 365 * years - 30
            approximate = abs((actual_start - requested_start).days) > 7
        return {
            "start": actual_start if sufficient_history else None,
            "end": latest,
            "requested_start": requested_start,
            "years": years,
            "approximate": approximate,
            "sufficient_history": sufficient_history,
        }

    rolling_10y = rolling_window(10)
    rolling_5y = rolling_window(5)
    rolling_3y = rolling_window(3)
    rolling_1y = rolling_window(1)

    return {
        "rolling_10y": {"type": "rolling", "years": 10, "tolerance_days": 30, **rolling_10y},
        "rolling_5y": {"type": "rolling", "years": 5, "tolerance_days": 30, **rolling_5y},
        "rolling_3y": {"type": "rolling", "years": 3, "tolerance_days": 30, **rolling_3y},
        "rolling_1y": {"type": "rolling", "years": 1, "tolerance_days": 30, **rolling_1y},
        "fixed_10y": {"type": "fixed", "start": start_2016, "end": end_2025},
        "fixed_5y": {"type": "fixed", "start": pd.Timestamp("2021-01-01"), "end": end_2025},
        "since_2016": {"type": "fixed", "start": start_2016, "end": latest},
        "latest_valid": latest,
    }


def get_common_window(series_dict):
    valid_ranges = []

    for series in series_dict.values():
        if series is None:
            continue
        clean = series.dropna()
        if not clean.empty:
            valid_ranges.append((clean.index.min(), clean.index.max()))

    if not valid_ranges:
        return None, None

    start = max(r[0] for r in valid_ranges)
    end = min(r[1] for r in valid_ranges)

    if start >= end:
        return None, None

    return start, end


def get_common_analysis_window(index_map, requested_start=pd.Timestamp("2016-01-01")):
    valid_series = {
        name: series.dropna().sort_index()
        for name, series in index_map.items()
        if series is not None and len(series.dropna()) >= 2
    }
    if not valid_series:
        return {"start": None, "end": None, "label": "No common period", "series": {}}

    common_start, common_end = get_common_window(valid_series)
    if common_start is None or common_end is None:
        return {"start": None, "end": None, "label": "No common period", "series": {}}

    start = max(requested_start, common_start)
    if start >= common_end:
        return {"start": None, "end": None, "label": "No common period", "series": {}}

    aligned = {}
    for name, series in valid_series.items():
        aligned_series = series.loc[start:common_end].dropna()
        if len(aligned_series) >= 2:
            aligned[name] = aligned_series

    if not aligned:
        return {"start": None, "end": None, "label": "No common period", "series": {}}

    label = f"{start.strftime('%d %b %Y')} -> {common_end.strftime('%d %b %Y')}"
    return {"start": start, "end": common_end, "label": label, "series": aligned}


def compute_window_metrics(series):
    clean = series.dropna().sort_index()
    if len(clean) < 2:
        return {"CAGR": np.nan, "Volatility": np.nan, "Sharpe": np.nan, "Max Drawdown": np.nan}

    returns = clean.pct_change().dropna()
    if returns.empty:
        return {"CAGR": np.nan, "Volatility": np.nan, "Sharpe": np.nan, "Max Drawdown": np.nan}

    years = (clean.index[-1] - clean.index[0]).days / 365.25
    cagr = np.nan
    if years > 0 and pd.notna(clean.iloc[0]) and pd.notna(clean.iloc[-1]) and clean.iloc[0] != 0:
        cagr = (clean.iloc[-1] / clean.iloc[0]) ** (1 / years) - 1
    vol = returns.std() * np.sqrt(252)
    annual_return = returns.mean() * 252
    sharpe = (annual_return - 0.02) / vol if pd.notna(vol) and vol != 0 else np.nan
    max_dd = calculate_drawdown(clean).min()
    return {"CAGR": cagr, "Volatility": vol, "Sharpe": sharpe, "Max Drawdown": max_dd}


def calculate_drawdown(series):
    series = series.copy()
    peak = series.cummax()
    drawdown = (series / peak) - 1
    return drawdown


def rolling_volatility(returns):
    return returns.rolling(30).std() * np.sqrt(252)


def annual_vol(returns):
    return returns.std() * np.sqrt(252)


def sharpe_from_returns(returns, risk_free_rate=0.02):
    vol = annual_vol(returns)
    if vol == 0:
        return np.nan
    return (returns.mean() * 252 - risk_free_rate) / vol


def info_ratio(asset_returns, benchmark_returns):
    active = asset_returns.align(benchmark_returns, join="inner")
    active_returns = active[0] - active[1]
    tracking_error = annual_vol(active_returns)
    if tracking_error == 0:
        return np.nan
    return (active_returns.mean() * 252) / tracking_error


def advanced_risk_score(volatility, max_drawdown, alpha, info_ratio_value):
    vol_score = min(volatility / 0.30, 1)
    dd_score = min(abs(max_drawdown) / 0.60, 1)
    alpha_score = 1 - min(max(alpha, 0) / 0.10, 1)
    info_score = 1 - min(max(info_ratio_value, 0) / 1.0, 1) if not np.isnan(info_ratio_value) else 1
    return 0.35 * vol_score + 0.30 * dd_score + 0.20 * alpha_score + 0.15 * info_score


def risk_label(score):
    if score < 0.3:
        return "Low"
    if score < 0.6:
        return "Moderate"
    return "High"


def block(series, returns):
    return {
        "CAGR": calculate_cagr(series.rename("Index Level").reset_index(), series.index.min(), series.index.max()),
        "Vol": annual_vol(returns),
        "Sharpe": sharpe_from_returns(returns),
        "Drawdown": calculate_drawdown(series).min(),
    }


def prep(df):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    df["Index Level"] = 1000 * (df["Index Level"] / df["Index Level"].iloc[0])
    series = df.set_index("Date")["Index Level"]
    returns = series.pct_change().dropna()
    return series, returns


def get_file_version(path):
    if not os.path.exists(path):
        return None
    return os.path.getmtime(path)


@st.cache_data
def load_country_exposure(path, _version=None):
    df = pd.read_csv(path, parse_dates=["Date"])
    latest_date = df["Date"].max()
    latest = df[df["Date"] == latest_date].copy()
    country_name_map = {
        "NIGERIA": "Nigeria",
        "SOUTH AFRICA": "South Africa",
        "EGYPT": "Egypt",
        "KENYA": "Kenya",
        "NAMIBIA": "Namibia",
        "MOROCCO": "Morocco",
        "MAURITIUS": "Mauritius",
        "GHANA": "Ghana",
        "RWANDA": "Rwanda",
        "SENEGAL": "Senegal",
        "TOGO": "Togo",
        "TANZANIA": "Tanzania",
        "TUNISIA": "Tunisia",
        "UGANDA": "Uganda",
        "NIGER": "Niger",
        "ZIMBABWE": "Zimbabwe",
    }
    latest["Country"] = latest["Country"].map(country_name_map).fillna(
        latest["Country"].str.title()
    )
    return latest


@st.cache_data
def load_csv_file(path, _version=None):
    return pd.read_csv(path)


@st.cache_data
def load_annual_returns(path, _version=None):
    df = pd.read_csv(path)
    if "Year" in df.columns:
        df["Year"] = df["Year"].astype(int)
    return df


@st.cache_data
def load_daily_panel(path, _version=None):
    df = pd.read_csv(path, parse_dates=["Date"])
    return df.sort_values("Date")


@st.cache_data
def load_benchmark_series(path, _version=None):
    df = load_benchmark(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date")


@st.cache_data
def load_benchmark_data(_version=None):
    path = "output/msci_africa.csv"
    if not os.path.exists(path):
        return None
    df = load_benchmark(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date")


@st.cache_data
def load_metrics_csv(path, _version=None):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    return df.iloc[0].to_dict()


@st.cache_data
def load_constituent_snapshot_insights(path, _version=None):
    if not os.path.exists(path):
        return None

    df = pd.read_excel(path)
    if df.empty or "Date" not in df.columns or "Weight" not in df.columns:
        return None

    df["Date"] = pd.to_datetime(df["Date"])
    latest_date = df["Date"].max()
    snapshot = (
        df[df["Date"] == latest_date]
        .copy()
        .sort_values(["Weight", "Company"], ascending=[False, True])
        .reset_index(drop=True)
    )
    if snapshot.empty:
        return None

    turnover_values = []
    rebalance_dates = sorted(df["Date"].dropna().unique())
    filtered_dates = []
    prev_set = None
    for rebalance_date in rebalance_dates:
        current_set = set(df[df["Date"] == rebalance_date]["Company"].dropna())
        if prev_set is None or current_set != prev_set:
            filtered_dates.append(rebalance_date)
        prev_set = current_set

    prev = None
    for rebalance_date in filtered_dates:
        current = df[df["Date"] == rebalance_date].set_index("Company")["Weight"]
        if prev is not None:
            aligned = prev.rename("prev").to_frame().join(
                current.rename("cur"), how="outer"
            ).fillna(0.0)
            one_way_turnover = 0.5 * (aligned["cur"] - aligned["prev"]).abs().sum()
            turnover_values.append(float(one_way_turnover))
        prev = current

    country_weights = (
        snapshot.groupby("Country")["Weight"].sum().sort_values(ascending=False)
        if "Country" in snapshot.columns
        else pd.Series(dtype=float)
    )

    return {
        "latest_date": latest_date,
        "constituents": int(len(snapshot)),
        "top10_weight": float(snapshot["Weight"].head(10).sum()),
        "top_country": country_weights.index[0] if not country_weights.empty else None,
        "avg_turnover": float(np.mean(turnover_values)) if turnover_values else None,
    }


@st.cache_data
def load_index_overlap_metrics(am100_path, am200_path, am300_path, _version=None):
    paths = [am100_path, am200_path, am300_path]
    if not all(os.path.exists(path) for path in paths):
        return None

    am100 = pd.read_excel(am100_path)
    am200 = pd.read_excel(am200_path)
    am300 = pd.read_excel(am300_path)

    for df in (am100, am200, am300):
        if df.empty or "Date" not in df.columns or "Company" not in df.columns:
            return None
        df["Date"] = pd.to_datetime(df["Date"])

    latest_date = am100["Date"].max()
    s100 = set(am100[am100["Date"] == latest_date]["Company"])
    s200 = set(am200[am200["Date"] == latest_date]["Company"])
    s300 = set(am300[am300["Date"] == latest_date]["Company"])

    def overlap_pct(left, right):
        if not left:
            return None
        return len(left & right) / len(left)

    return {
        "AM100_in_AM200": overlap_pct(s100, s200),
        "AM100_in_AM300": overlap_pct(s100, s300),
        "AM200_in_AM300": overlap_pct(s200, s300),
    }


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(BASE_DIR, "..", "assets", "veri_logo.png")
AM100_COLOR = "#4DA3FF"
AM200_COLOR = "#FF9F1C"
AM300_COLOR = "#22C55E"
BUILD_VERSION = "21aa1d7"
SHOW_AM200 = True
SHOW_AM300 = False

INTERNAL_PASSWORD = st.secrets.get("internal_password", "internal123")
INVESTOR_PASSWORD = st.secrets.get("investor_password", "investor123")

if "auth_mode" not in st.session_state:
    st.session_state.auth_mode = None

if st.session_state.auth_mode is None:
    st.title("Veri African Indices")
    password = st.text_input("Enter Access Code", type="password")

    if password:
        if password == INTERNAL_PASSWORD:
            st.session_state.auth_mode = "internal"
            st.rerun()
        elif password == INVESTOR_PASSWORD:
            st.session_state.auth_mode = "investor"
            st.rerun()
        else:
            st.error("Invalid access code")

    st.stop()

MODE = st.session_state.auth_mode
IS_INVESTOR = MODE == "investor"
IS_INTERNAL = MODE == "internal"
allocator_mode = IS_INVESTOR

intro_text = """
**AM100 is a rules-based, institutional-grade equity index designed to reflect the investable African equity universe.**

### Key Principles:
- USD-based liquidity filtering (>= $1M ADV)
- Consistent trading activity (>= 90% of trading days)
- Verified total return methodology (price + dividends)
- Quarterly rebalancing with implementation lag

### What AM100 Represents:
AM100 reflects where institutional capital can realistically be deployed across African equity markets.

### What AM100 Does Not Do:
- It does not include illiquid or intermittently traded securities
- It does not artificially expand into frontier markets without sufficient liquidity
- It does not optimise for backtested performance

The result is a transparent, investable benchmark aligned with real-world constraints.
"""

am100_metrics = load_metrics_csv("output/AM100_metrics.csv", BUILD_VERSION)
am200_metrics = load_metrics_csv("output/AM200_metrics.csv", BUILD_VERSION)
am300_metrics = load_metrics_csv("output/AM300_metrics.csv", BUILD_VERSION)
annual_returns_df = load_annual_returns("output/AM_annual_returns.csv", BUILD_VERSION)
annual_returns_daily_df = load_daily_panel(
    "output/annual_calendar_returns_daily_panel.csv", BUILD_VERSION
)
am100_snapshot_insights = load_constituent_snapshot_insights(
    "output/AM100_history.xlsx", BUILD_VERSION
)
am200_snapshot_insights = load_constituent_snapshot_insights(
    "output/AM200_history.xlsx", BUILD_VERSION
)
am300_snapshot_insights = load_constituent_snapshot_insights(
    "output/AM300_history.xlsx", BUILD_VERSION
)
index_overlap_metrics = load_index_overlap_metrics(
    "output/AM100_history.xlsx",
    "output/AM200_history.xlsx",
    "output/AM300_history.xlsx",
    BUILD_VERSION,
)


def render_annual_returns_chart(df, title="Annual Calendar Returns (%)"):
    if df is None or df.empty:
        st.info("Annual calendar return data is not available yet.")
        return

    chart_df = df.copy().sort_values("Year").set_index("Year")
    fig, ax = plt.subplots(figsize=(12, 6))
    chart_df.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Return (%)")
    ax.set_xlabel("Year")
    ax.axhline(0, color="black", linewidth=0.8)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_annual_returns_summary(df):
    if df is None or df.empty:
        return

    st.markdown("### Performance Summary")

    summary = {}
    for col in ["AM100", "AM200", "AM300"]:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if series.empty:
            continue
        summary[col] = {
            "Best Year (%)": round(series.max(), 2),
            "Worst Year (%)": round(series.min(), 2),
            "Positive Years (%)": round((series > 0).mean() * 100, 1),
        }

    if summary:
        summary_df = pd.DataFrame(summary).T
        st.dataframe(summary_df, use_container_width=True)


def render_annual_returns_section(download_key):
    st.markdown("## Annual Calendar Returns (%)")
    st.markdown("Performance shown as calendar year returns based on total return index values.")

    file_path = "output/AM_annual_returns.csv"

    if not os.path.exists(file_path):
        st.error(f"File not found: {file_path}")
        return

    try:
        annual_returns = pd.read_csv(file_path)

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("### Annual Returns (%)")
            display_table = annual_returns.copy().round(1)
            display_table.columns = ["Year", "AM100", "AM200", "AM300"]
            display_table = display_table.set_index("Year")
            st.dataframe(display_table, use_container_width=True, height=400)

        with col2:
            st.markdown("### Annual Performance Chart")
            years = annual_returns["Year"].astype(str)
            x = np.arange(len(years))
            width = 0.22

            fig, ax = plt.subplots(figsize=(10, 5), facecolor="#0E1117")

            ax.bar(x - width, annual_returns["AM100"], width, label="AM100", color=CHART_COLORS["AM100"])
            ax.bar(x, annual_returns["AM200"], width, label="AM200", color=CHART_COLORS["AM200"])
            ax.bar(x + width, annual_returns["AM300"], width, label="AM300", color=CHART_COLORS["AM300"])

            ax.set_xticks(x)
            ax.set_xticklabels(years, rotation=0)
            ax.set_ylabel("Return (%)")
            ax.axhline(0, color="white", linewidth=1)
            style_dark_axes(ax)
            ax.legend(frameon=False)

            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        csv = annual_returns.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Annual Returns",
            csv,
            "AM_annual_returns.csv",
            "text/csv",
            key=download_key,
        )
    except Exception as e:
        st.error(f"Error loading annual returns: {e}")
        raise


def compute_annual_returns(df, column):
    yearly = df[["Date", column]].dropna().copy()
    yearly["Year"] = yearly["Date"].dt.year

    results = []
    for year, group in yearly.groupby("Year"):
        if len(group) < 2:
            continue
        start_value = group.iloc[0][column]
        end_value = group.iloc[-1][column]
        annual_return = (end_value / start_value) - 1
        results.append({"Year": year, column: annual_return * 100})

    return pd.DataFrame(results)


def compute_rolling_returns(df, column, years):
    rolling = df[["Date", column]].dropna().copy()
    rolling["Date"] = pd.to_datetime(rolling["Date"])
    rolling = rolling.set_index("Date")
    monthly = rolling[column].resample("ME").last()
    rolling_series = ((monthly / monthly.shift(years * 12)) ** (1 / years) - 1) * 100
    return rolling_series.to_frame(name=f"{years}Y").reset_index()


PORTFOLIOS = {
    "Conservative": {"AM100": 0.7, "AM200": 0.2, "AM300": 0.1},
    "Balanced": {"AM100": 0.4, "AM200": 0.4, "AM300": 0.2},
    "Growth": {"AM100": 0.2, "AM200": 0.3, "AM300": 0.5},
}


def build_portfolio_index(df, weights):
    return (
        df["AM100"] * weights["AM100"]
        + df["AM200"] * weights["AM200"]
        + df["AM300"] * weights["AM300"]
    )


@st.cache_data
def load_total_return_series_from_path(path, _version=None):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=["Date"])
    validate_df(df, os.path.basename(path), ["Date", "Index Level"])
    return validate_series(df.set_index("Date")["Index Level"], os.path.basename(path))


@st.cache_data
def load_eac_portfolio_file(path, _version=None):
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def format_pct(value, decimals=2):
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.{decimals}%}"


def format_num(value, decimals=2):
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.{decimals}f}"


def render_eac_banner():
    st.warning(
        "Historical performance reflects limited market depth and periods of high concentration, "
        "including single-constituent exposure. Data is presented for research transparency and "
        "does not represent an institutional benchmark track record."
    )


def render_eac_dashboard():
    eac25_series = load_total_return_series_from_path(
        "output/EAC25_total_return.csv",
        get_file_version("output/EAC25_total_return.csv"),
    )
    eac_ext_series = load_total_return_series_from_path(
        "output/EAC_EXT_total_return.csv",
        get_file_version("output/EAC_EXT_total_return.csv"),
    )
    eac25_metrics = load_metrics_csv(
        "output/EAC25_metrics.csv",
        get_file_version("output/EAC25_metrics.csv"),
    )
    eac_ext_metrics = load_metrics_csv(
        "output/EAC_EXT_metrics.csv",
        get_file_version("output/EAC_EXT_metrics.csv"),
    )
    eac25_portfolio = load_eac_portfolio_file(
        "output/EAC25_v2_core_portfolio.csv",
        get_file_version("output/EAC25_v2_core_portfolio.csv"),
    )
    eac_ext_portfolio = load_eac_portfolio_file(
        "output/EAC_extended_35_portfolio.csv",
        get_file_version("output/EAC_extended_35_portfolio.csv"),
    )
    eac25_summary = load_eac_portfolio_file(
        "output/EAC25_v2_core_summary.csv",
        get_file_version("output/EAC25_v2_core_summary.csv"),
    )
    eac_ext_summary = load_eac_portfolio_file(
        "output/EAC_extended_35_summary.csv",
        get_file_version("output/EAC_extended_35_summary.csv"),
    )
    eac25_annual = load_eac_portfolio_file(
        "output/EAC25_annual_returns.csv",
        get_file_version("output/EAC25_annual_returns.csv"),
    )
    eac_ext_annual = load_eac_portfolio_file(
        "output/EAC_EXT_annual_returns.csv",
        get_file_version("output/EAC_EXT_annual_returns.csv"),
    )

    required = [eac25_series, eac_ext_series, eac25_metrics, eac_ext_metrics, eac25_portfolio, eac_ext_portfolio]
    if any(item is None for item in required):
        st.error("EAC Series files are not fully available yet. Rebuild the EAC outputs before opening this section.")
        return

    eac_tabs = st.tabs(["Overview", "Risk", "Allocator"])
    eac25_country = compute_country_weights(eac25_portfolio)
    eac_ext_country = compute_country_weights(eac_ext_portfolio)
    eac25_drawdown = calculate_drawdown(eac25_series)
    eac_ext_drawdown = calculate_drawdown(eac_ext_series)
    eac25_vol_1y = rolling_volatility(eac25_series.pct_change().dropna()).dropna()
    eac_ext_vol_1y = rolling_volatility(eac_ext_series.pct_change().dropna()).dropna()

    eac25_current_top3 = float(eac25_portfolio["Weight"].nlargest(3).sum())
    eac_ext_current_top3 = float(eac_ext_portfolio["Weight"].nlargest(3).sum())
    eac25_effective_n = float(1 / np.square(eac25_portfolio["Weight"]).sum())
    eac_ext_effective_n = float(1 / np.square(eac_ext_portfolio["Weight"]).sum())

    def compute_trailing_growth_metrics(series):
        clean = series.dropna().sort_index()
        if clean.empty:
            return {
                "1Y": np.nan,
                "3Y": np.nan,
                "5Y": np.nan,
                "10Y": np.nan,
                "AverageAnnualReturn": np.nan,
                "CAGR": np.nan,
            }

        latest_date = clean.index[-1]
        latest_value = float(clean.iloc[-1])
        metrics = {}

        for years in [1, 3, 5, 10]:
            target_date = latest_date - pd.DateOffset(years=years)
            hist = clean[clean.index <= target_date]
            if hist.empty:
                metrics[f"{years}Y"] = np.nan
                continue
            start_date = hist.index[-1]
            start_value = float(hist.iloc[-1])
            elapsed_years = (latest_date - start_date).days / 365.25
            if elapsed_years <= 0 or start_value <= 0:
                metrics[f"{years}Y"] = np.nan
                continue
            metrics[f"{years}Y"] = (latest_value / start_value) ** (1 / elapsed_years) - 1

        daily_returns = clean.pct_change().dropna()
        if daily_returns.empty:
            metrics["AverageAnnualReturn"] = np.nan
            metrics["CAGR"] = np.nan
        else:
            metrics["AverageAnnualReturn"] = float(daily_returns.mean() * 252)
            total_years = (clean.index[-1] - clean.index[0]).days / 365.25
            metrics["CAGR"] = (
                (float(clean.iloc[-1]) / float(clean.iloc[0])) ** (1 / total_years) - 1
                if total_years > 0 and float(clean.iloc[0]) > 0
                else np.nan
            )
        return metrics

    eac25_growth_metrics = compute_trailing_growth_metrics(eac25_series)
    eac_ext_growth_metrics = compute_trailing_growth_metrics(eac_ext_series)
    eac25_adv_usd = float(eac25_portfolio["LiquidityUSD"].fillna(0).sum()) if "LiquidityUSD" in eac25_portfolio.columns else np.nan
    eac_ext_adv_usd = float(eac_ext_portfolio["LiquidityUSD"].fillna(0).sum()) if "LiquidityUSD" in eac_ext_portfolio.columns else np.nan
    eac25_capacity = eac25_adv_usd * 0.20 if pd.notna(eac25_adv_usd) else np.nan
    eac_ext_capacity = eac_ext_adv_usd * 0.20 if pd.notna(eac_ext_adv_usd) else np.nan

    with eac_tabs[0]:
        st.markdown("## EAC SERIES — EAST AFRICA DEPLOYABLE EQUITY PORTFOLIOS")
        st.caption("EAC25 Core | EAC Extended (35)")
        st.markdown(
            "The EAC Series provides a rules-based, liquidity-screened representation of deployable "
            "East African equity markets. Portfolios are constructed using observed trading activity "
            "and execution-aware constraints."
        )
        render_eac_banner()
        st.info(
            "No period within the available history meets institutional maturity thresholds "
            "(>=15 constituents and effective diversification).\n\n"
            "Accordingly, the EAC Series is presented as a current deployable portfolio framework, "
            "with historical data provided for transparency only."
        )

        left_col, right_col = st.columns([1, 1])
        with left_col:
            st.markdown("### Portfolio Snapshot")
            snapshot_df = pd.DataFrame(
                {
                    "Metric": ["Constituents", "Countries", "Max Stock Weight", "Max Country Weight", "Rebalance"],
                    "EAC25 Core": [
                        int(len(eac25_portfolio)),
                        int(eac25_portfolio["Country"].nunique()),
                        "10%",
                        "40%",
                        "Quarterly",
                    ],
                    "EAC Extended": [
                        int(len(eac_ext_portfolio)),
                        int(eac_ext_portfolio["Country"].nunique()),
                        "10%",
                        "40%",
                        "Quarterly",
                    ],
                }
            )
            st.dataframe(snapshot_df, use_container_width=True, hide_index=True)

        with right_col:
            st.markdown("### Investment Summary")
            st.markdown(
                """
                **EAC25 Core** represents the concentrated, institutionally investable segment of East African equities.

                **EAC Extended** captures the full set of validated and liquid securities, providing broader regional exposure while maintaining execution feasibility.
                """
            )

        st.markdown("### Research Index Series (USD Total Return)")
        st.caption("Not representative of an investable benchmark track record.")
        fig, ax = plt.subplots(figsize=(10, 4), facecolor="#0E1117")
        style_chart(fig, ax)
        ax.plot(eac25_series.index, eac25_series, label="EAC25 Core", color="#4DA3FF", linewidth=2)
        ax.plot(eac_ext_series.index, eac_ext_series, label="EAC Extended", color="#22C55E", linewidth=2)
        ax.legend(frameon=False, fontsize=10)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.markdown("#### Research Performance Summary (Non-Benchmark)")
        perf_df = pd.DataFrame(
            {
                "Metric": ["CAGR", "Volatility", "Sharpe", "Max Drawdown"],
                "EAC25": [
                    format_pct(eac25_metrics.get("CAGR")),
                    format_pct(eac25_metrics.get("Volatility")),
                    format_num(eac25_metrics.get("Sharpe")),
                    format_pct(eac25_metrics.get("Max Drawdown")),
                ],
                "EAC EXT": [
                    format_pct(eac_ext_metrics.get("CAGR")),
                    format_pct(eac_ext_metrics.get("Volatility")),
                    format_num(eac_ext_metrics.get("Sharpe")),
                    format_pct(eac_ext_metrics.get("Max Drawdown")),
                ],
            }
        )
        st.dataframe(perf_df, use_container_width=True, hide_index=True)
        st.caption(
            "These metrics use each EAC series over its full historical path, including early periods of high concentration and limited breadth."
        )

        st.markdown("#### Trailing Returns & Average Growth")
        growth_df = pd.DataFrame(
            {
                "Metric": ["1Y", "3Y CAGR", "5Y CAGR", "10Y CAGR", "Average Annual Return", "CAGR"],
                "EAC25": [
                    format_pct(eac25_growth_metrics.get("1Y")),
                    format_pct(eac25_growth_metrics.get("3Y")),
                    format_pct(eac25_growth_metrics.get("5Y")),
                    format_pct(eac25_growth_metrics.get("10Y")),
                    format_pct(eac25_growth_metrics.get("AverageAnnualReturn")),
                    format_pct(eac25_growth_metrics.get("CAGR")),
                ],
                "EAC EXT": [
                    format_pct(eac_ext_growth_metrics.get("1Y")),
                    format_pct(eac_ext_growth_metrics.get("3Y")),
                    format_pct(eac_ext_growth_metrics.get("5Y")),
                    format_pct(eac_ext_growth_metrics.get("10Y")),
                    format_pct(eac_ext_growth_metrics.get("AverageAnnualReturn")),
                    format_pct(eac_ext_growth_metrics.get("CAGR")),
                ],
            }
        )
        st.dataframe(growth_df, use_container_width=True, hide_index=True)
        st.caption("Average Annual Return is the arithmetic annualized mean from daily returns. CAGR is the compounded growth rate over full live history.")

        if eac25_annual is not None and eac_ext_annual is not None:
            st.markdown("### Annual Returns (%)")
            annual_display = pd.merge(
                eac25_annual.rename(columns={"ReturnPct": "EAC25 Core"}),
                eac_ext_annual.rename(columns={"ReturnPct": "EAC Extended"}),
                on="Year",
                how="outer",
            ).sort_values("Year")
            annual_table = annual_display.copy()
            for col in ["EAC25 Core", "EAC Extended"]:
                annual_table[col] = annual_table[col].map(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A")

            annual_col1, annual_col2 = st.columns([1, 1.25])
            with annual_col1:
                st.dataframe(annual_table, use_container_width=True, hide_index=True)
            with annual_col2:
                chart_df = annual_display.set_index("Year")
                fig, ax = plt.subplots(figsize=(8, 4), facecolor="#0E1117")
                style_chart(fig, ax)
                chart_df.plot(kind="bar", ax=ax, color=["#4DA3FF", "#22C55E"], width=0.75)
                ax.set_ylabel("Return (%)", color="#CCCCCC")
                ax.legend(frameon=False)
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

        st.markdown("### Structural Insight")
        st.markdown(
            "The East African public equity market currently supports approximately 30-40 securities that meet "
            "minimum liquidity and trading consistency thresholds.\n\n"
            "This structural constraint drives concentration and impacts both portfolio construction and historical performance characteristics."
        )

    with eac_tabs[1]:
        st.markdown("## EAC SERIES — RISK CHARACTERISTICS")
        st.caption("Risk metrics are influenced by early-period concentration and should be interpreted with caution.")
        render_eac_banner()

        st.markdown("### Drawdown (Full History — Research Context)")
        fig, ax = plt.subplots(figsize=(10, 4), facecolor="#0E1117")
        style_chart(fig, ax)
        ax.plot(eac25_drawdown.index, eac25_drawdown, label="EAC25 Core", color="#4DA3FF", linewidth=2)
        ax.plot(eac_ext_drawdown.index, eac_ext_drawdown, label="EAC Extended", color="#22C55E", linewidth=2)
        ax.legend(frameon=False)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        vol_col, conc_col = st.columns([1, 1])
        with vol_col:
            st.markdown("### Volatility")
            vol_df = pd.DataFrame(
                {
                    "Metric": ["Annualised Volatility"],
                    "EAC25": [format_pct(eac25_metrics.get("Volatility"))],
                    "EAC EXT": [format_pct(eac_ext_metrics.get("Volatility"))],
                }
            )
            st.dataframe(vol_df, use_container_width=True, hide_index=True)

        with conc_col:
            st.markdown("### Concentration Risk")
            conc_df = pd.DataFrame(
                {
                    "Metric": ["Top 1 Weight", "Top 3 Weight", "Max Stock Cap", "Country Cap", "Effective N (current)"],
                    "EAC25": [f"{eac25_portfolio['Weight'].max():.2%}", f"{eac25_current_top3:.2%}", "10%", "40%", f"{eac25_effective_n:.2f}"],
                    "EAC EXT": [f"{eac_ext_portfolio['Weight'].max():.2%}", f"{eac_ext_current_top3:.2%}", "10%", "40%", f"{eac_ext_effective_n:.2f}"],
                }
            )
            st.dataframe(conc_df, use_container_width=True, hide_index=True)

        st.markdown("### Rolling Volatility")
        fig, ax = plt.subplots(figsize=(10, 4), facecolor="#0E1117")
        style_chart(fig, ax)
        ax.plot(eac25_vol_1y.index, eac25_vol_1y, label="EAC25 Core", color="#4DA3FF", linewidth=2)
        ax.plot(eac_ext_vol_1y.index, eac_ext_vol_1y, label="EAC Extended", color="#22C55E", linewidth=2)
        ax.legend(frameon=False)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.markdown(
            "Portfolio risk is primarily driven by concentration and liquidity constraints rather than broad market diversification."
        )

    with eac_tabs[2]:
        st.markdown("## EAC SERIES — ALLOCATION & DEPLOYMENT")

        st.markdown("### Country Allocation (Post-Constraint Weights)")
        alloc_col1, alloc_col2 = st.columns(2)
        with alloc_col1:
            fig, ax = plt.subplots(figsize=(6, 3.2), facecolor="#0E1117")
            style_chart(fig, ax)
            ax.bar(eac25_country["Country"], eac25_country["Weight"], color="#4DA3FF")
            ax.set_title("EAC25 Core", color="#CCCCCC")
            ax.tick_params(axis="x", rotation=45)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        with alloc_col2:
            fig, ax = plt.subplots(figsize=(6, 3.2), facecolor="#0E1117")
            style_chart(fig, ax)
            ax.bar(eac_ext_country["Country"], eac_ext_country["Weight"], color="#22C55E")
            ax.set_title("EAC Extended (35)", color="#CCCCCC")
            ax.tick_params(axis="x", rotation=45)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        top_col1, top_col2 = st.columns(2)
        with top_col1:
            st.markdown("### Top 10 Holdings — EAC25 Core")
            top25 = eac25_portfolio.sort_values(["Weight", "Company"], ascending=[False, True]).head(10).copy()
            top25["Rank"] = range(1, len(top25) + 1)
            top25["Weight"] = top25["Weight"].map("{:.2%}".format)
            top25 = top25[["Rank", "Company", "Country", "Weight"]]
            st.dataframe(top25, use_container_width=True, hide_index=True)
        with top_col2:
            st.markdown("### Top 10 Holdings — EAC Extended")
            top_ext = eac_ext_portfolio.sort_values(["Weight", "Company"], ascending=[False, True]).head(10).copy()
            top_ext["Rank"] = range(1, len(top_ext) + 1)
            top_ext["Weight"] = top_ext["Weight"].map("{:.2%}".format)
            top_ext = top_ext[["Rank", "Company", "Country", "Weight"]]
            st.dataframe(top_ext, use_container_width=True, hide_index=True)

        capacity_display = pd.DataFrame(
            {
                "Metric": [
                    "Average Daily Traded Value (USD)",
                    "Investable Capacity (USD, 20%)",
                ],
                "EAC25": [f"{eac25_adv_usd:,.0f}", f"{eac25_capacity:,.0f}"],
                "EAC EXT": [f"{eac_ext_adv_usd:,.0f}", f"{eac_ext_capacity:,.0f}"],
            }
        )
        st.dataframe(capacity_display, use_container_width=True, hide_index=True)
        st.caption(
            "Average Daily Traded Value (USD) reflects summed constituent liquidity. Investable Capacity (USD, 20%) applies a conservative institutional participation assumption."
        )

        st.markdown("### Liquidity Profile")
        fig = px.histogram(
            eac25_portfolio,
            x="LiquidityUSD",
            nbins=30,
            title="EAC25 Liquidity Distribution",
            template="plotly_dark",
        )
        fig.update_layout(
            xaxis_title="Liquidity (USD)",
            yaxis_title="Count",
            hovermode="x unified",
            margin=dict(l=20, r=20, t=60, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Liquidity is concentrated in a limited number of securities, with a long tail of lower-trading names.")

        st.markdown("### Minimum Deployment Size")
        deploy_df = pd.DataFrame(
            {
                "Metric": ["Theoretical Minimum (1 share per constituent)", "Practical Minimum", "Recommended Institutional Ticket"],
                "EAC25": [
                    f"${float(eac25_summary.iloc[0]['StrictMinDeploymentUSD']):,.0f}" if eac25_summary is not None else "N/A",
                    "$50k-$100k",
                    "$250k-$500k",
                ],
                "EAC EXT": [
                    f"${float(eac_ext_summary.iloc[0]['StrictMinDeploymentUSD']):,.0f}" if eac_ext_summary is not None else "N/A",
                    "$75k-$150k",
                    "$300k-$600k",
                ],
            }
        )
        st.dataframe(deploy_df, use_container_width=True, hide_index=True)
        st.caption(
            "While theoretical minimum investment levels are low, practical deployment requires significantly higher capital "
            "to achieve diversification and execution efficiency."
        )

        st.markdown("### Interpretation")
        st.markdown(
            "The EAC Series should be viewed as a deployable regional allocation sleeve rather than a broad market benchmark.\n\n"
            "Capital deployment is constrained by:\n"
            "- Limited market depth\n"
            "- Concentrated liquidity\n"
            "- Execution considerations\n\n"
            "The framework provides transparency into these constraints rather than masking them."
        )


def render_am_eac_comparison():
    eac25_series = load_total_return_series_from_path(
        "output/EAC25_total_return.csv",
        get_file_version("output/EAC25_total_return.csv"),
    )
    eac25_metrics = load_metrics_csv(
        "output/EAC25_metrics.csv",
        get_file_version("output/EAC25_metrics.csv"),
    )
    eac25_portfolio = load_eac_portfolio_file(
        "output/EAC25_v2_core_portfolio.csv",
        get_file_version("output/EAC25_v2_core_portfolio.csv"),
    )
    eac25_summary = load_eac_portfolio_file(
        "output/EAC25_v2_core_summary.csv",
        get_file_version("output/EAC25_v2_core_summary.csv"),
    )

    if any(item is None for item in [eac25_series, eac25_metrics, eac25_portfolio, eac25_summary, am100_metrics]):
        st.error("Comparison view is not fully available yet. Build the EAC and AM outputs before opening this section.")
        return

    am100_snapshot = am100_latest.copy()
    am100_top1 = float(am100_snapshot["Weight"].max())
    am100_top3 = float(am100_snapshot["Weight"].nlargest(3).sum())
    am100_effective_n = float(1 / np.square(am100_snapshot["Weight"]).sum())
    am100_country_count = int(am100_snapshot["Country"].nunique())
    am100_theoretical_min = float((am100_snapshot["Price"] / am100_snapshot["Weight"]).replace([np.inf, -np.inf], np.nan).max())

    eac25_top1 = float(eac25_portfolio["Weight"].max())
    eac25_top3 = float(eac25_portfolio["Weight"].nlargest(3).sum())
    eac25_effective_n = float(1 / np.square(eac25_portfolio["Weight"]).sum())
    eac25_country_df = compute_country_weights(eac25_portfolio)
    merged = pd.merge(
        am100_series.rename("AM100").reset_index(),
        eac25_series.rename("EAC25").reset_index(),
        on="Date",
        how="inner",
    ).sort_values("Date")
    merged = merged.set_index("Date")
    drawdown_compare = pd.DataFrame(
        {
            "AM100": calculate_drawdown(merged["AM100"]),
            "EAC25": calculate_drawdown(merged["EAC25"]),
        }
    )

    st.markdown("## AM vs EAC — Portfolio Comparison")
    st.caption("AM100 vs EAC25 Core")
    st.markdown(
        "This comparison highlights the structural, risk, and deployment differences between pan-African "
        "index exposure and East African regional allocation."
    )
    st.warning(
        "EAC historical performance is provided for research transparency only and does not represent a mature benchmark series."
    )

    snap_left, snap_right = st.columns([1.3, 1])
    with snap_left:
        st.markdown("### Side-by-Side Snapshot")
        snapshot_df = pd.DataFrame(
            {
                "Metric": ["Constituents", "Countries", "Max Stock Weight", "Max Country Weight", "Rebalance"],
                "AM100": [len(am100_snapshot), am100_country_count, f"{am100_top1:.2%}", "25%", "Quarterly"],
                "EAC25 Core": [len(eac25_portfolio), int(eac25_portfolio["Country"].nunique()), "10%", "40%", "Quarterly"],
            }
        )
        st.dataframe(snapshot_df, use_container_width=True, hide_index=True)
    with snap_right:
        st.markdown("### Allocation Insight")
        st.markdown(
            "EAC portfolios are significantly more concentrated, reflecting the limited depth of regional markets "
            "compared to the broader African universe."
        )

    st.markdown("### Performance Comparison (Contextual)")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=merged.index,
            y=merged["AM100"],
            mode="lines",
            name="AM100",
            line=dict(width=2, color=AM100_COLOR),
            hovertemplate="%{x}<br>Value: %{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=merged.index,
            y=merged["EAC25"],
            mode="lines",
            name="EAC25",
            line=dict(width=2, dash="dash", color="#22C55E"),
            hovertemplate="%{x}<br>Value: %{y:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Research Index Series (USD Total Return)",
        xaxis_title="Date",
        yaxis_title="Index Level",
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.05),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    perf_df = pd.DataFrame(
        {
            "Metric": ["CAGR", "Volatility", "Sharpe", "Max Drawdown"],
            "AM100": [
                format_pct(am100_metrics.get("CAGR")),
                format_pct(am100_metrics.get("Volatility")),
                format_num(am100_metrics.get("Sharpe")),
                format_pct(am100_metrics.get("Max Drawdown")),
            ],
            "EAC25": [
                format_pct(eac25_metrics.get("CAGR")),
                format_pct(eac25_metrics.get("Volatility")),
                format_num(eac25_metrics.get("Sharpe")),
                format_pct(eac25_metrics.get("Max Drawdown")),
            ],
        }
    )
    st.dataframe(perf_df, use_container_width=True, hide_index=True)
    st.caption(
        "EAC performance reflects periods of limited market depth and high concentration and is not directly comparable to AM series benchmarks."
    )

    st.markdown("### Concentration Comparison")
    conc_col1, conc_col2 = st.columns(2)
    with conc_col1:
        top10_am100 = am100_snapshot.sort_values(["Weight", "Company"], ascending=[False, True]).head(10).copy()
        top10_eac = eac25_portfolio.sort_values(["Weight", "Company"], ascending=[False, True]).head(10).copy()
        top10_am100["WeightPct"] = top10_am100["Weight"] * 100
        top10_eac["WeightPct"] = top10_eac["Weight"] * 100
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=top10_am100["Company"],
                y=top10_am100["WeightPct"],
                name="AM100",
                marker_color=AM100_COLOR,
            )
        )
        fig.add_trace(
            go.Bar(
                x=top10_eac["Company"],
                y=top10_eac["WeightPct"],
                name="EAC25",
                marker_color="#22C55E",
            )
        )
        fig.update_layout(
            title="Top 10 Holdings (Weight Comparison)",
            barmode="group",
            template="plotly_dark",
            xaxis_tickangle=-45,
            yaxis_title="Weight (%)",
            hovermode="x unified",
            margin=dict(l=20, r=20, t=60, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)
    with conc_col2:
        conc_df = pd.DataFrame(
            {
                "Metric": ["Top 1 Weight", "Top 3 Weight", "Effective N"],
                "AM100": [f"{am100_top1:.2%}", f"{am100_top3:.2%}", f"{am100_effective_n:.2f}"],
                "EAC25": [f"{eac25_top1:.2%}", f"{eac25_top3:.2%}", f"{eac25_effective_n:.2f}"],
            }
        )
        st.dataframe(conc_df, use_container_width=True, hide_index=True)
        st.markdown(
            "EAC exposure concentrates risk in a small number of securities, whereas AM100 distributes risk across a broader and more diversified universe."
        )

    st.markdown("### Country Exposure Comparison")
    am_country_plot = am100_country_df.copy()
    eac_country_plot = eac25_country_df.copy()
    am_country_plot.columns = ["Country", "Weight"]
    eac_country_plot.columns = ["Country", "Weight"]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=am_country_plot["Country"],
            y=am_country_plot["Weight"] * 100,
            name="AM100",
            marker_color=AM100_COLOR,
        )
    )
    fig.add_trace(
        go.Bar(
            x=eac_country_plot["Country"],
            y=eac_country_plot["Weight"] * 100,
            name="EAC25",
            marker_color="#22C55E",
        )
    )
    fig.update_layout(
        title="Country Allocation Comparison",
        barmode="group",
        template="plotly_dark",
        xaxis_title="Country",
        yaxis_title="Weight (%)",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        "EAC portfolios are structurally concentrated in Kenya and Tanzania, whereas AM100 provides broader continental exposure across multiple markets."
    )

    st.markdown("### Drawdown Comparison")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=drawdown_compare.index,
            y=drawdown_compare["AM100"],
            name="AM100 Drawdown",
            mode="lines",
            line=dict(width=2, color=AM100_COLOR),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=drawdown_compare.index,
            y=drawdown_compare["EAC25"],
            name="EAC25 Drawdown",
            mode="lines",
            line=dict(width=2, color="#22C55E"),
        )
    )
    fig.update_layout(
        title="Drawdown Comparison",
        template="plotly_dark",
        yaxis_title="Drawdown",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Liquidity & Deployment")
    liquidity_df = pd.DataFrame(
        {
            "Metric": ["Avg Liquidity", "Tail Liquidity Risk", "Market Depth"],
            "AM100": [f"${am100_adv_usd:,.0f} ADV", "Low", "Broad"],
            "EAC25": [f"${float(eac25_portfolio['LiquidityUSD'].mean()):,.0f} ADV", "High", "Limited"],
        }
    )
    st.dataframe(liquidity_df, use_container_width=True, hide_index=True)
    st.markdown(
        "EAC deployment is constrained by market depth and liquidity concentration, requiring more careful execution and higher sensitivity to trade size."
    )

    st.markdown("### Minimum Investment Comparison")
    min_df = pd.DataFrame(
        {
            "Metric": ["Theoretical Minimum", "Practical Minimum", "Institutional Ticket"],
            "AM100": [f"${am100_theoretical_min:,.0f}", "$300k-$500k", "$500k+"],
            "EAC25": [
                f"${float(eac25_summary.iloc[0]['StrictMinDeploymentUSD']):,.0f}",
                "$50k-$100k",
                "$250k-$500k",
            ],
        }
    )
    st.dataframe(min_df, use_container_width=True, hide_index=True)
    st.markdown(
        "While EAC portfolios appear accessible at lower capital levels, practical deployment remains constrained by liquidity and execution considerations."
    )

    st.markdown("### Allocator Interpretation")
    st.markdown(
        "AM100 represents a diversified, pan-African allocation suitable for broad exposure.\n\n"
        "EAC25 Core represents a concentrated regional sleeve, offering targeted exposure to East African markets "
        "but with higher concentration and liquidity risk.\n\n"
        "Allocators should consider EAC exposure as a complementary allocation rather than a substitute for pan-African strategies."
    )

    st.markdown("### Key Takeaways")
    st.markdown(
        "- Africa is investable, but unevenly\n"
        "- Regional exposure introduces concentration risk\n"
        "- Liquidity constraints define deployability\n"
        "- EAC portfolios reflect real market structure, not theoretical coverage"
    )

validated_securities_count = 0
eligible_securities_count = 0
full_trading_coverage_count = 0
ineligible_securities_count = 0

# Initialized early so any pre-load UI block can fail safely to N/A
# instead of crashing before the real metrics object is built later.
comparison_metrics = {}
common_start = None
common_end = None
common_period_label = "N/A"
comparison_series = {}


def render_allocator_view():
    local_common = get_common_analysis_window(
        {"AM100": am100_series, "AM200": am200_series, "AM300": am300_series}
    )
    local_common_start = local_common["start"]
    local_common_end = local_common["end"]
    local_comparison_series = local_common["series"]
    local_comparison_metrics = normalize_comparison_metrics(
        {
            name: compute_window_metrics(series)
            for name, series in local_comparison_series.items()
        }
        if isinstance(local_comparison_series, dict)
        else {}
    )

    def comparison_metric(index_name):
        if not isinstance(local_comparison_metrics, dict):
            return {}
        return local_comparison_metrics.get(index_name, {})

    def pct_metric(metrics, key):
        value = metrics.get(key) if metrics else None
        return f"{value * 100:.2f}%" if value is not None and pd.notna(value) else "N/A"

    def num_metric(metrics, key):
        value = metrics.get(key) if metrics else None
        return f"{value:.2f}" if value is not None and pd.notna(value) else "N/A"

    def turnover_metric(snapshot):
        value = snapshot.get("avg_turnover")
        return f"{value * 100:.1f}%" if value is not None else "N/A"

    st.markdown(
        """
        # AM African Index Suite

        ### Institutional Access to Investable African Equities
        """
    )

    st.markdown(
        """
        AM100 represents the investable African equity universe under strict institutional liquidity and trading constraints.

        The AM Index family provides a tiered framework:
        - AM100: Institutional Core
        - AM200: Expanded Investable Universe
        - AM300: Frontier Opportunity Set with controlled trading-frequency relaxation
        """
    )
    am100_compare = comparison_metric("AM100")
    am200_compare = comparison_metric("AM200")
    am300_compare = comparison_metric("AM300")

    st.markdown(
        """
        ### Index Structure

        AM100 ⊂ AM200 ⊂ AM300  
        (All tiers are fully nested and methodology-consistent)
        """
    )

    st.markdown(
        """
        ### What Makes This Different

        • Investability-driven (not market-cap driven)  
        • USD liquidity enforced  
        • Trading consistency required  
        • Fully rules-based and auditable
        """
    )

    comparison_df = pd.DataFrame(
        {
            "Metric": [
                "CAGR",
                "Volatility",
                "Sharpe Ratio",
                "Max Drawdown",
                "Constituents",
                "Turnover",
            ],
            "AM100": [
                pct_metric(am100_compare, "CAGR"),
                pct_metric(am100_compare, "Volatility"),
                num_metric(am100_compare, "Sharpe"),
                pct_metric(am100_compare, "Max Drawdown"),
                am100_snapshot_insights.get("constituents"),
                turnover_metric(am100_snapshot_insights),
            ],
            "AM200": [
                pct_metric(am200_compare, "CAGR"),
                pct_metric(am200_compare, "Volatility"),
                num_metric(am200_compare, "Sharpe"),
                pct_metric(am200_compare, "Max Drawdown"),
                am200_snapshot_insights.get("constituents"),
                turnover_metric(am200_snapshot_insights),
            ],
            "AM300": [
                pct_metric(am300_compare, "CAGR"),
                pct_metric(am300_compare, "Volatility"),
                num_metric(am300_compare, "Sharpe"),
                pct_metric(am300_compare, "Max Drawdown"),
                am300_snapshot_insights.get("constituents"),
                turnover_metric(am300_snapshot_insights),
            ],
        }
    )
    start_date = local_common_start.strftime("%d %b %Y") if local_common_start is not None else "N/A"
    end_date = local_common_end.strftime("%d %b %Y") if local_common_end is not None else "N/A"
    st.markdown(
        f"""
        **Performance Period**  
        {start_date} → {end_date}
        """
    )
    st.caption(
        "All performance metrics are calculated over the shared period where all index series are concurrently available. Turnover remains the average across rebalance events."
    )
    st.dataframe(comparison_df, use_container_width=True, hide_index=True, height=240)
    st.caption(
        "Metrics are calculated using USD total return methodology. Volatility and Sharpe Ratio are annualised. Max Drawdown is measured over the full shared period."
    )

    st.markdown(
        """
        ### Interpretation

        • **AM100** represents the institutional core, comprising the most liquid and consistently traded equities.

        • **AM200** extends into mid-cap and frontier segments, delivering higher returns over the shared period, with:
          - increased volatility
          - deeper drawdowns

        • **AM300** provides broader frontier exposure, where trading activity is observable but less consistent.

        Performance dispersion across indices reflects liquidity constraints and investability, not differences in methodology.
        """
    )

    if investability_map is not None and not investability_map.empty:
        st.markdown("### Africa Investability Map")
        iso_map = {
            "SOUTH AFRICA": "ZAF",
            "EGYPT": "EGY",
            "MOROCCO": "MAR",
            "NIGERIA": "NGA",
            "KENYA": "KEN",
            "NAMIBIA": "NAM",
            "ZIMBABWE": "ZWE",
            "SENEGAL": "SEN",
            "MAURITIUS": "MUS",
            "UGANDA": "UGA",
            "GHANA": "GHA",
            "TANZANIA": "TZA",
            "MALAWI": "MWI",
            "ZAMBIA": "ZMB",
            "NIGER": "NER",
            "TOGO": "TGO",
            "RWANDA": "RWA",
            "TUNISIA": "TUN",
        }
        map_df = investability_map.copy()
        map_df["Country"] = map_df["Country"].astype(str).str.upper()
        map_df["ISO"] = map_df["Country"].map(iso_map)
        map_df = map_df.dropna(subset=["ISO"]).copy()
        map_df["StatusColor"] = map_df["Inclusion Status"].map({"Included": 1, "Excluded": 0})
        fig = px.choropleth(
            map_df,
            locations="ISO",
            color="StatusColor",
            hover_name="Country",
            hover_data={"StatusColor": False, "ISO": False},
            color_continuous_scale=["#d62728", "#2ca02c"],
            range_color=(0, 1),
        )
        fig.update_layout(
            coloraxis_showscale=False,
            geo=dict(
                scope="africa",
                projection_type="natural earth",
                showland=True,
                landcolor="#0E1117",
                bgcolor="#0E1117",
            ),
            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117",
            font=dict(size=10),
            margin=dict(l=0, r=0, t=0, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        """
        ### Use Case

        Suitable for:
        - Pension fund allocations
        - Sovereign capital deployment
        - Institutional portfolio construction
        """
    )

    with st.expander("📘 Methodology & Overview", expanded=False):
        st.markdown(methodology_text)

if IS_INVESTOR:
    st.title("AM100 - African Institutional Equity Index")
    st.markdown(intro_text)

    if am100_metrics:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("CAGR", f"{am100_metrics.get('CAGR', 0) * 100:.1f}%")
        col2.metric("Volatility", f"{am100_metrics.get('Volatility', 0) * 100:.1f}%")
        col3.metric("Sharpe", f"{am100_metrics.get('Sharpe', 0):.2f}")
        col4.metric(
            "Max Drawdown", f"{am100_metrics.get('Max Drawdown', 0) * 100:.1f}%"
        )

    if am100_snapshot_insights:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(
            "Top 10 Concentration",
            f"{am100_snapshot_insights.get('top10_weight', 0) * 100:.1f}%",
        )
        avg_turnover = am100_snapshot_insights.get("avg_turnover")
        col2.metric(
            "Avg Turnover",
            f"{avg_turnover * 100:.1f}%" if avg_turnover is not None else "N/A",
        )
        col3.metric(
            "Constituents", f"{am100_snapshot_insights.get('constituents', 0)}"
        )
        col4.metric(
            "Top Country", safe_display(am100_snapshot_insights.get("top_country"))
        )
        st.caption(
            """
            Turnover reflects quarterly rebalancing in markets with evolving liquidity conditions.
            Higher turnover is typical of frontier equity markets and reflects real investability constraints.
            """
        )

    if index_overlap_metrics:
        st.subheader("Index Structure")
        col1, col2, col3 = st.columns(3)
        col1.metric(
            "AM100 ⊂ AM200",
            f"{index_overlap_metrics.get('AM100_in_AM200', 0) * 100:.0f}%",
        )
        col2.metric(
            "AM100 ⊂ AM300",
            f"{index_overlap_metrics.get('AM100_in_AM300', 0) * 100:.0f}%",
        )
        col3.metric(
            "AM200 ⊂ AM300",
            f"{index_overlap_metrics.get('AM200_in_AM300', 0) * 100:.0f}%",
        )
        st.caption(
            """
            Each index is a strict superset of the previous tier, ensuring consistency,
            transparency, and comparability across the AM index family.
            """
        )
else:
    pass

if IS_INTERNAL and all(
    x is not None
    for x in [
        am100_metrics,
        am200_metrics,
        am300_metrics,
        am100_snapshot_insights,
        am200_snapshot_insights,
        am300_snapshot_insights,
    ]
):
    def comparison_metric(index_name):
        if not isinstance(comparison_metrics, dict):
            return {}
        return comparison_metrics.get(index_name, {})

    def pct_metric(metrics, key):
        value = metrics.get(key) if metrics else None
        return f"{value * 100:.2f}%" if value is not None and pd.notna(value) else "N/A"

    def num_metric(metrics, key):
        value = metrics.get(key) if metrics else None
        return f"{value:.2f}" if value is not None and pd.notna(value) else "N/A"

    def turnover_metric(snapshot):
        value = snapshot.get("avg_turnover")
        return f"{value * 100:.1f}%" if value is not None else "N/A"

    am100_compare = comparison_metric("AM100")
    am200_compare = comparison_metric("AM200")
    am300_compare = comparison_metric("AM300")

    comparison_df = pd.DataFrame(
        {
            "Metric": [
                "CAGR",
                "Volatility",
                "Sharpe Ratio",
                "Max Drawdown",
                "Constituents",
                "Turnover",
            ],
            "AM100": [
                pct_metric(am100_compare, "CAGR"),
                pct_metric(am100_compare, "Volatility"),
                num_metric(am100_compare, "Sharpe"),
                pct_metric(am100_compare, "Max Drawdown"),
                am100_snapshot_insights.get("constituents"),
                turnover_metric(am100_snapshot_insights),
            ],
            "AM200": [
                pct_metric(am200_compare, "CAGR"),
                pct_metric(am200_compare, "Volatility"),
                num_metric(am200_compare, "Sharpe"),
                pct_metric(am200_compare, "Max Drawdown"),
                am200_snapshot_insights.get("constituents"),
                turnover_metric(am200_snapshot_insights),
            ],
            "AM300": [
                pct_metric(am300_compare, "CAGR"),
                pct_metric(am300_compare, "Volatility"),
                num_metric(am300_compare, "Sharpe"),
                pct_metric(am300_compare, "Max Drawdown"),
                am300_snapshot_insights.get("constituents"),
                turnover_metric(am300_snapshot_insights),
            ],
        }
    )


methodology_text = """
## African Market Indices Framework

### Overview

The African Market Indices platform provides a **liquidity-driven, investable benchmark system** designed to represent the most accessible equity opportunities across African markets.

The indices are constructed to reflect **real-world investability**, prioritising securities that institutional and professional investors can realistically access, trade, and scale.

---

### Index Structure

* **AM100 (Core Index)**
  A concentrated, institutional-grade index of the **100 most liquid and investable equities** across Africa.

* **AM200 (Expansion Index)**
  A broader index capturing the **next 100 securities**, offering exposure to emerging, frontier, and mid-cap opportunities.

* **AM300 (Opportunity Set)**
  A broader total return index that includes securities with lower but still observable trading frequency, subject to minimum thresholds of both historical and recent trading activity.

---

### Data Coverage

* **Time Period:** January 2016 -> Present
* **Frequency:** Daily (no interpolation)
* **Markets Covered:** Multi-country African equity universe
* **Data Inputs:**

  * Daily closing prices
  * Trading volumes
  * Corporate actions (where available)
  * FX-adjusted pricing (USD base)

All data is standardised into a **consistent USD framework** to enable cross-country comparability.

---

### Methodology

#### Institutional Selection Rules

Constituents are screened using hard institutional rules rather than performance optimisation:

* USD-based liquidity filter: **30-day ADV >= $1M**
* Trading activity: **>= 90% of trading days**
* Verified total return inputs: **price, volume, and dividends**
* USD-normalised comparison across markets

---

#### Index Construction Rules

* **Quarterly observation**
* **One-month implementation lag**
* **Entry/Exit Buffers:** 100 / 120
* **Stock Cap:** 5% maximum
* **Country Cap:** 25% maximum
* **No synthetic smoothing or interpolation**

---

### Eligibility Criteria

To qualify for inclusion, securities must:

* Exhibit consistent trading activity
* Meet minimum liquidity thresholds
* Have reliable and continuous price data
* Be accessible to institutional investors

AM300 applies a controlled extension of these rules by allowing lower, but still observable, trading frequency while retaining minimum thresholds for both historical positive trading coverage and recent non-zero trading activity.

---

### Inclusion & Removal

**Entry into Index:**

* Meet hard liquidity and trading thresholds
* Sustain data integrity and dividend coverage
* Rank within the eligible investable set

**Removal from Index:**

* Fall below liquidity or trading thresholds
* Lose required data integrity
* Fail to maintain ranking within the buffer range

This ensures **stability while remaining responsive to market conditions**.

---

### Investment Rationale

The indices are designed to:

* Provide a **true representation of investable Africa**
* Enable **cross-border portfolio construction**
* Support **institutional allocation decisions**
* Deliver **transparent, rules-based exposure**

---

### Key Characteristics

* Institutional liquidity methodology
* Multi-country diversification
* FX-normalised performance (USD)
* Rules-based, transparent construction
* Designed for scalability and replication

---

### Intended Users

This platform is built for:

* Asset managers
* Pension funds
* Banks and wealth platforms
* Institutional investors
* Regulators and market participants

---

### Purpose

> **A unified, investable benchmark for African equities**

Enabling Africa to be viewed not as fragmented markets, but as a **cohesive investment opportunity set**.

---

### Ongoing Development

The platform continues to evolve with:

* Expanded market coverage
* Enhanced data depth
* API access for institutional integration
* Automated reporting and distribution

---

*This platform is intended for informational and benchmarking purposes. Methodology and data inputs are continuously refined to ensure accuracy and investability.*
"""

DEBUG = False

def render_global_styles():
    if DEBUG:
        st.write(os.listdir())

    st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .ticker {
        white-space: nowrap;
        overflow: hidden;
        box-sizing: border-box;
    }
    .ticker span {
        display: inline-block;
        padding-left: 100%;
        animation: ticker 25s linear infinite;
    }
    @keyframes ticker {
        0%   { transform: translate(0, 0); }
        100% { transform: translate(-100%, 0); }
    }
    /* Background */
    body {
        background-color: #0B0F14;
    }

    /* Main container tighter */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0rem !important;
    }

    /* Remove vertical gaps */
    .element-container {
        margin-bottom: 0.3rem !important;
    }

    /* Headings smaller + tighter */
    h1 {
        font-size: 20px !important;
    }
    h2 {
        font-size: 16px !important;
        margin-bottom: 0.2rem !important;
    }
    h3 {
        font-size: 14px !important;
    }

    /* Captions = chart titles */
    .caption {
        font-size: 11px !important;
        color: #9AA4AF;
    }

    /* Columns tighter */
    div[data-testid="column"] {
        padding-left: 0.3rem !important;
        padding-right: 0.3rem !important;
    }

    .kpi-card {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(154, 164, 175, 0.18);
        border-radius: 8px;
        min-height: 68px;
        padding: 0.45rem 0.6rem;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    .kpi-label {
        font-size: 13px;
        color: #9CA3AF;
        line-height: 1.1;
        margin-bottom: 0.15rem;
    }

    .kpi-value {
        font-size: 28px;
        font-weight: 600;
        line-height: 1.1;
    }

    .metric-label {
        font-size: 13px;
        color: #9CA3AF;
    }

    .metric-value {
        font-size: 28px;
        font-weight: 600;
    }

    .section-header {
        font-size: 18px;
        font-weight: 600;
        margin-top: 20px;
    }

    .small-note {
        font-size: 12px;
        color: #6B7280;
    }

    img {
        object-fit: contain !important;
    }

    </style>
    """, unsafe_allow_html=True)
    st.markdown("""
    <style>

    @media print {
        /* Hide sidebar */
        section[data-testid="stSidebar"] {
            display: none;
        }

        /* Remove padding */
        .main {
            padding: 0 !important;
        }

        /* Force white background */
        body {
        background: white !important;
        color: black !important;
    }

    /* Charts spacing */
    .element-container {
        margin-bottom: 10px !important;
    }

    /* Prevent page breaks inside charts */
    .element-container {
        page-break-inside: avoid;
    }
}

</style>
""", unsafe_allow_html=True)
plt.style.use("dark_background")
plt.rcParams.update({
    "figure.figsize": (10, 4),
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "font.family": "sans-serif",
    "figure.facecolor": "#0E1117",
    "axes.facecolor": "#0E1117",
    "savefig.facecolor": "#0E1117",
    "text.color": "white",
    "axes.labelcolor": "white",
    "axes.titlecolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
})

CHART_COLORS = {
    "AM100": "#4A90E2",
    "AM200": "#50E3C2",
    "AM300": "#F5A623",
    "Conservative": "#7ED321",
    "Balanced": "#BD10E0",
    "Growth": "#FF6F61",
}


def style_dark_axes(ax):
    ax.set_facecolor("#0E1117")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    ax.grid(True, linestyle="--", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def style_chart(fig, ax):
    ax.set_facecolor("#0b0f14")
    fig.patch.set_facecolor("#0b0f14")
    ax.grid(True, alpha=0.1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#444")
    ax.spines["bottom"].set_color("#444")
    ax.tick_params(colors="#aaa")
    ax.xaxis.label.set_color("#9AA4AF")
    ax.yaxis.label.set_color("#9AA4AF")
    ax.title.set_color("#E6EDF3")
    return fig, ax


# ----------------------------
# LOAD DATA
# ----------------------------
@st.cache_data
def load_total_return(name, _version=None):
    path = f"output/{name}_total_return.csv"
    if not os.path.exists(path):
        raise ValueError(f"No data file found for {name}: {path}")

    df = load_index(name)
    validate_df(df, f"{name} total return file", ["Date", "Index Level"])

    return validate_series(df.set_index("Date")["Index Level"], f"{name} total return series")


def load_data():
    am100_path = "output/AM100_total_return.csv"
    am200_path = "output/AM200_total_return.csv"
    am300_path = "output/AM300_total_return.csv"
    am100 = load_total_return("AM100", get_file_version(am100_path))
    am200 = load_total_return("AM200", get_file_version(am200_path))
    am300 = load_total_return("AM300", get_file_version(am300_path))

    return am100, am200, am300


am100, am200, am300 = load_data()
am100_series = am100.copy()
am200_series = am200.copy()
am300_series = am300.copy()


def normalize_comparison_metrics(raw_metrics):
    if isinstance(raw_metrics, dict):
        return {
            str(index_name): metric_row
            for index_name, metric_row in raw_metrics.items()
            if isinstance(metric_row, dict)
        }
    if isinstance(raw_metrics, pd.DataFrame) and "Index" in raw_metrics.columns:
        return raw_metrics.set_index("Index").to_dict(orient="index")
    return {}


common_analysis = get_common_analysis_window(
    {"AM100": am100_series, "AM200": am200_series, "AM300": am300_series}
)
common_start = common_analysis["start"]
common_end = common_analysis["end"]
common_period_label = common_analysis["label"]
comparison_series = common_analysis["series"]
comparison_metrics = normalize_comparison_metrics(
    {name: compute_window_metrics(series) for name, series in comparison_series.items()}
    if isinstance(comparison_series, dict)
    else {}
)


@st.cache_data
def load_history(am100_version=None, am200_version=None, am300_version=None):
    am100_hist = pd.read_excel("output/AM100_history.xlsx")
    am200_hist = pd.read_excel("output/AM200_history.xlsx")
    am300_hist = pd.read_excel("output/AM300_history.xlsx")
    return am100_hist, am200_hist, am300_hist


@st.cache_data
def load_price_panel(_version=None):
    path = "output/master.csv"
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["Date"])
    else:
        df = pd.read_excel("output/master.xlsx")
        df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").set_index("Date")


am100_hist, am200_hist, am300_hist = load_history(
    get_file_version("output/AM100_history.xlsx"),
    get_file_version("output/AM200_history.xlsx"),
    get_file_version("output/AM300_history.xlsx"),
)
price_panel = load_price_panel(
    get_file_version("output/master.csv")
    if os.path.exists("output/master.csv")
    else get_file_version("output/master.xlsx")
)
MIN_POSITIVE_COVERAGE = 0.80
volume_audit_df, validated_securities_count, eligible_securities_count = get_volume_audit(price_panel)
no_volume_df = volume_audit_df[volume_audit_df["Positive Days"] == 0].copy()
missing_volume_df = volume_audit_df[volume_audit_df["Missing Days"] == volume_audit_df["Total Days"]].copy()
zero_only_volume_df = volume_audit_df[
    (volume_audit_df["Missing Days"] == 0)
    & (volume_audit_df["Positive Days"] == 0)
    & (volume_audit_df["Zero Days"] > 0)
].copy()
low_volume_coverage_df = volume_audit_df[
    volume_audit_df["Positive Coverage %"] < MIN_POSITIVE_COVERAGE * 100
].copy()
eligible_volume_df = volume_audit_df[
    volume_audit_df["Positive Coverage %"] >= MIN_POSITIVE_COVERAGE * 100
].copy()
full_trading_coverage_count = int((volume_audit_df["Positive Coverage %"] == 100).sum())
ineligible_securities_count = len(low_volume_coverage_df)
assert validated_securities_count > 0, "Validated count is zero — pipeline failure"
assert eligible_securities_count > 0, "Eligible count is zero — filter failure"
NO_VOLUME_EXPORT_FILE = "output/missing_volume_companies.csv"
LOW_COVERAGE_EXPORT_FILE = "output/low_coverage_companies.csv"
no_volume_df.to_csv(NO_VOLUME_EXPORT_FILE, index=False)
low_volume_coverage_df.to_csv(LOW_COVERAGE_EXPORT_FILE, index=False)
latest_date = am100_hist["Date"].max()

am100_latest = am100_hist[am100_hist["Date"] == latest_date]
am200_latest = am200_hist[am200_hist["Date"] == latest_date]
am300_latest = am300_hist[am300_hist["Date"] == latest_date]

am100_country_df = compute_country_weights(am100_latest)
am100_adv_usd, am100_capacity = external_capacity_metrics(am100_latest)
am200_adv_usd, am200_capacity = external_capacity_metrics(am200_latest)
am300_adv_usd, am300_capacity = external_capacity_metrics(am300_latest)

am100_constituents = am100_latest["Company"].tolist()
am200_constituents = am200_latest["Company"].tolist()
am300_constituents = am300_latest["Company"].tolist()

am100_periods = get_periods(price_panel, am100_constituents, am100)
am200_periods = get_periods(price_panel, am200_constituents, am200)
am300_periods = get_periods(price_panel, am300_constituents, am300)

cagr_outputs = {
    "AM100": compute_cagr_for_periods(am100, am100_periods),
    "AM200": compute_cagr_for_periods(am200, am200_periods),
    "AM300": compute_cagr_for_periods(am300, am300_periods),
}

if "fixed_10y" not in cagr_outputs["AM100"]:
    raise ValueError("fixed_10y missing from AM100 CAGR outputs")
if "fixed_5y" not in cagr_outputs["AM100"]:
    raise ValueError("fixed_5y missing from AM100 CAGR outputs")
if "fixed_10y" not in cagr_outputs["AM200"]:
    raise ValueError("fixed_10y missing from AM200 CAGR outputs")
if "fixed_10y" not in cagr_outputs["AM300"]:
    raise ValueError("fixed_10y missing from AM300 CAGR outputs")

analytics_coverage_notes = []

for index_name, index_series, periods, period_results in [
    ("AM100", am100, am100_periods, cagr_outputs["AM100"]),
    ("AM200", am200, am200_periods, cagr_outputs["AM200"]),
    ("AM300", am300, am300_periods, cagr_outputs["AM300"]),
]:
    latest_valid = periods["latest_valid"]
    index_latest = index_series.index.max()
    gap_days = (index_latest - latest_valid).days
    if gap_days > 0:
        analytics_coverage_notes.append(
            f"{index_name}: analytics capped at {latest_valid.date()} due to incomplete constituent data"
        )
        if gap_days > 5:
            analytics_coverage_notes.append(
                f"{index_name}: recent data incomplete - analytics truncated"
            )
    for rolling_key in ["rolling_10y", "rolling_5y", "rolling_3y", "rolling_1y"]:
        rolling_period = periods[rolling_key]
        rolling_result = period_results[rolling_key]
        if rolling_result["status"] != "OK":
            analytics_coverage_notes.append(
                f"{index_name}: {rolling_key.replace('_', ' ')} unavailable due to insufficient history"
            )
            continue
        if rolling_period["approximate"]:
            analytics_coverage_notes.append(
                f"{index_name}: {rolling_key.replace('_', ' ')} uses nearest available date within 30-day tolerance"
            )

# Latest returns (daily)
if len(am100) > 1:
    ret100 = am100.pct_change().iloc[-1]
else:
    ret100 = 0

if len(am200) > 1:
    ret200 = am200.pct_change().iloc[-1]
else:
    ret200 = 0

if len(am300) > 1:
    ret300 = am300.pct_change().iloc[-1]
else:
    ret300 = 0


# ----------------------------
# METRICS
# ----------------------------
@st.cache_data
def load_metric_snapshot(path, _version=None):
    if "AM100" in path:
        name = "AM100"
    elif "AM200" in path:
        name = "AM200"
    else:
        name = "AM300"
    df = load_index(name)
    index_series = df["Index Level"]
    returns = index_series.pct_change().dropna()

    start = index_series.iloc[0]
    end = index_series.iloc[-1]
    years = (df["Date"].iloc[-1] - df["Date"].iloc[0]).days / 365

    cagr = (end / start) ** (1 / years) - 1
    vol = returns.std() * np.sqrt(252)
    annual_return = returns.mean() * 252
    rf = 0.02
    sharpe = (annual_return - rf) / vol if vol != 0 else 0
    drawdown = calculate_drawdown(index_series)
    max_dd = drawdown.min()

    return cagr, vol, sharpe, max_dd


def color_return(val):
    if val > 0:
        return "color: #00FF9C"
    elif val < 0:
        return "color: #FF4D4D"
    else:
        return "color: #AAAAAA"


def display_with_row_numbers(df):
    df_display = df.reset_index(drop=True)
    df_display.index = df_display.index + 1
    return df_display

def generate_pdf(cagr100, cagr200):
    output_path = "output/AM_report.pdf"
    doc = SimpleDocTemplate(output_path)
    styles = getSampleStyleSheet()

    content = []
    cagr300, _, _, _ = load_metric_snapshot(
        "output/AM300_total_return.csv",
        get_file_version("output/AM300_total_return.csv"),
    )
    content.append(Paragraph("AM100 / AM200 / AM300 Report", styles["Title"]))
    content.append(Paragraph(f"CAGR AM100: {cagr100:.2%}", styles["Normal"]))
    content.append(Paragraph(f"CAGR AM200: {cagr200:.2%}", styles["Normal"]))
    content.append(Paragraph(f"CAGR AM300: {cagr300:.2%}", styles["Normal"]))

    doc.build(content)

    with open(output_path, "rb") as pdf_file:
        return pdf_file.read()


# ----------------------------
# SIDEBAR
# ----------------------------
if get_selected_series() == "AM Series":
    st.sidebar.title("Controls")
    start_date = st.sidebar.date_input("Start Date", am100_series.index.min())
    end_date = st.sidebar.date_input("End Date", am100_series.index.max())
else:
    start_date = am100_series.index.min()
    end_date = am100_series.index.max()

# Filter data
am100 = am100_series[
    (am100_series.index >= pd.to_datetime(start_date))
    & (am100_series.index <= pd.to_datetime(end_date))
]
am200 = am200_series[
    (am200_series.index >= pd.to_datetime(start_date))
    & (am200_series.index <= pd.to_datetime(end_date))
]
am300 = am300_series[
    (am300_series.index >= pd.to_datetime(start_date))
    & (am300_series.index <= pd.to_datetime(end_date))
]

am100_df = am100.rename("Index Level").reset_index()
am200_df = am200.rename("Index Level").reset_index()
am300_df = am300.rename("Index Level").reset_index()

def render_am_series():
    render_global_styles()
    render_static_header()
    render_data_header(validated_securities_count, eligible_securities_count)

    local_common = get_common_analysis_window(
        {"AM100": am100_series, "AM200": am200_series, "AM300": am300_series}
    )
    local_comparison_metrics = normalize_comparison_metrics(
        {
            name: compute_window_metrics(series)
            for name, series in local_common["series"].items()
        }
        if isinstance(local_common.get("series"), dict)
        else {}
    )

    def header_comparison_metric(index_name):
        if not isinstance(local_comparison_metrics, dict):
            return {}
        return local_comparison_metrics.get(index_name, {})

    def header_pct_metric(metrics, key):
        value = metrics.get(key) if metrics else None
        return f"{value * 100:.2f}%" if value is not None and pd.notna(value) else "N/A"

    def header_num_metric(metrics, key):
        value = metrics.get(key) if metrics else None
        return f"{value:.2f}" if value is not None and pd.notna(value) else "N/A"

    am100_header_compare = header_comparison_metric("AM100")
    am200_header_compare = header_comparison_metric("AM200")
    am300_header_compare = header_comparison_metric("AM300")

    am_header_metrics = pd.DataFrame(
        {
            "Metric": ["Constituents", "CAGR", "Volatility", "Sharpe", "Max Drawdown"],
            "AM100": [
                am100_snapshot_insights.get("constituents"),
                header_pct_metric(am100_header_compare, "CAGR"),
                header_pct_metric(am100_header_compare, "Volatility"),
                header_num_metric(am100_header_compare, "Sharpe"),
                header_pct_metric(am100_header_compare, "Max Drawdown"),
            ],
            "AM200": [
                am200_snapshot_insights.get("constituents"),
                header_pct_metric(am200_header_compare, "CAGR"),
                header_pct_metric(am200_header_compare, "Volatility"),
                header_num_metric(am200_header_compare, "Sharpe"),
                header_pct_metric(am200_header_compare, "Max Drawdown"),
            ],
            "AM300": [
                am300_snapshot_insights.get("constituents"),
                header_pct_metric(am300_header_compare, "CAGR"),
                header_pct_metric(am300_header_compare, "Volatility"),
                header_num_metric(am300_header_compare, "Sharpe"),
                header_pct_metric(am300_header_compare, "Max Drawdown"),
            ],
        }
    )
    st.dataframe(am_header_metrics, use_container_width=True, hide_index=True)
    st.caption(
        "This table uses the shared overlapping period across AM100, AM200, and AM300 so the three series are directly comparable."
    )
    st.markdown(
        """
        **INTERPRETATION**

        • AM100 = Institutional core (highest liquidity)  
        • AM200 = Expanded investable universe  
        • AM300 = Broader frontier exposure with still-observable trading activity  

        Performance dispersion reflects liquidity constraints, not methodology changes.
        """
    )

    st.markdown("---")
    with st.expander("📘 Methodology & Overview", expanded=False):
        st.markdown(methodology_text)
    st.write("Last data date:", am100_df["Date"].max())

    # Visual-only smoothing for plotting. This does not affect stored data or metrics.
    am100_plot = am100.copy()
    am200_plot = am200.copy()
    am200_plot = am200_plot.interpolate()


    st.caption("Key Insights")

    col1, col2, col3 = st.columns(3)

    col1.info("AM100: Concentrated, institutional-grade core index")
    col2.info("AM200: Expansion sleeve for broader frontier exposure")
    col3.info("AM300: All-share flagship total return benchmark")

    # ----------------------------
    # METRICS DISPLAY
    # ----------------------------
    cagr100, vol100, sharpe100, dd100 = load_metric_snapshot(
        "output/AM100_total_return.csv",
        get_file_version("output/AM100_total_return.csv"),
    )
    cagr200, vol200, sharpe200, dd200 = load_metric_snapshot(
        "output/AM200_total_return.csv",
        get_file_version("output/AM200_total_return.csv"),
    )
    cagr300, vol300, sharpe300, dd300 = load_metric_snapshot(
        "output/AM300_total_return.csv",
        get_file_version("output/AM300_total_return.csv"),
    )

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    col1.markdown(
        f"""
    <div class="kpi-card">
        <div class="kpi-label">AM100</div>
        <div class="kpi-value" style="{color_return(ret100)}">{ret100:.2%}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    col2.markdown(
        f"""
    <div class="kpi-card">
        <div class="kpi-label">AM200</div>
        <div class="kpi-value" style="{color_return(ret200)}">{ret200:.2%}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    col3.markdown(
        f"""
    <div class="kpi-card">
        <div class="kpi-label">AM300</div>
        <div class="kpi-value" style="{color_return(ret300)}">{ret300:.2%}</div>
    </div>
    """
    ,
        unsafe_allow_html=True,
    )

    col4.markdown(
        f"""
    <div class="kpi-card">
        <div class="kpi-label">Vol (AM100)</div>
        <div class="kpi-value">{vol100:.2%}</div>
    </div>
    """
    ,
        unsafe_allow_html=True,
    )

    col5.markdown(
        f"""
    <div class="kpi-card">
        <div class="kpi-label">Vol (AM200)</div>
        <div class="kpi-value">{vol200:.2%}</div>
    </div>
    """
    ,
        unsafe_allow_html=True,
    )

    col6.markdown(
        f"""
    <div class="kpi-card">
        <div class="kpi-label">Vol (AM300)</div>
        <div class="kpi-value">{vol300:.2%}</div>
    </div>
    """
    ,
        unsafe_allow_html=True,
    )

    st.caption("Performance Summary")

    # =========================
    # CAGR SECTION (STATIC)
    # =========================

    st.markdown("### CAGR by Period")

    common_window_display = common_period_label
    am100_outputs = cagr_outputs.get("AM100", {})

    fixed_10y = am100_outputs.get("fixed_10y")
    if fixed_10y is None:
        raise ValueError(
            f"AM100 fixed_10y missing. Available keys: {list(am100_outputs.keys())}"
        )

    fixed_5y = am100_outputs.get("fixed_5y")
    if fixed_5y is None:
        raise ValueError(
            f"AM100 fixed_5y missing. Available keys: {list(am100_outputs.keys())}"
        )

    fixed_10y_display = safe_period_range(fixed_10y)
    fixed_5y_display = safe_period_range(fixed_5y)

    st.markdown(
        f"""
        ### Performance Windows

        **Common Window (Comparable Across Indices)**  
        {common_window_display}  

        **Fixed Windows (Index-Specific History)**  
        • 10 Year: {fixed_10y_display}  
        • 5 Year: {fixed_5y_display}
        """
    )

    latest_valid_dates = {
        "AM100": am100_periods["latest_valid"],
        "AM200": am200_periods["latest_valid"],
        "AM300": am300_periods["latest_valid"],
    }
    min_valid = min(latest_valid_dates.values())

    st.markdown(
        f"<div class='small-note'>Analytics calculated using data through {min_valid.date()} based on full constituent coverage.</div>",
        unsafe_allow_html=True,
    )

    with st.expander("Data Coverage Details"):
        for index_name, valid_date in latest_valid_dates.items():
            st.write(f"{index_name}: {valid_date.date()}")
        requested_rolling_start = am100_periods["rolling_10y"]["requested_start"]
        actual_rolling_start = am100_periods["rolling_10y"]["start"]
        rolling_end = am100_periods["rolling_10y"]["end"]
        st.write("Rolling requested start:", requested_rolling_start)
        st.write("Rolling actual start used:", actual_rolling_start)
        st.write("Rolling end date:", rolling_end)
        if actual_rolling_start is not None:
            st.write(
                "10Y Rolling:",
                f"{actual_rolling_start.strftime('%d %b %Y')} -> {rolling_end.strftime('%d %b %Y')}",
            )
        st.write("Shared comparison start:", common_start)
        if actual_rolling_start is not None:
            window_length = (rolling_end - actual_rolling_start).days / 365
            st.write("Rolling window length (years):", round(window_length, 3))
            rolling_gap_days = (actual_rolling_start - requested_rolling_start).days
            if rolling_gap_days > 7:
                st.write("10Y rolling status:", "Approximate 10Y")


    def color(val):
        if val is None:
            return "white"
        return "lime" if val > 0 else "red"


    def styled(val):
        if val is None:
            return "-"
        return f"<span style='color:{color(val)}'>{val:.2%}</span>"


    rolling_results = {
        "AM100": {
            "rolling_10y": cagr_outputs["AM100"]["rolling_10y"],
            "rolling_5y": cagr_outputs["AM100"]["rolling_5y"],
            "rolling_3y": cagr_outputs["AM100"]["rolling_3y"],
            "rolling_1y": cagr_outputs["AM100"]["rolling_1y"],
        },
        "AM200": {
            "rolling_10y": cagr_outputs["AM200"]["rolling_10y"],
            "rolling_5y": cagr_outputs["AM200"]["rolling_5y"],
            "rolling_3y": cagr_outputs["AM200"]["rolling_3y"],
            "rolling_1y": cagr_outputs["AM200"]["rolling_1y"],
        },
        "AM300": {
            "rolling_10y": cagr_outputs["AM300"]["rolling_10y"],
            "rolling_5y": cagr_outputs["AM300"]["rolling_5y"],
            "rolling_3y": cagr_outputs["AM300"]["rolling_3y"],
            "rolling_1y": cagr_outputs["AM300"]["rolling_1y"],
        },
    }

    index_series_map = {
        "AM100": am100_series,
        "AM200": am200_series,
        "AM300": am300_series,
    }

    comparison_period_result = {
        name: compute_performance(
            index_series_map[name],
            "fixed",
            name=name,
            start_date=common_start,
            end_date=common_end,
        )
        if common_start is not None and common_end is not None and name in comparison_series
        else {"status": "NO_DATA", "cagr": None, "start": None, "end": None}
        for name in index_series_map
    }

    for index_name, common_result in comparison_period_result.items():
        if index_name in cagr_outputs:
            cagr_outputs[index_name]["common"] = common_result

    fixed_results = {
        "AM100": {
            "common_period": cagr_outputs["AM100"]["common"],
            "fixed_10y": cagr_outputs["AM100"]["fixed_10y"],
            "fixed_5y": cagr_outputs["AM100"]["fixed_5y"],
        },
        "AM200": {
            "common_period": cagr_outputs["AM200"]["common"],
            "fixed_10y": cagr_outputs["AM200"]["fixed_10y"],
            "fixed_5y": cagr_outputs["AM200"]["fixed_5y"],
        },
        "AM300": {
            "common_period": cagr_outputs["AM300"]["common"],
            "fixed_10y": cagr_outputs["AM300"]["fixed_10y"],
            "fixed_5y": cagr_outputs["AM300"]["fixed_5y"],
        },
    }
    with st.expander("Rolling Performance Analysis"):
        for index_name, results in rolling_results.items():
            st.markdown(f"**{index_name}**")
            st.caption("Trailing annualised returns over the latest valid 10Y, 5Y, 3Y, and 1Y windows.")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("10Y", safe_metric(results["rolling_10y"]))
            col2.metric("5Y", safe_metric(results["rolling_5y"]))
            col3.metric("3Y", safe_metric(results["rolling_3y"]))
            col4.metric("1Y", safe_metric(results["rolling_1y"]))
            st.markdown(
                f"<div class='small-note'>"
                f"10Y: {safe_period_range(results['rolling_10y'])} | "
                f"5Y: {safe_period_range(results['rolling_5y'])} | "
                f"3Y: {safe_period_range(results['rolling_3y'])} | "
                f"1Y: {safe_period_range(results['rolling_1y'])}"
                f"</div>",
                unsafe_allow_html=True,
            )

    comparison_df = pd.DataFrame(
        {
            "Metric": ["CAGR", "Volatility", "Sharpe", "Max Drawdown"],
            "AM100": [f"{cagr100:.2%}", f"{vol100:.2%}", f"{sharpe100:.2f}", f"{dd100:.2%}"],
            "AM200": [f"{cagr200:.2%}", f"{vol200:.2%}", f"{sharpe200:.2f}", f"{dd200:.2%}"],
            "AM300": [f"{cagr300:.2%}", f"{vol300:.2%}", f"{sharpe300:.2f}", f"{dd300:.2%}"],
        }
    )
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    st.caption(
        "These headline metrics use each index's full available history, so they may differ from the shared-period comparison shown above."
    )


    def metric_help_text(metric_name, period_label):
        if metric_name == "CAGR":
            return (
                "Compound Annual Growth Rate.\n\n"
                "Calculated as:\n"
                "(end value / start value)^(1/years) - 1\n\n"
                f"Period:\n{period_label}\n\n"
                "Includes:\n"
                "✔ Dividends (total return)\n"
                "✔ FX-normalised (USD)\n\n"
                "Excludes:\n"
                "✖ Fees\n"
                "✖ Slippage"
            )
        if metric_name == "Volatility":
            return (
                "Annualised standard deviation of returns.\n\n"
                "Measures risk over the observed return path.\n\n"
                "Includes:\n"
                "✔ Total return moves\n"
                "✔ FX-normalised (USD)\n\n"
                "Excludes:\n"
                "✖ Fees\n"
                "✖ Slippage"
            )
        if metric_name == "Sharpe Ratio":
            return (
                "Return per unit of risk.\n\n"
                "Calculated from annualised return and annualised volatility with a 2% risk-free rate assumption."
            )
        if metric_name == "Max Drawdown":
            return "Largest peak-to-trough decline over the period."
        return ""


    headline_period_label = common_period_label

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("AM100")
        st.metric(
            label="CAGR ℹ️",
            value=f"{cagr100:.2%}",
            help=metric_help_text("CAGR", headline_period_label),
        )
        st.metric(
            label="Volatility ℹ️",
            value=f"{vol100:.2%}",
            help=metric_help_text("Volatility", headline_period_label),
        )
        st.metric(
            label="Sharpe Ratio ℹ️",
            value=f"{sharpe100:.2f}",
            help=metric_help_text("Sharpe Ratio", headline_period_label),
        )
        st.metric(
            label="Max Drawdown ℹ️",
            value=f"{dd100:.2%}",
            help=metric_help_text("Max Drawdown", headline_period_label),
        )

    with col2:
        st.write("AM200")
        st.metric(
            label="CAGR ℹ️",
            value=f"{cagr200:.2%}",
            help=metric_help_text("CAGR", headline_period_label),
        )
        st.metric(
            label="Volatility ℹ️",
            value=f"{vol200:.2%}",
            help=metric_help_text("Volatility", headline_period_label),
        )
        st.metric(
            label="Sharpe Ratio ℹ️",
            value=f"{sharpe200:.2f}",
            help=metric_help_text("Sharpe Ratio", headline_period_label),
        )
        st.metric(
            label="Max Drawdown ℹ️",
            value=f"{dd200:.2%}",
            help=metric_help_text("Max Drawdown", headline_period_label),
        )

    with col3:
        st.write("AM300")
        st.metric(
            label="CAGR ℹ️",
            value=f"{cagr300:.2%}",
            help=metric_help_text("CAGR", headline_period_label),
        )
        st.metric(
            label="Volatility ℹ️",
            value=f"{vol300:.2%}",
            help=metric_help_text("Volatility", headline_period_label),
        )
        st.metric(
            label="Sharpe Ratio ℹ️",
            value=f"{sharpe300:.2f}",
            help=metric_help_text("Sharpe Ratio", headline_period_label),
        )
        st.metric(
            label="Max Drawdown ℹ️",
            value=f"{dd300:.2%}",
            help=metric_help_text("Max Drawdown", headline_period_label),
        )

    st.caption("Key Observations")

    obs = []

    if cagr200 > cagr100:
        obs.append(
            "AM200 is outperforming AM100, indicating stronger growth in mid-cap and frontier segments."
        )

    if vol200 > vol100:
        obs.append(
            "AM200 exhibits higher volatility, reflecting increased exposure to emerging markets."
        )

    if dd200 < dd100:
        obs.append("AM200 has experienced deeper drawdowns, highlighting higher risk.")

    top_country_am100 = get_top_country(am100_latest)
    top_country_am200 = get_top_country(am200_latest)
    top_country_am300 = get_top_country(am300_latest) if SHOW_AM300 else None
    am100_country = get_country_weights(am100_latest)
    am200_country = get_country_weights(am200_latest)
    am300_country = get_country_weights(am300_latest) if SHOW_AM300 else pd.Series(dtype=float)
    am100_country_df = compute_country_weights(am100_latest)
    am200_country_df = compute_country_weights(am200_latest)
    latest_am100 = am100.iloc[-1]
    latest_am200 = am200.iloc[-1] if len(am200) else None
    latest_am300 = am300.iloc[-1] if SHOW_AM300 and len(am300) else None

    ticker_parts = [
        f"AM100: {latest_am100:.2f} ({ret100:.2%})",
        f"Top Country AM100: {safe_display(top_country_am100)}",
    ]
    if latest_am200 is not None:
        ticker_parts.append(f"AM200: {latest_am200:.2f} ({ret200:.2%})")
        ticker_parts.append(f"Top Country AM200: {safe_display(top_country_am200)}")
    if SHOW_AM300 and latest_am300 is not None:
        ticker_parts.append(f"AM300: {latest_am300:.2f} ({ret300:.2%})")
        ticker_parts.append(f"Top Country AM300: {safe_display(top_country_am300)}")
    ticker_text = " •\n".join(ticker_parts)

    if top_country_am100:
        obs.append(f"AM100 is most exposed to {top_country_am100}.")
    if top_country_am200:
        obs.append(f"AM200 shows strongest exposure to {top_country_am200}.")
    if SHOW_AM300 and top_country_am300:
        obs.append(f"AM300 shows flagship concentration in {top_country_am300}.")

    @st.cache_data
    def load_capacity_usd_file(path, _version=None):
        return pd.read_csv(path, parse_dates=["Date"]).sort_values("Date")


    @st.cache_data
    def load_investability_map(_version=None):
        path = "output/investability_map.csv"
        if not os.path.exists(path):
            return None
        return pd.read_csv(path)


    investability_map = load_investability_map(get_file_version("output/investability_map.csv"))

    if allocator_mode and all(
        x is not None
        for x in [
            am100_metrics,
            am200_metrics,
            am300_metrics,
            am100_snapshot_insights,
            am200_snapshot_insights,
            am300_snapshot_insights,
        ]
    ):
        render_allocator_view()
        st.stop()

    am100_s, am100_r = prep(am100_df)
    am200_s, am200_r = prep(am200_df)
    am300_s, am300_r = prep(am300_df)

    risk_ratings_path = "output/index_risk_ratings.csv"
    benchmark_metrics_path = "output/am100_vs_msci_africa.csv"
    benchmark_series_path = "output/msci_africa.csv"
    drivers_path = "output/am100_volatility_drivers.csv"

    rolling_vol_100 = rolling_volatility(am100_r)
    rolling_vol_200 = rolling_volatility(am200_r)
    rolling_vol_300 = rolling_volatility(am300_r)

    rolling_mean_100 = am100_r.rolling(30).mean() * 252
    rolling_mean_200 = am200_r.rolling(30).mean() * 252
    rolling_mean_300 = am300_r.rolling(30).mean() * 252

    rolling_sharpe_100 = (rolling_mean_100 - 0.02) / rolling_vol_100
    rolling_sharpe_200 = (rolling_mean_200 - 0.02) / rolling_vol_200
    rolling_sharpe_300 = (rolling_mean_300 - 0.02) / rolling_vol_300

    rolling_perf_1y_100 = am100_s.pct_change(252).dropna()
    rolling_perf_1y_200 = am200_s.pct_change(252).dropna()
    rolling_perf_1y_300 = am300_s.pct_change(252).dropna()

    drawdown_100 = calculate_drawdown(am100_s)
    drawdown_200 = calculate_drawdown(am200_s)
    drawdown_300 = calculate_drawdown(am300_s)

    dashboard_outputs = build_dashboard_outputs(
        am100_df=am100_df,
        am200_df=am200_df,
        am300_df=am300_df,
        rolling_perf_1y_100=rolling_perf_1y_100,
        rolling_perf_1y_200=rolling_perf_1y_200,
        rolling_perf_1y_300=rolling_perf_1y_300,
        rolling_vol_100=rolling_vol_100,
        rolling_vol_200=rolling_vol_200,
        rolling_vol_300=rolling_vol_300,
        drawdown_100=drawdown_100,
        drawdown_200=drawdown_200,
        drawdown_300=drawdown_300,
        am100_latest=am100_latest,
        am200_latest=am200_latest,
        am300_latest=am300_latest,
        stats_dict={
            "AM100": {"CAGR": cagr100, "Volatility": vol100, "Sharpe": sharpe100, "Max_Drawdown": dd100},
            "AM200": {"CAGR": cagr200, "Volatility": vol200, "Sharpe": sharpe200, "Max_Drawdown": dd200},
            "AM300": {"CAGR": cagr300, "Volatility": vol300, "Sharpe": sharpe300, "Max_Drawdown": dd300},
        },
    )
    perf = validate_df(
        dashboard_outputs["performance"], "performance", ["Date", "Index", "Index_Level"]
    )
    roll = validate_df(
        dashboard_outputs["rolling"], "rolling", ["Date", "Index", "Metric", "Value"]
    )
    dd = validate_df(
        dashboard_outputs["drawdown"], "drawdown", ["Date", "Index", "Metric", "Value"]
    )
    constituents = validate_df(
        dashboard_outputs["constituents"],
        "constituents",
        ["Index", "Company", "Country", "Weight"],
    )

    for o in obs:
        st.write(f"• {o}")

    st.markdown(
        f"<div class='ticker'><span>{ticker_text}</span></div>",
        unsafe_allow_html=True,
    )

    allocator_levels = pd.concat(
        [
            am100.rename("AM100"),
            am200.rename("AM200"),
            am300.rename("AM300"),
        ],
        axis=1,
        join="inner",
    ).dropna()
    allocator_returns = prepare_returns_frame(allocator_levels)
    high_risk_result = optimize_high_risk_portfolio(allocator_returns)
    MODEL_WEIGHTS = build_model_weight_system(high_risk_result["Weights"])
    model_levels, model_returns = build_model_series(
        allocator_levels,
        model_weights=MODEL_WEIGHTS,
    )
    model_metrics = build_model_metrics_table(
        model_levels,
        model_returns,
        {"AM100": am100_capacity, "AM200": am200_capacity, "AM300": am300_capacity},
        model_weights=MODEL_WEIGHTS,
    )

    overview_tab, risk_tab, allocator_tab = st.tabs(["Overview", "Risk", "Allocator"])

    with overview_tab:
        st.caption("Overview")
        st.write(
            "Use the Risk tab for the full institutional risk view: summary metrics, "
            "risk ratings, rolling stability, benchmark comparison, and volatility decomposition."
        )

    with risk_tab:
        st.markdown("## Risk Summary")
        risk_summary_df = pd.DataFrame(
            {
                "Metric": ["Volatility", "Drawdown", "Sharpe", "Capacity", "Risk Rating"],
                "AM100": [
                    f"{vol100:.2%}",
                    f"{dd100:.2%}",
                    f"{sharpe100:.2f}",
                    f"{am100_capacity:,.0f}",
                    "-",
                ],
                "AM200": [
                    f"{vol200:.2%}",
                    f"{dd200:.2%}",
                    f"{sharpe200:.2f}",
                    f"{am200_capacity:,.0f}",
                    "-",
                ],
                "AM300": [
                    f"{vol300:.2%}",
                    f"{dd300:.2%}",
                    f"{sharpe300:.2f}",
                    f"{am300_capacity:,.0f}",
                    "-",
                ],
            }
        )

        if os.path.exists(risk_ratings_path):
            risk_ratings = load_csv_file(
                risk_ratings_path, get_file_version(risk_ratings_path)
            )
            for _, row in risk_ratings.iterrows():
                risk_summary_df.loc[
                    risk_summary_df["Metric"] == "Risk Rating", row["Index"]
                ] = f"{row['Rating']} ({row['Risk Score']:.2f})"
        else:
            risk_ratings = None

        st.dataframe(risk_summary_df, use_container_width=True, hide_index=True)
        st.caption(
            "Capacity is calculated assuming a 20% participation rate in average daily traded value, representing a prudent institutional execution threshold."
        )

        capacity_display = pd.DataFrame(
            {
                "Metric": [
                    "Average Daily Traded Value (USD)",
                    "Investable Capacity (USD, 20%)",
                ],
                "AM100": [f"{am100_adv_usd:,.0f}", f"{am100_capacity:,.0f}"],
                "AM200": [f"{am200_adv_usd:,.0f}", f"{am200_capacity:,.0f}"],
                "AM300": [f"{am300_adv_usd:,.0f}", f"{am300_capacity:,.0f}"],
            }
        )
        st.write(capacity_display.to_html(index=False, escape=False), unsafe_allow_html=True)
        st.caption(
            "Average Daily Traded Value (USD) represents the total value of shares traded daily across index constituents. Estimated Investable Capacity (USD, 20% participation) reflects a conservative estimate of capital that can be deployed without materially impacting market prices."
        )

        st.markdown("## Risk Rating")
        risk_cols = st.columns(3)
        am300_label = "Unavailable"
        for i, name in enumerate(INDEX_POOL):
            with risk_cols[i]:
                if risk_ratings is not None:
                    row = risk_ratings[risk_ratings["Index"] == name].iloc[0]
                    if name == "AM300":
                        am300_label = row["Rating"]
                    render_metric_card(
                        f"{name} Risk",
                        row["Rating"],
                        f"Multi-factor risk score: {row['Risk Score']:.2f}",
                    )
                    st.caption(
                        f"Score {row['Risk Score']:.2f} | Investable Capacity {row['Capacity']:,.0f}"
                    )
                else:
                    render_metric_card(f"{name} Risk", "Unavailable")

        st.markdown("### AM300 Risk Snapshot")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            render_metric_card("AM300 Volatility", f"{vol300:.2%}", "Annualised standard deviation of returns. Measures risk.")
        with col2:
            render_metric_card("AM300 Drawdown", f"{dd300:.2%}", "Largest peak-to-trough decline over the period.")
        with col3:
            render_metric_card("AM300 Sharpe", f"{sharpe300:.2f}", "Return per unit of risk. Higher indicates better risk-adjusted performance.")
        with col4:
            render_metric_card("AM300 Risk", am300_label, "Multi-factor risk label combining volatility, drawdown, liquidity, and concentration.")

        st.markdown("## Rolling Risk")

        st.subheader("Rolling 1Y Performance")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rolling_perf_1y_100.index, y=rolling_perf_1y_100, name="AM100 1Y", line=dict(color=AM100_COLOR, width=2)))
        fig.add_trace(go.Scatter(x=rolling_perf_1y_200.index, y=rolling_perf_1y_200, name="AM200 1Y", line=dict(color=AM200_COLOR, width=2)))
        fig.add_trace(go.Scatter(x=rolling_perf_1y_300.index, y=rolling_perf_1y_300, name="AM300 1Y", line=dict(color=AM300_COLOR, width=2)))
        fig.update_layout(
            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117",
            font=dict(size=10, color="#CCCCCC"),
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            yaxis_tickformat=".0%",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Rolling Volatility (30D)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rolling_vol_100.index, y=rolling_vol_100, name="AM100 Volatility", line=dict(color=AM100_COLOR, width=2)))
        fig.add_trace(go.Scatter(x=rolling_vol_200.index, y=rolling_vol_200, name="AM200 Volatility", line=dict(color=AM200_COLOR, width=2)))
        fig.add_trace(go.Scatter(x=rolling_vol_300.index, y=rolling_vol_300, name="AM300 Volatility", line=dict(color=AM300_COLOR, width=2)))
        fig.update_layout(
            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117",
            font=dict(size=10, color="#CCCCCC"),
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Rolling Sharpe Ratio")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rolling_sharpe_100.index, y=rolling_sharpe_100, name="AM100 Sharpe", line=dict(color=AM100_COLOR, width=2)))
        fig.add_trace(go.Scatter(x=rolling_sharpe_200.index, y=rolling_sharpe_200, name="AM200 Sharpe", line=dict(color=AM200_COLOR, width=2)))
        fig.add_trace(go.Scatter(x=rolling_sharpe_300.index, y=rolling_sharpe_300, name="AM300 Sharpe", line=dict(color=AM300_COLOR, width=2)))
        fig.update_layout(
            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117",
            font=dict(size=10, color="#CCCCCC"),
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Drawdown")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=drawdown_100.index, y=drawdown_100, name="AM100 Drawdown", line=dict(color=AM100_COLOR, width=2)))
        fig.add_trace(go.Scatter(x=drawdown_200.index, y=drawdown_200, name="AM200 Drawdown", line=dict(color=AM200_COLOR, width=2)))
        fig.add_trace(go.Scatter(x=drawdown_300.index, y=drawdown_300, name="AM300 Drawdown", line=dict(color=AM300_COLOR, width=2)))
        fig.update_layout(
            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117",
            font=dict(size=10, color="#CCCCCC"),
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("## Benchmark Comparison")
        if os.path.exists(benchmark_metrics_path):
            if os.path.exists(benchmark_series_path):
                benchmark_series = load_benchmark_data(
                    get_file_version(benchmark_series_path)
                )
                msci_s, msci_r = prep(benchmark_series)
                am100_s, am100_r = prep(am100_df)
                am300_s, am300_r = prep(am300_df)

                am100_stats = block(am100_s, am100_r)
                am300_stats = block(am300_s, am300_r)
                msci_stats = block(msci_s, msci_r)

                benchmark_display = pd.DataFrame(
                    [am100_stats, am300_stats, msci_stats],
                    index=["AM100", "AM300", "MSCI Africa"],
                )
                formatted_benchmark = benchmark_display.copy()
                for col in ["CAGR", "Vol", "Drawdown"]:
                    formatted_benchmark[col] = formatted_benchmark[col].map("{:.2%}".format)
                formatted_benchmark["Sharpe"] = formatted_benchmark["Sharpe"].map("{:.2f}".format)
                st.dataframe(formatted_benchmark, use_container_width=True)

                am100_alpha = am100_r.mean() * 252 - msci_r.mean() * 252
                am300_alpha = am300_r.mean() * 252 - msci_r.mean() * 252
                am100_ir = info_ratio(am100_r, msci_r)
                am300_ir = info_ratio(am300_r, msci_r)
                am100_benchmark_score = advanced_risk_score(vol100, dd100, am100_alpha, am100_ir)
                am300_benchmark_score = advanced_risk_score(vol300, dd300, am300_alpha, am300_ir)

                alpha_df = pd.DataFrame(
                    {
                        "Index": ["AM100", "AM300"],
                        "Alpha": [am100_alpha, am300_alpha],
                        "Information Ratio": [am100_ir, am300_ir],
                        "Advanced Risk Score": [am100_benchmark_score, am300_benchmark_score],
                        "Risk Label": [risk_label(am100_benchmark_score), risk_label(am300_benchmark_score)],
                    }
                )
                alpha_display = alpha_df.copy()
                alpha_display["Alpha"] = alpha_display["Alpha"].map("{:.2%}".format)
                alpha_display["Information Ratio"] = alpha_display["Information Ratio"].map("{:.2f}".format)
                alpha_display["Advanced Risk Score"] = alpha_display["Advanced Risk Score"].map("{:.2f}".format)
                st.dataframe(alpha_display, use_container_width=True, hide_index=True)

                benchmark_overlay = pd.merge(
                    am100_s.rename("Index Level_AM100").reset_index(),
                    msci_s.rename("Index Level_MSCI").reset_index(),
                    on="Date",
                ).sort_values("Date")
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=benchmark_overlay["Date"],
                        y=benchmark_overlay["Index Level_AM100"],
                        name="AM100",
                        line=dict(color=AM100_COLOR, width=2),
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=benchmark_overlay["Date"],
                        y=benchmark_overlay["Index Level_MSCI"],
                        name="MSCI Africa",
                        line=dict(color="#A855F7", width=2),
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=am300_s.index,
                        y=am300_s,
                        name="AM300",
                        line=dict(color=AM300_COLOR, width=2),
                    )
                )
                fig.update_layout(
                    paper_bgcolor="#0E1117",
                    plot_bgcolor="#0E1117",
                    font=dict(size=10, color="#CCCCCC"),
                    margin=dict(l=0, r=0, t=30, b=0),
                    legend=dict(bgcolor="rgba(0,0,0,0)"),
                )
                st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Compared to traditional African benchmarks, the AM100 demonstrates improved risk-adjusted returns."
            )
        else:
            st.info("Benchmark comparison unavailable. Run scripts/benchmark_compare.py to populate this section.")

        st.markdown("## Volume Data Quality Audit")
        st.caption(
            f"Minimum internal threshold: {MIN_POSITIVE_COVERAGE:.0%} positive trading days. Missing volume indicates data failure, zero volume indicates no trading, and only positive volume days count toward liquidity eligibility."
        )

        if not volume_audit_df.empty:
            st.dataframe(volume_audit_df, use_container_width=True, hide_index=True, height=280)

            full_coverage = int((volume_audit_df["Positive Coverage %"] == 100).sum())
            cov_col1, cov_col2, cov_col3 = st.columns(3)
            cov_col1.metric("Full Positive Coverage", f"{full_coverage}")
            cov_col2.metric("Eligible (>=80%)", f"{len(eligible_volume_df)}")
            cov_col3.metric("Ineligible", f"{len(low_volume_coverage_df)}")

            if not no_volume_df.empty:
                st.error("Companies with NO Volume Data")
                st.write(sorted(no_volume_df["Security"].tolist()))

            if not zero_only_volume_df.empty:
                st.warning("Securities with ZERO volume on all observed days")
                st.write(sorted(zero_only_volume_df["Security"].tolist()))

            if not low_volume_coverage_df.empty:
                st.warning(
                    f"Companies with LOW Trading Coverage (<{MIN_POSITIVE_COVERAGE:.0%})"
                )
                df_low = pd.DataFrame(
                    sorted(low_volume_coverage_df["Security"].tolist()),
                    columns=["Company"],
                )
                df_low["Country"] = df_low["Company"].str.extract(r"\((.*?)\)")
                df_low = df_low.sort_values(["Country", "Company"], na_position="last")
                st.dataframe(
                    df_low,
                    use_container_width=True,
                    hide_index=True,
                    height=400,
                )
            st.caption(
                f"Exports written to `{NO_VOLUME_EXPORT_FILE}` and `{LOW_COVERAGE_EXPORT_FILE}`."
            )
        else:
            st.info("No volume columns were found in the master panel.")

        st.markdown("## Return Decomposition")
        st.caption("Portion of total return generated from dividends rather than price appreciation.")
        price_only_path = "output/AM100_PRICE_ONLY_total_return.csv"
        am300_price_only_path = "output/AM300_PRICE_ONLY_total_return.csv"
        if os.path.exists(price_only_path):
            price_only_df = load_index("AM100_PRICE_ONLY")
            total_return_df = load_index("AM100")

            price_only_series = price_only_df.set_index("Date")["Index Level"]
            total_return_series = total_return_df.set_index("Date")["Index Level"]

            aligned_return_df = pd.concat(
                [
                    price_only_series.rename("PriceOnly"),
                    total_return_series.rename("TotalReturn"),
                ],
                axis=1,
                join="inner",
            ).dropna()

            price_r = aligned_return_df["PriceOnly"].pct_change().dropna()
            tr_r = aligned_return_df["TotalReturn"].pct_change().dropna()
            div_r = tr_r - price_r

            price_cagr = calculate_cagr(
                aligned_return_df["PriceOnly"].rename("Index Level").reset_index(),
                aligned_return_df.index.min(),
                aligned_return_df.index.max(),
            )
            tr_cagr = calculate_cagr(
                aligned_return_df["TotalReturn"].rename("Index Level").reset_index(),
                aligned_return_df.index.min(),
                aligned_return_df.index.max(),
            )
            div_contribution = tr_cagr - price_cagr

            return_decomp_df = pd.DataFrame(
                {
                    "Component": ["Price Return", "Dividend Contribution", "Total Return"],
                    "Value": [price_cagr, div_contribution, tr_cagr],
                }
            )
            return_decomp_display = return_decomp_df.copy()
            return_decomp_display["Value"] = return_decomp_display["Value"].map("{:.2%}".format)
            st.dataframe(return_decomp_display, use_container_width=True, hide_index=True)

            fig = go.Figure()
            fig.add_trace(go.Bar(name="Price", x=["AM100"], y=[price_cagr], marker_color=AM100_COLOR))
            fig.add_trace(go.Bar(name="Dividend", x=["AM100"], y=[div_contribution], marker_color="#F59E0B"))
            fig.update_layout(
                barmode="stack",
                paper_bgcolor="#0E1117",
                plot_bgcolor="#0E1117",
                font=dict(size=10, color="#CCCCCC"),
                margin=dict(l=0, r=0, t=30, b=0),
                legend=dict(bgcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                f"Approximately {div_contribution:.2%} annual return is generated from income, not price movement."
            )
            st.caption("Dividend contribution isolates the portion of total return generated from dividends rather than price appreciation.")

            if os.path.exists(am300_price_only_path):
                am300_price_df = load_index("AM300_PRICE_ONLY")
                am300_total_df = load_index("AM300")

                am300_price_series = am300_price_df.set_index("Date")["Index Level"]
                am300_total_series = am300_total_df.set_index("Date")["Index Level"]
                am300_aligned = pd.concat(
                    [
                        am300_price_series.rename("PriceOnly"),
                        am300_total_series.rename("TotalReturn"),
                    ],
                    axis=1,
                    join="inner",
                ).dropna()
                am300_price_r = am300_aligned["PriceOnly"].pct_change().dropna()
                am300_tr_r = am300_aligned["TotalReturn"].pct_change().dropna()
                am300_div_r = am300_tr_r - am300_price_r
                am300_price_cagr = calculate_cagr(
                    am300_aligned["PriceOnly"].rename("Index Level").reset_index(),
                    am300_aligned.index.min(),
                    am300_aligned.index.max(),
                )
                am300_tr_cagr = calculate_cagr(
                    am300_aligned["TotalReturn"].rename("Index Level").reset_index(),
                    am300_aligned.index.min(),
                    am300_aligned.index.max(),
                )
                am300_div_contribution = am300_tr_cagr - am300_price_cagr

                am300_decomp = pd.DataFrame(
                    {
                        "Component": ["Price Return", "Dividend Contribution", "Total Return"],
                        "Value": [am300_price_cagr, am300_div_contribution, am300_tr_cagr],
                    }
                )
                am300_display = am300_decomp.copy()
                am300_display["Value"] = am300_display["Value"].map("{:.2%}".format)
                st.dataframe(am300_display, use_container_width=True, hide_index=True)

                fig = go.Figure()
                fig.add_trace(go.Bar(name="Price", x=["AM300"], y=[am300_price_cagr], marker_color=AM300_COLOR))
                fig.add_trace(go.Bar(name="Dividend", x=["AM300"], y=[am300_div_contribution], marker_color="#F59E0B"))
                fig.update_layout(
                    barmode="stack",
                    paper_bgcolor="#0E1117",
                    plot_bgcolor="#0E1117",
                    font=dict(size=10, color="#CCCCCC"),
                    margin=dict(l=0, r=0, t=30, b=0),
                    legend=dict(bgcolor="rgba(0,0,0,0)"),
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption(
                    f"AM300 annual dividend contribution is {am300_div_contribution:.2%}, supporting the flagship total return profile."
                )
        else:
            st.info("Return decomposition unavailable. Run scripts/compare_price_only_vs_total_return.py to create the price-only AM100 series.")

        st.markdown("## Volatility Decomposition")
        if os.path.exists(drivers_path):
            drivers_df = load_csv_file(drivers_path, get_file_version(drivers_path))
            drivers_display = drivers_df.copy()
            drivers_display["Contribution"] = drivers_display["Contribution"].map("{:.2%}".format)
            st.dataframe(drivers_display, use_container_width=True, hide_index=True)
            st.caption("Volatility reduced by diversification and liquidity filtering, with dividend dampening providing an additional cushion.")
        else:
            st.info("Volatility decomposition unavailable. Run scripts/am100_analytics.py to populate this section.")

    with allocator_tab:
        st.header("Allocator Performance")
        st.caption(
            "Client-ready model portfolio system built from AM100 (Core), AM200 (Growth), and AM300 (Broad Market) using daily total return index data."
        )

        st.subheader("Performance Summary")
        allocator_summary_df = pd.DataFrame(
            {
                "Metric": ["CAGR", "Volatility", "Sharpe Ratio", "Max Drawdown"],
                "AM100": [
                    f"{am100_metrics['CAGR'] * 100:.2f}%",
                    f"{am100_metrics['Volatility'] * 100:.2f}%",
                    f"{am100_metrics['Sharpe']:.2f}",
                    f"{am100_metrics['Max Drawdown'] * 100:.2f}%",
                ],
                "AM200": [
                    f"{am200_metrics['CAGR'] * 100:.2f}%",
                    f"{am200_metrics['Volatility'] * 100:.2f}%",
                    f"{am200_metrics['Sharpe']:.2f}",
                    f"{am200_metrics['Max Drawdown'] * 100:.2f}%",
                ],
                "AM300": [
                    f"{am300_metrics['CAGR'] * 100:.2f}%",
                    f"{am300_metrics['Volatility'] * 100:.2f}%",
                    f"{am300_metrics['Sharpe']:.2f}",
                    f"{am300_metrics['Max Drawdown'] * 100:.2f}%",
                ],
            }
        )
        st.dataframe(allocator_summary_df, use_container_width=True, hide_index=True)

        st.subheader("Annual Returns")
        render_annual_returns_section("download_annual_returns_allocator_tab")

        allocator_descriptions = {
            "Conservative": "Lower risk portfolio with higher allocation to liquid, stable constituents.",
            "Balanced": "Moderate risk portfolio combining growth and stability.",
            "Growth": "Higher risk portfolio targeting higher returns through frontier exposure.",
            "Aggressive": "Maximum return portfolio using the optimizer-led high-risk allocation.",
        }

        allocator_cols = st.columns(len(MODEL_WEIGHTS))
        for allocator_col, (model_name, weights) in zip(allocator_cols, MODEL_WEIGHTS.items()):
            with allocator_col:
                st.markdown(f"### {model_name}")
                st.caption(allocator_descriptions.get(model_name, MODEL_METADATA[model_name]["Characteristics"]))
                st.markdown(f"**Objective:** {MODEL_METADATA[model_name]['Objective']}")
                st.write(MODEL_METADATA[model_name]["Characteristics"])
                for index_name, weight in weights.items():
                    st.write(f"{index_name}: {weight:.0%}")

        st.markdown("### Allocation Weights")
        weights_df = pd.DataFrame(MODEL_WEIGHTS).T.reset_index().rename(columns={"index": "Model"})
        weights_display = weights_df.copy()
        for col in ["AM100", "AM200", "AM300"]:
            weights_display[col] = weights_display[col].map("{:.0%}".format)
        st.dataframe(weights_display, use_container_width=True, hide_index=True)

        st.markdown("### Portfolio Metrics")
        model_metrics_display = model_metrics.copy()
        for col in ["CAGR", "Volatility", "Max Drawdown"]:
            model_metrics_display[col] = model_metrics_display[col].map("{:.2%}".format)
        model_metrics_display["Sharpe"] = model_metrics_display["Sharpe"].map("{:.2f}".format)
        model_metrics_display["Latest Level"] = model_metrics_display["Latest Level"].map("{:,.2f}".format)
        model_metrics_display["Estimated Capacity"] = model_metrics_display["Estimated Capacity"].map("{:,.0f}".format)
        st.dataframe(model_metrics_display, use_container_width=True, hide_index=True)

        if annual_returns_daily_df is not None and not annual_returns_daily_df.empty:
            portfolio_source = annual_returns_daily_df.copy().sort_values("Date")
            portfolio_source[["AM100", "AM200", "AM300"]] = (
                portfolio_source[["AM100", "AM200", "AM300"]].ffill().bfill()
            )
            portfolio_returns = []
            for name, weights in PORTFOLIOS.items():
                portfolio_source[name] = build_portfolio_index(portfolio_source, weights)
                ret = compute_annual_returns(portfolio_source, name)
                portfolio_returns.append(ret)

            portfolio_annual = portfolio_returns[0]
            for p in portfolio_returns[1:]:
                portfolio_annual = portfolio_annual.merge(p, on="Year", how="outer")

            portfolio_annual = portfolio_annual.sort_values("Year").round(2)

            st.subheader("Portfolio Returns")
            st.markdown("### Portfolio Annual Returns (%)")
            st.dataframe(portfolio_annual, use_container_width=True, hide_index=True)

            rolling_3y = compute_rolling_returns(portfolio_source, "AM100", 3).rename(columns={"3Y": "AM100"})
            rolling_3y["AM200"] = compute_rolling_returns(portfolio_source, "AM200", 3)["3Y"]
            rolling_3y["AM300"] = compute_rolling_returns(portfolio_source, "AM300", 3)["3Y"]
            for name in PORTFOLIOS.keys():
                rolling_3y[name] = compute_rolling_returns(portfolio_source, name, 3)["3Y"]

            rolling_5y = compute_rolling_returns(portfolio_source, "AM100", 5).rename(columns={"5Y": "AM100"})
            rolling_5y["AM200"] = compute_rolling_returns(portfolio_source, "AM200", 5)["5Y"]
            rolling_5y["AM300"] = compute_rolling_returns(portfolio_source, "AM300", 5)["5Y"]
            for name in PORTFOLIOS.keys():
                rolling_5y[name] = compute_rolling_returns(portfolio_source, name, 5)["5Y"]

            st.subheader("Rolling Returns")
            st.markdown("### Rolling 3-Year Returns (%)")
            rolling_3y_chart = rolling_3y.set_index("Date").dropna(how="all")
            fig, ax = plt.subplots(figsize=(12, 5), facecolor="#0E1117")
            for col in ["AM100", "AM200", "AM300", "Conservative", "Balanced", "Growth"]:
                ax.plot(
                    rolling_3y_chart.index,
                    rolling_3y_chart[col],
                    label=col,
                    linewidth=1.8,
                    color=CHART_COLORS[col],
                )
            ax.set_title("Rolling 3-Year Returns (%)")
            ax.set_ylabel("Return (%)")
            ax.set_xlim(rolling_3y_chart.index.min(), rolling_3y_chart.index.max())
            style_dark_axes(ax)
            ax.legend(frameon=False, ncol=3)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            st.markdown("### Rolling 5-Year Returns (%)")
            rolling_5y_chart = rolling_5y.set_index("Date").dropna(how="all")
            fig, ax = plt.subplots(figsize=(12, 5), facecolor="#0E1117")
            for col in ["AM100", "AM200", "AM300", "Conservative", "Balanced", "Growth"]:
                ax.plot(
                    rolling_5y_chart.index,
                    rolling_5y_chart[col],
                    label=col,
                    linewidth=1.8,
                    color=CHART_COLORS[col],
                )
            ax.set_title("Rolling 5-Year Returns (%)")
            ax.set_ylabel("Return (%)")
            ax.set_xlim(rolling_5y_chart.index.min(), rolling_5y_chart.index.max())
            style_dark_axes(ax)
            ax.legend(frameon=False, ncol=3)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            rolling_3y_stats = rolling_3y.drop(columns=["Date"]).describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).round(2)
            st.markdown("### Rolling Return Statistics - Interpretation")
            st.markdown(
                """
                The table below summarises the distribution of rolling returns over time.

                Count: Number of observations used in the calculation  
                Mean: Average rolling return across all periods  
                Std (Standard Deviation): Measure of variability in returns  
                Min: Lowest observed rolling return (worst-case period)  
                10% / 25% / 50% / 75% / 90%: Percentile distribution of outcomes  
                Example: 10% = worst 10% of periods  
                Max: Highest observed rolling return

                These statistics provide insight into:
                - Return consistency
                - Downside risk
                - Distribution of outcomes over time
                """
            )
            st.dataframe(rolling_3y_stats, use_container_width=True)

            st.markdown("### Performance Characteristics")
            st.markdown(
                """
                Returns are driven by:
                - Structural growth across African markets
                - Dividend contribution through total return methodology
                - Concentration in the most liquid qualifying securities

                Risk is driven by:
                - Market structure and liquidity constraints
                - Country concentration
                - USD-normalised currency exposure
                """
            )

            st.markdown("### Key Observations")
            st.markdown(
                """
                - Strong long-term performance is achieved through compounding
                - Year-to-year returns show real volatility and recovery cycles
                - Rolling returns demonstrate how consistency evolves over time
                - Portfolio blends provide different risk / return profiles for allocators
                """
            )

            st.markdown("### Key Takeaway")
            st.markdown(
                """
                The AM Index framework delivers investable, execution-aware performance, reflecting where capital can actually be deployed and grown over time.
                """
            )

        st.markdown("### Output Table")
        selected_model = st.selectbox(
            "Model",
            list(MODEL_WEIGHTS.keys()),
            key="allocator_model_select",
        )
        cagr, vol, sharpe, dd = portfolio_metrics(model_levels[selected_model])
        with st.container():
            st.markdown(f"### PORTFOLIO: {selected_model.upper()}")
            st.markdown(
                f"""
                **CAGR**: {cagr * 100:.2f}%  
                **Volatility**: {vol * 100:.2f}%  
                **Sharpe**: {sharpe:.2f}  
                **Max Drawdown**: {dd * 100:.2f}%
                """
            )

        st.markdown("### MPS Portfolio Builder")
        model = st.selectbox(
            "Select Portfolio",
            list(MODEL_WEIGHTS.keys()),
            key="mps_model_select",
        )
        weights = MODEL_WEIGHTS[model]
        port_returns = build_portfolio(allocator_returns, weights)
        port_index = build_index(port_returns)
        cagr, vol, sharpe, dd = portfolio_metrics(port_index)

        with st.container():
            st.markdown(f"### PORTFOLIO: {model.upper()}")
            st.markdown(
                f"""
                **CAGR**: {cagr * 100:.2f}%  
                **Volatility**: {vol * 100:.2f}%  
                **Sharpe**: {sharpe:.2f}  
                **Max Drawdown**: {dd * 100:.2f}%
                """
            )
            st.markdown("---")
            st.markdown(
                f"""
                **ALLOCATION**

                AM100 ⟶ {weights.get('AM100', 0) * 100:.0f}%  
                AM200 ⟶ {weights.get('AM200', 0) * 100:.0f}%  
                AM300 ⟶ {weights.get('AM300', 0) * 100:.0f}%
                """
            )

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=port_index.index,
                y=port_index,
                name=model,
                line=dict(color="#F59E0B", width=2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=am100_s.index,
                y=am100_s,
                name="AM100",
                line=dict(color=AM100_COLOR, width=2),
            )
        )
        fig.update_layout(
            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117",
            font=dict(size=10, color="#CCCCCC"),
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Portfolio Performance")
        fig = go.Figure()
        model_colors = {
            "Conservative": AM100_COLOR,
            "Balanced": "#38BDF8",
            "Growth": AM200_COLOR,
            "Aggressive": AM300_COLOR,
        }
        for model_name in model_levels.columns:
            fig.add_trace(
                go.Scatter(
                    x=model_levels.index,
                    y=model_levels[model_name],
                    name=model_name,
                    line=dict(color=model_colors[model_name], width=2),
                )
            )
        fig.update_layout(
            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117",
            font=dict(size=10, color="#CCCCCC"),
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Efficient Frontier")
        st.caption(
            "The efficient frontier shows the highest expected return for each level of risk. Points represent portfolio allocations."
        )
        frontier = simulate_random_frontier(prepare_returns_frame())
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=frontier["Vol"],
                y=frontier["Return"],
                mode="markers",
                marker=dict(size=2, color="#7DD3FC", opacity=0.3),
                name="Efficient Frontier",
                hovertemplate="Return: %{y:.2%}<br>Risk: %{x:.2%}<extra></extra>",
            )
        )

        def plot_point(name, vol_value, ret_value, color):
            fig.add_trace(
                go.Scatter(
                    x=[vol_value],
                    y=[ret_value],
                    mode="markers+text",
                    text=[name],
                    textposition="top center",
                    marker=dict(size=10, color=color),
                    name=name,
                    hovertemplate="Return: %{y:.2%}<br>Risk: %{x:.2%}<extra></extra>",
                )
            )

        for model_name, color in model_colors.items():
            row = model_metrics[model_metrics["Model"] == model_name].iloc[0]
            plot_point(model_name, row["Volatility"], row["CAGR"], color)
        fig.update_layout(
            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117",
            font=dict(size=10, color="#CCCCCC"),
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            xaxis_title="Risk (Annualised Volatility)",
            yaxis_title="Expected Return (Annualised)",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Each point represents a portfolio allocation. The curve shows the best achievable return for a given level of risk."
        )

        st.markdown("### Building Block Mix")
        fig = go.Figure()
        for index_name, color in [("AM100", AM100_COLOR), ("AM200", AM200_COLOR), ("AM300", AM300_COLOR)]:
            fig.add_trace(
                go.Bar(
                    x=list(MODEL_WEIGHTS.keys()),
                    y=[MODEL_WEIGHTS[model][index_name] for model in MODEL_WEIGHTS],
                    name=index_name,
                    marker_color=color,
                )
            )
        fig.update_layout(
            barmode="stack",
            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117",
            font=dict(size=10, color="#CCCCCC"),
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Estimated capacity is shown as investable capacity in USD, calculated as 20% of average daily traded value to reflect prudent institutional execution."
        )

    with overview_tab:
        st.markdown("## Performance")

        col1, col2 = st.columns(2)

        with col1:
            st.caption("Index Performance")
            fig, ax = plt.subplots(figsize=(6, 2.4))
            style_chart(fig, ax)
            ax.plot(am100_plot, label="AM100", color=AM100_COLOR, linewidth=2)
            ax.plot(am200_plot, label="AM200", color=AM200_COLOR, linewidth=2)
            ax.plot(am300, label="AM300", color=AM300_COLOR, linewidth=2)
            ax.legend(frameon=False, fontsize=10)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)

        with col2:
            st.caption("Log Performance")
            fig, ax = plt.subplots(figsize=(6, 2.4))
            style_chart(fig, ax)
            ax.plot(am100_plot, color=AM100_COLOR, linewidth=2)
            ax.plot(am200_plot, color=AM200_COLOR, linewidth=2)
            ax.plot(am300, color=AM300_COLOR, linewidth=2)
            ax.set_yscale("log")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            st.caption("Series begin at first eligible inclusion date per index.")

        st.caption("Index composition shift due to liquidity-driven rebalance")


    with overview_tab:
        st.markdown("## 🌍 Allocation")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### AM100 Country Allocation")
            if not am100_country_df.empty:
                fig, ax = plt.subplots(figsize=(6, 2.4), facecolor="#0E1117")
                style_chart(fig, ax)
                ax.bar(am100_country_df["Country"], am100_country_df["Weight"], color=AM100_COLOR)
                ax.tick_params(axis="x", rotation=90)
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)

        with col2:
            st.markdown("### AM200 Country Allocation")
            if SHOW_AM200 and not am200_country_df.empty:
                fig, ax = plt.subplots(figsize=(6, 2.4), facecolor="#0E1117")
                style_chart(fig, ax)
                ax.bar(am200_country_df["Country"], am200_country_df["Weight"], color=AM200_COLOR)
                ax.tick_params(axis="x", rotation=90)
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
            else:
                st.info(
                    "AM200 country allocation is being standardised under the updated liquidity framework and will be included in the next release."
                )

        st.markdown(
            """
            ### 🌍 Africa Investability Map

            This map shows which African markets meet institutional liquidity and trading standards under the AM100 methodology.

            Markets are included based on:
            - Minimum liquidity thresholds
            - Trading consistency
            - Data integrity

            Exclusions reflect structural market constraints, not qualitative judgement.
            """
        )

        if investability_map is not None and not investability_map.empty:
            iso_map = {
                "SOUTH AFRICA": "ZAF",
                "EGYPT": "EGY",
                "MOROCCO": "MAR",
                "NIGERIA": "NGA",
                "KENYA": "KEN",
                "NAMIBIA": "NAM",
                "ZIMBABWE": "ZWE",
                "SENEGAL": "SEN",
                "MAURITIUS": "MUS",
                "UGANDA": "UGA",
                "GHANA": "GHA",
                "TANZANIA": "TZA",
                "MALAWI": "MWI",
                "ZAMBIA": "ZMB",
                "NIGER": "NER",
                "TOGO": "TGO",
                "RWANDA": "RWA",
                "TUNISIA": "TUN",
            }

            map_df = investability_map.copy()
            map_df["Country"] = map_df["Country"].astype(str).str.upper()
            map_df["ISO"] = map_df["Country"].map(iso_map)
            map_df = map_df.dropna(subset=["ISO"]).copy()
            map_df["StatusColor"] = map_df["Inclusion Status"].map(
                {"Included": 1, "Excluded": 0}
            )
            map_df["Tooltip"] = (
                "Country: "
                + map_df["Country"].str.title()
                + "<br>Avg ADV: $"
                + map_df["AvgADV_USD_30D"].fillna(0).round(0).map("{:,.0f}".format)
                + "<br>Median ADV: $"
                + map_df["MedianADV_USD_30D"].fillna(0).round(0).map("{:,.0f}".format)
                + "<br>Avg Trading Days: "
                + map_df["AvgTradingDays90d"].fillna(0).round(1).astype(str)
                + "<br>Selected: "
                + map_df["SelectedCount"].fillna(0).astype(int).astype(str)
                + "<br>Status: "
                + map_df["Inclusion Status"].astype(str)
            )

            fig = px.choropleth(
                map_df,
                locations="ISO",
                color="StatusColor",
                hover_name="Country",
                hover_data={"StatusColor": False, "ISO": False},
                color_continuous_scale=["#d62728", "#2ca02c"],
                range_color=(0, 1),
            )

            fig.update_traces(hovertemplate=map_df["Tooltip"])
            fig.update_layout(
                coloraxis_showscale=False,
                geo=dict(
                    scope="africa",
                    projection_type="natural earth",
                    showland=True,
                    landcolor="#0E1117",
                    bgcolor="#0E1117",
                ),
                paper_bgcolor="#0E1117",
                plot_bgcolor="#0E1117",
                font=dict(size=10),
                margin=dict(l=0, r=0, t=0, b=0),
            )

            st.plotly_chart(fig, use_container_width=True)
            st.markdown(
                "*Note: Markets excluded from AM100 may still present attractive investment opportunities but currently fall below institutional liquidity and trading thresholds.*"
            )
        else:
            st.info("Investability map will appear once output/investability_map.csv is available.")

        st.markdown("---")

    with overview_tab:
        st.markdown("## 🏦 Constituents")

        if IS_INVESTOR:
            st.caption("Top Holdings")

            investor_col1, investor_col2, investor_col3 = st.columns(3)

            with investor_col1:
                st.markdown("**AM100 Top 10**")
                investor_am100 = prepare_constituent_table(am100_latest, limit=10)[
                    ["Company", "Country", "Weight"]
                ]
                st.dataframe(display_with_row_numbers(investor_am100), use_container_width=True)

            with investor_col2:
                st.markdown("**AM200 Top 10**")
                if am200_latest is None or am200_latest.empty:
                    st.info("AM200 not yet populated")
                else:
                    top10_am200 = (
                        am200_latest.sort_values("Weight", ascending=False)
                        .head(10)[["Company", "Country", "Weight"]]
                        .copy()
                    )
                    top10_am200["Weight"] = top10_am200["Weight"].map("{:.2%}".format)
                    st.dataframe(
                        display_with_row_numbers(top10_am200),
                        use_container_width=True,
                    )

            with investor_col3:
                st.markdown("**AM300 Top 10**")
                investor_am300 = prepare_constituent_table(am300_latest, limit=10)[
                    ["Company", "Country", "Weight"]
                ]
                st.dataframe(display_with_row_numbers(investor_am300), use_container_width=True)

            st.caption("Geographic Exposure")
            investor_country_exposure = pd.concat(
                [
                    am100_country.rename("AM100"),
                    am200_country.rename("AM200"),
                    am300_country.rename("AM300"),
                ],
                axis=1,
            ).fillna(0)
            investor_country_exposure = (
                investor_country_exposure.sort_values("AM300", ascending=False)
                .reset_index()
                .rename(columns={"index": "Country"})
            )
            st.dataframe(investor_country_exposure, use_container_width=True, hide_index=True)
        else:
            col1, col2 = st.columns(2)

            with col1:
                st.caption("AM100 Top 10")
                am100_top = prepare_constituent_table(am100_latest, limit=10)
                st.dataframe(display_with_row_numbers(am100_top), use_container_width=True)

            with col2:
                st.caption("AM200 Top 10")
                if am200_latest is None or am200_latest.empty:
                    st.info("AM200 not yet populated")
                else:
                    top10_am200 = (
                        am200_latest.sort_values("Weight", ascending=False)
                        .head(10)[["Company", "Country", "Weight"]]
                        .copy()
                    )
                    top10_am200["Weight"] = top10_am200["Weight"].map("{:.2%}".format)
                    st.dataframe(
                        display_with_row_numbers(top10_am200),
                        use_container_width=True,
                    )

            with st.expander("🔍 View Full AM100 Constituents (Top 100)"):
                search = st.text_input("Search AM100", key="am100_search")
                am100_full = prepare_constituent_table(
                    am100_latest, include_rank=True, search=search
                )
                st.dataframe(display_with_row_numbers(am100_full), use_container_width=True)

            with st.expander("🔍 View Full AM200 Constituents (101–200)"):
                search = st.text_input("Search AM200", key="am200_search")
                am200_full = prepare_constituent_table(
                    am200_latest, include_rank=True, search=search
                )
                st.dataframe(display_with_row_numbers(am200_full), use_container_width=True)

            st.caption("Holdings Explorer")

            view_option = st.selectbox(
                "Select View",
                [
                    "Top 10 (AM100 & AM200)",
                    "Full AM100 (Top 100)",
                    "Full AM200 (101–200)",
                ],
            )

        st.markdown("---")

        st.markdown("## Methodology")

        col1, col2 = st.columns([2, 1])

    with overview_tab:
        with col1:
            st.markdown("### Methodology Flow")

            st.markdown(
                """
    **Data Pipeline -> Index Construction**

    **1. Raw Market Data**
    - Prices, volumes, corporate actions

    **2. Cleaning & Validation**
    - Outlier detection  
    - Missing data handling  

    **3. Liquidity Scoring**
    - Traded Value x Participation^2  

    **4. Ranking Engine**
    - Cross-market comparability  

    **5. Eligibility Filters**
    - Liquidity thresholds  
    - Data consistency  

    **6. Index Construction**
    - AM100 (Top 100)  
    - AM200 (Next 100)  
    - AM300 (All Share Total Return)  

    **7. Rebalancing**
    - Monthly  
    - Buffer-controlled turnover  
    """
            )

            st.markdown("### Methodology Overview")

            st.markdown(
                """
    - Liquidity-driven selection  
    - Participation-adjusted scoring  
    - Monthly rebalancing  
    - Country caps applied iteratively  
    - No interpolation of missing data  
    - Total return methodology with dividend integration  
    """
            )

        with col2:
            st.markdown("### Qualification Rules")

            st.markdown("**Eligibility**")
            st.markdown(
                """
    - Minimum trading history  
    - Valid price series  
    - Volume or value data available  
    """
            )

            st.markdown("**Liquidity Model**")
            st.markdown(
                """
    - 30 valid trading days  
    - Traded Value x Participation^2  
    - Country/regime scaling  
    """
            )

        pdf_data = generate_pdf(cagr100, cagr200)
        st.download_button(
            "📄 Download PDF Report",
            data=pdf_data,
            file_name="AM_Report.pdf",
        )

        st.markdown("---")
        st.caption("Veri AM Indices • Liquidity-Driven African Equity Benchmarks • 2026")


def render_eac_series():
    render_global_styles()
    st.title("EAC Series")
    st.subheader("EAC25 Core & EAC Extended")
    render_eac_dashboard()


def render_comparison():
    render_global_styles()
    st.title("Comparison")
    render_am_eac_comparison()


def main():
    configure_app()

    series_choice = st.radio(
        "Select Series",
        ["AM Series", "EAC Series", "Comparison"],
        horizontal=True,
        key="global_series_selector",
    )

    st.markdown("---")

    if series_choice == "AM Series":
        render_am_series()
        st.stop()
    elif series_choice == "EAC Series":
        render_eac_series()
        st.stop()
    elif series_choice == "Comparison":
        render_comparison()
        st.stop()


if __name__ == "__main__":
    main()
