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

# Temporary deployment cache-buster for Streamlit Cloud refreshes.
st.cache_data.clear()


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


def render_global_header():
    col1, col2 = st.columns([0.8, 9])

    with col1:
        st.image(logo_path, width=60)

    with col2:
        st.markdown(
            """
            <div style="line-height: 1.2;">
                <span style="font-size:16px; font-weight:600;">African Market Indices</span><br>
                <span style="font-size:12px; color:#9AA4AF;">
                AM100 / AM200 / AM300 Total Return Indices
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.caption(f"Build: {BUILD_VERSION}")

    header_spacer, header_logout = st.columns([8, 1])
    with header_logout:
        if st.button("Logout", key="global_logout"):
            st.session_state.auth_mode = None
            st.rerun()


def safe_display(value, fallback="N/A"):
    return value if value is not None else fallback


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


def compute_performance(series, window_type, **kwargs):
    index = series.dropna().index.sort_values()
    if len(index) < 2:
        return {"status": "NO_DATA", "cagr": None, "start": None, "end": None}

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
    results = {}

    for name, period in periods.items():
        if name == "latest_valid" or not isinstance(period, dict):
            continue
        if period.get("type") == "rolling":
            results[name] = compute_performance(
                index_series,
                "rolling",
                years=period["years"],
                tolerance_days=period.get("tolerance_days", 30),
            )
        else:
            results[name] = compute_performance(
                index_series,
                "fixed",
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


def get_common_analysis_window(index_map, requested_start=pd.Timestamp("2016-01-01")):
    valid_series = {
        name: series.dropna().sort_index()
        for name, series in index_map.items()
        if series is not None and len(series.dropna()) >= 2
    }
    if not valid_series:
        return {"start": None, "end": None, "label": "No common period", "series": {}}

    common_dates = None
    for series in valid_series.values():
        common_dates = series.index if common_dates is None else common_dates.intersection(series.index)

    if common_dates is None or len(common_dates) < 2:
        return {"start": None, "end": None, "label": "No common period", "series": {}}

    common_dates = common_dates.sort_values()
    common_start = common_dates.min()
    common_end = common_dates.max()
    start = max(requested_start, common_start)

    aligned = {}
    for name, series in valid_series.items():
        aligned_series = series.loc[start:common_end].dropna()
        if len(aligned_series) >= 2:
            aligned[name] = aligned_series

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
SHOW_AM200 = False
SHOW_AM300 = False

st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

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
allocator_mode = st.toggle("Allocator Presentation Mode", value=False)

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

render_global_header()


def render_allocator_view():
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

    st.markdown("### Performance Snapshot")
    col1, col2, col3 = st.columns(3)

    col1.metric("AM100 CAGR", pct_metric(comparison_metrics.get("AM100"), "CAGR"))
    col1.metric("Sharpe", num_metric(comparison_metrics.get("AM100"), "Sharpe"))

    col2.metric("AM200 CAGR", pct_metric(comparison_metrics.get("AM200"), "CAGR"))
    col2.metric("Sharpe", num_metric(comparison_metrics.get("AM200"), "Sharpe"))

    col3.metric("AM300 CAGR", pct_metric(comparison_metrics.get("AM300"), "CAGR"))
    col3.metric("Sharpe", num_metric(comparison_metrics.get("AM300"), "Sharpe"))

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
                pct_metric(comparison_metrics.get("AM100"), "CAGR"),
                pct_metric(comparison_metrics.get("AM100"), "Volatility"),
                num_metric(comparison_metrics.get("AM100"), "Sharpe"),
                pct_metric(comparison_metrics.get("AM100"), "Max Drawdown"),
                am100_snapshot_insights.get("constituents"),
                turnover_metric(am100_snapshot_insights),
            ],
            "AM200": [
                pct_metric(comparison_metrics.get("AM200"), "CAGR"),
                pct_metric(comparison_metrics.get("AM200"), "Volatility"),
                num_metric(comparison_metrics.get("AM200"), "Sharpe"),
                pct_metric(comparison_metrics.get("AM200"), "Max Drawdown"),
                am200_snapshot_insights.get("constituents"),
                turnover_metric(am200_snapshot_insights),
            ],
            "AM300": [
                pct_metric(comparison_metrics.get("AM300"), "CAGR"),
                pct_metric(comparison_metrics.get("AM300"), "Volatility"),
                num_metric(comparison_metrics.get("AM300"), "Sharpe"),
                pct_metric(comparison_metrics.get("AM300"), "Max Drawdown"),
                am300_snapshot_insights.get("constituents"),
                turnover_metric(am300_snapshot_insights),
            ],
        }
    )
    st.markdown(f"**Performance Period:** {common_period_label}")
    st.caption(
        "All comparison metrics are calculated over the shared AM100 / AM200 / AM300 window. Turnover remains the average across rebalance events."
    )
    st.dataframe(comparison_df, use_container_width=True, hide_index=True, height=240)
    st.caption(
        "Metrics are calculated using USD total return methodology. Volatility and Sharpe Ratio are annualised. Max Drawdown is measured over the full shared period."
    )

    st.markdown(
        """
        ### Interpretation

        AM100 defines where institutional capital can be deployed today.

        AM200 and AM300 expand the opportunity set as liquidity conditions allow.

        Performance dispersion reflects real market depth, not model variation.
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
    st.title("Veri African Indices")
    st.markdown(
        """
        **AM100 is the institutional core of the Veri African Indices suite.**

        AM100 applies hard USD liquidity, trading-consistency, and total-return standards to define the investable African equity universe. AM200 and AM300 remain part of the wider framework, but AM100 is the benchmark tier currently positioned for institutional use.
        """
    )

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
    def pct_metric(metrics, key):
        value = metrics.get(key) if metrics else None
        return f"{value * 100:.2f}%" if value is not None and pd.notna(value) else "N/A"

    def num_metric(metrics, key):
        value = metrics.get(key) if metrics else None
        return f"{value:.2f}" if value is not None and pd.notna(value) else "N/A"

    def turnover_metric(snapshot):
        value = snapshot.get("avg_turnover")
        return f"{value * 100:.1f}%" if value is not None else "N/A"

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
                pct_metric(comparison_metrics.get("AM100"), "CAGR"),
                pct_metric(comparison_metrics.get("AM100"), "Volatility"),
                num_metric(comparison_metrics.get("AM100"), "Sharpe"),
                pct_metric(comparison_metrics.get("AM100"), "Max Drawdown"),
                am100_snapshot_insights.get("constituents"),
                turnover_metric(am100_snapshot_insights),
            ],
            "AM200": [
                pct_metric(comparison_metrics.get("AM200"), "CAGR"),
                pct_metric(comparison_metrics.get("AM200"), "Volatility"),
                num_metric(comparison_metrics.get("AM200"), "Sharpe"),
                pct_metric(comparison_metrics.get("AM200"), "Max Drawdown"),
                am200_snapshot_insights.get("constituents"),
                turnover_metric(am200_snapshot_insights),
            ],
            "AM300": [
                pct_metric(comparison_metrics.get("AM300"), "CAGR"),
                pct_metric(comparison_metrics.get("AM300"), "Volatility"),
                num_metric(comparison_metrics.get("AM300"), "Sharpe"),
                pct_metric(comparison_metrics.get("AM300"), "Max Drawdown"),
                am300_snapshot_insights.get("constituents"),
                turnover_metric(am300_snapshot_insights),
            ],
        }
    )

    st.markdown("### AM INDEX COMPARISON")
    with st.container():
        st.markdown(
            """
            **The AM index family provides a tiered view of African equity markets,
            from institutional core exposure to broader frontier opportunity sets.**
            """
        )
        st.markdown(
            f"""
            **Performance Period:**  
            {common_period_label}

            All comparison metrics (CAGR, Volatility, Sharpe Ratio, Drawdown) are calculated over this shared period.
            """
        )
        st.dataframe(comparison_df, use_container_width=True, hide_index=True, height=240)
        st.caption(
            "Metrics are calculated using USD total return methodology. Volatility and Sharpe Ratio are annualised. Max Drawdown is measured over the full shared period. Turnover is shown as the average across rebalance events."
        )
        st.divider()

        col1, col2, col3 = st.columns(3)
        col1.markdown(
            f"""
            **AM100**  
            CAGR: {pct_metric(comparison_metrics.get("AM100"), "CAGR")}  
            Sharpe: {num_metric(comparison_metrics.get("AM100"), "Sharpe")}  
            Cons: {am100_snapshot_insights.get("constituents")}  
            Turnover: {turnover_metric(am100_snapshot_insights)}
            """
        )
        col2.markdown(
            f"""
            **AM200**  
            CAGR: {pct_metric(comparison_metrics.get("AM200"), "CAGR")}  
            Sharpe: {num_metric(comparison_metrics.get("AM200"), "Sharpe")}  
            Cons: {am200_snapshot_insights.get("constituents")}  
            Turnover: {turnover_metric(am200_snapshot_insights)}
            """
        )
        col3.markdown(
            f"""
            **AM300**  
            CAGR: {pct_metric(comparison_metrics.get("AM300"), "CAGR")}  
            Sharpe: {num_metric(comparison_metrics.get("AM300"), "Sharpe")}  
            Cons: {am300_snapshot_insights.get("constituents")}  
            Turnover: {turnover_metric(am300_snapshot_insights)}
            """
        )
        st.divider()

        if index_overlap_metrics:
            st.markdown(
                f"""
                **STRUCTURE**

                AM100 ⊂ AM200: {index_overlap_metrics.get('AM100_in_AM200', 0) * 100:.0f}%  
                AM100 ⊂ AM300: {index_overlap_metrics.get('AM100_in_AM300', 0) * 100:.0f}%  
                AM200 ⊂ AM300: {index_overlap_metrics.get('AM200_in_AM300', 0) * 100:.0f}%
                """
            )
            st.caption(
                """
                Each index is a strict superset of the previous tier, ensuring consistency,
                transparency, and comparability across the AM index family.
                """
            )
        st.divider()

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

if DEBUG:
    st.write(os.listdir())

with st.expander("📘 Methodology & Overview", expanded=False):
    st.markdown(methodology_text)

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
plt.style.use("default")
plt.rcParams.update({
    "figure.figsize": (10, 4),
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "font.family": "sans-serif",
})


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
        st.error(f"No data file found for {name}: {path}")
        return None

    df = load_index(name)
    if "Index Level" not in df.columns:
        st.error(f"{name} file is missing required column: Index Level")
        return None

    return df.set_index("Date")["Index Level"]


def load_data():
    am100_path = "output/AM100_total_return.csv"
    am200_path = "output/AM200_total_return.csv"
    am300_path = "output/AM300_total_return.csv"
    am100 = load_total_return("AM100", get_file_version(am100_path))
    am200 = load_total_return("AM200", get_file_version(am200_path))
    am300 = load_total_return("AM300", get_file_version(am300_path))

    if am100 is None or am200 is None or am300 is None:
        st.stop()

    return am100, am200, am300


am100, am200, am300 = load_data()

common_analysis = get_common_analysis_window({"AM100": am100, "AM200": am200, "AM300": am300})
common_start = common_analysis["start"]
common_end = common_analysis["end"]
common_period_label = common_analysis["label"]
comparison_series = common_analysis["series"]
comparison_metrics = {name: compute_window_metrics(series) for name, series in comparison_series.items()}

st.write("Data Last Updated:", am100.index.max())


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
volume_audit_df = audit_volume_quality(price_panel)
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
NO_VOLUME_EXPORT_FILE = "output/missing_volume_companies.csv"
LOW_COVERAGE_EXPORT_FILE = "output/low_coverage_companies.csv"
no_volume_df.to_csv(NO_VOLUME_EXPORT_FILE, index=False)
low_volume_coverage_df.to_csv(LOW_COVERAGE_EXPORT_FILE, index=False)
latest_date = am100_hist["Date"].max()

am100_latest = am100_hist[am100_hist["Date"] == latest_date]
am200_latest = am200_hist[am200_hist["Date"] == latest_date]
am300_latest = am300_hist[am300_hist["Date"] == latest_date]

am100_constituents = am100_latest["Company"].tolist()
am200_constituents = am200_latest["Company"].tolist()
am300_constituents = am300_latest["Company"].tolist()

am100_periods = get_periods(price_panel, am100_constituents, am100)
am200_periods = get_periods(price_panel, am200_constituents, am200)
am300_periods = get_periods(price_panel, am300_constituents, am300)

am100_cagrs = compute_cagr_for_periods(am100, am100_periods)
am200_cagrs = compute_cagr_for_periods(am200, am200_periods)
am300_cagrs = compute_cagr_for_periods(am300, am300_periods)

am100_cagr_10y = am100_cagrs["rolling_10y"]["cagr"]
am100_cagr_rolling_5y = am100_cagrs["rolling_5y"]["cagr"]
am100_cagr_rolling_3y = am100_cagrs["rolling_3y"]["cagr"]
am100_cagr_rolling_1y = am100_cagrs["rolling_1y"]["cagr"]
am100_cagr_2016 = am100_cagrs["since_2016"]["cagr"]
am100_cagr_fixed_10y = am100_cagrs["fixed_10y"]["cagr"]
am100_cagr_fixed_5y = am100_cagrs["fixed_5y"]["cagr"]

am200_cagr_10y = am200_cagrs["rolling_10y"]["cagr"]
am200_cagr_rolling_5y = am200_cagrs["rolling_5y"]["cagr"]
am200_cagr_rolling_3y = am200_cagrs["rolling_3y"]["cagr"]
am200_cagr_rolling_1y = am200_cagrs["rolling_1y"]["cagr"]
am200_cagr_2016 = am200_cagrs["since_2016"]["cagr"]
am200_cagr_fixed_10y = am200_cagrs["fixed_10y"]["cagr"]
am200_cagr_fixed_5y = am200_cagrs["fixed_5y"]["cagr"]

am300_cagr_10y = am300_cagrs["rolling_10y"]["cagr"]
am300_cagr_rolling_5y = am300_cagrs["rolling_5y"]["cagr"]
am300_cagr_rolling_3y = am300_cagrs["rolling_3y"]["cagr"]
am300_cagr_rolling_1y = am300_cagrs["rolling_1y"]["cagr"]
am300_cagr_2016 = am300_cagrs["since_2016"]["cagr"]
am300_cagr_fixed_10y = am300_cagrs["fixed_10y"]["cagr"]
am300_cagr_fixed_5y = am300_cagrs["fixed_5y"]["cagr"]

analytics_coverage_notes = []

for index_name, index_series, periods, period_results in [
    ("AM100", am100, am100_periods, am100_cagrs),
    ("AM200", am200, am200_periods, am200_cagrs),
    ("AM300", am300, am300_periods, am300_cagrs),
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
st.sidebar.title("Controls")

start_date = st.sidebar.date_input("Start Date", am100.index.min())
end_date = st.sidebar.date_input("End Date", am100.index.max())

# Filter data
am100 = am100[
    (am100.index >= pd.to_datetime(start_date))
    & (am100.index <= pd.to_datetime(end_date))
]
am200 = am200[
    (am200.index >= pd.to_datetime(start_date))
    & (am200.index <= pd.to_datetime(end_date))
]
am300 = am300[
    (am300.index >= pd.to_datetime(start_date))
    & (am300.index <= pd.to_datetime(end_date))
]

am100_df = am100.rename("Index Level").reset_index()
am200_df = am200.rename("Index Level").reset_index()
am300_df = am300.rename("Index Level").reset_index()

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

st.caption(f"Shared comparison window across AM100 / AM200 / AM300: {common_period_label}")

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
        "rolling_10y": am100_cagrs["rolling_10y"],
        "rolling_5y": am100_cagrs["rolling_5y"],
        "rolling_3y": am100_cagrs["rolling_3y"],
        "rolling_1y": am100_cagrs["rolling_1y"],
    },
    "AM200": {
        "rolling_10y": am200_cagrs["rolling_10y"],
        "rolling_5y": am200_cagrs["rolling_5y"],
        "rolling_3y": am200_cagrs["rolling_3y"],
        "rolling_1y": am200_cagrs["rolling_1y"],
    },
    "AM300": {
        "rolling_10y": am300_cagrs["rolling_10y"],
        "rolling_5y": am300_cagrs["rolling_5y"],
        "rolling_3y": am300_cagrs["rolling_3y"],
        "rolling_1y": am300_cagrs["rolling_1y"],
    },
}

comparison_period_result = {
    name: compute_performance(series, "fixed", start_date=common_start, end_date=common_end)
    if common_start is not None and common_end is not None and name in comparison_series
    else {"status": "NO_DATA", "cagr": None, "start": None, "end": None}
    for name, series in {"AM100": am100, "AM200": am200, "AM300": am300}.items()
}

fixed_results = {
    "AM100": {
        "common_period": comparison_period_result["AM100"],
        "fixed_10y": am100_cagrs["fixed_10y"],
        "fixed_5y": am100_cagrs["fixed_5y"],
    },
    "AM200": {
        "common_period": comparison_period_result["AM200"],
        "fixed_10y": am200_cagrs["fixed_10y"],
        "fixed_5y": am200_cagrs["fixed_5y"],
    },
    "AM300": {
        "common_period": comparison_period_result["AM300"],
        "fixed_10y": am300_cagrs["fixed_10y"],
        "fixed_5y": am300_cagrs["fixed_5y"],
    },
}


def safe_metric(result):
    if isinstance(result, dict):
        value = result.get("cagr")
    else:
        value = result
    return f"{value:.2%}" if value is not None and pd.notnull(value) else "N/A"


def safe_period_range(result):
    if not isinstance(result, dict) or result.get("status") != "OK":
        return "Insufficient data"
    start = result.get("start")
    end = result.get("end")
    if start is None or end is None:
        return "Insufficient data"
    return f"{start.strftime('%d %b %Y')} -> {end.strftime('%d %b %Y')}"


st.subheader("Headline Performance")
for index_name, results in fixed_results.items():
    st.markdown(f"**{index_name}**")
    col1, col2, col3 = st.columns(3)
    col1.metric("Common Window", safe_metric(results["common_period"]))
    col2.metric("10Y (Fixed)", safe_metric(results["fixed_10y"]))
    col3.metric("5Y (Fixed)", safe_metric(results["fixed_5y"]))
    st.markdown(
        f"<div class='small-note'>"
        f"Common Window: {safe_period_range(results['common_period'])} | "
        f"10Y (Fixed): {safe_period_range(results['fixed_10y'])} | "
        f"5Y (Fixed): {safe_period_range(results['fixed_5y'])}"
        f"</div>",
        unsafe_allow_html=True,
    )

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
top_country_am200 = get_top_country(am200_latest) if SHOW_AM200 else None
top_country_am300 = get_top_country(am300_latest) if SHOW_AM300 else None
am100_country = get_country_weights(am100_latest)
am200_country = get_country_weights(am200_latest) if SHOW_AM200 else pd.Series(dtype=float)
am300_country = get_country_weights(am300_latest) if SHOW_AM300 else pd.Series(dtype=float)
latest_am100 = am100.iloc[-1]
latest_am200 = am200.iloc[-1] if SHOW_AM200 and len(am200) else None
latest_am300 = am300.iloc[-1] if SHOW_AM300 and len(am300) else None

ticker_parts = [
    f"AM100: {latest_am100:.2f} ({ret100:.2%})",
    f"Top Country AM100: {safe_display(top_country_am100)}",
]
if SHOW_AM200 and latest_am200 is not None:
    ticker_parts.append(f"AM200: {latest_am200:.2f} ({ret200:.2%})")
    ticker_parts.append(f"Top Country AM200: {safe_display(top_country_am200)}")
if SHOW_AM300 and latest_am300 is not None:
    ticker_parts.append(f"AM300: {latest_am300:.2f} ({ret300:.2%})")
    ticker_parts.append(f"Top Country AM300: {safe_display(top_country_am300)}")
ticker_text = " •\n".join(ticker_parts)

if top_country_am100:
    obs.append(f"AM100 is most exposed to {top_country_am100}.")
if SHOW_AM200 and top_country_am200:
    obs.append(f"AM200 shows strongest exposure to {top_country_am200}.")
if SHOW_AM300 and top_country_am300:
    obs.append(f"AM300 shows flagship concentration in {top_country_am300}.")

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


@st.cache_data
def load_capacity_usd_file(path, _version=None):
    return pd.read_csv(path, parse_dates=["Date"]).sort_values("Date")


@st.cache_data
def load_investability_map(_version=None):
    path = "output/investability_map.csv"
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


am100_adv_usd, am100_capacity = external_capacity_metrics(am100_latest)
am200_adv_usd, am200_capacity = external_capacity_metrics(am200_latest)
am300_adv_usd, am300_capacity = external_capacity_metrics(am300_latest)
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
            st.write(sorted(low_volume_coverage_df["Security"].tolist()))
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
    st.markdown("## Model Portfolio Allocator")
    st.caption(
        "Client-ready model portfolio system built from AM100 (Core), AM200 (Growth), and AM300 (Broad Market) using daily total return index data."
    )

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

    st.caption("Index composition shift due to liquidity-driven rebalance")


with overview_tab:
    st.markdown("## 🌍 Allocation")

    col1, col2 = st.columns(2)

    with col1:
        st.caption("AM100")
        fig, ax = plt.subplots(figsize=(6, 2.4))
        style_chart(fig, ax)
        ax.bar(am100_country.index, am100_country.values, color=AM100_COLOR)
        ax.tick_params(axis="x", rotation=90)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

    with col2:
        st.caption("AM200")
        if SHOW_AM200 and not am200_country.empty:
            fig, ax = plt.subplots(figsize=(6, 2.4))
            style_chart(fig, ax)
            ax.bar(am200_country.index, am200_country.values, color=AM200_COLOR)
            ax.tick_params(axis="x", rotation=90)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
        else:
            st.info(
                """
                AM200 — Expanding coverage through validated data integration (Phase 2)

                This tier is being expanded through validated data integration while preserving liquidity integrity, positive trading coverage requirements, and execution realism.
                """
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
            investor_am200 = prepare_constituent_table(am200_latest, limit=10)[
                ["Company", "Country", "Weight"]
            ]
            st.dataframe(display_with_row_numbers(investor_am200), use_container_width=True)

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
            am200_top = prepare_constituent_table(am200_latest, limit=10)
            st.dataframe(display_with_row_numbers(am200_top), use_container_width=True)

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
