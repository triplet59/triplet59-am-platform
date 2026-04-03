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

    years = (df_period["Date"].iloc[-1] - df_period["Date"].iloc[0]).days / 365

    return (end_val / start_val) ** (1 / years) - 1


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


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(BASE_DIR, "..", "assets", "veri_logo.png")
AM100_COLOR = "#4DA3FF"
AM200_COLOR = "#FF9F1C"
AM300_COLOR = "#22C55E"

st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
password = st.text_input("Enter access code", type="password")

if password != "veri2026":
    st.stop()

intro_text = """
## African Market Indices (AM100, AM200 & AM300)

### Overview

The African Market Indices platform provides a **liquidity-driven, investable benchmark system** designed to represent the most accessible equity opportunities across African markets.

The indices are constructed to reflect **real-world investability**, prioritising securities that institutional and professional investors can realistically access, trade, and scale.

---

### Index Structure

* **AM100 (Core Index)**
  A concentrated, institutional-grade index of the **100 most liquid and investable equities** across Africa.

* **AM200 (Expansion Index)**
  A broader index capturing the **next 100 securities**, offering exposure to emerging, frontier, and mid-cap opportunities.

* **AM300 (All Share)**
  A flagship **all-share total return index** combining the core and expansion sleeves into a unified benchmark.

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

#### Liquidity-Driven Selection

Constituents are ranked using a proprietary liquidity model:

**Liquidity Score = Traded Value x Participation^2**

This ensures:

* High turnover securities are prioritised
* Illiquid names are systematically excluded
* Rankings reflect **true execution capacity**, not just market size

---

#### Index Construction Rules

* **Monthly Rebalance**
* **Entry/Exit Buffers** to reduce turnover
* **Turnover Target:** ~2-4% per rebalance
* **Country Cap:** 40% maximum exposure
* **No synthetic smoothing or interpolation**

---

### Eligibility Criteria

To qualify for inclusion, securities must:

* Exhibit consistent trading activity
* Meet minimum liquidity thresholds
* Have reliable and continuous price data
* Be accessible to institutional investors

---

### Inclusion & Removal

**Entry into Index:**

* Achieve sufficient liquidity ranking
* Sustain trading consistency
* Pass buffer thresholds

**Removal from Index:**

* Drop below liquidity thresholds
* Exhibit deteriorating trading activity
* Fail to maintain ranking within buffer range

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

* Liquidity-first methodology
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

st.write(os.listdir())

with st.expander("📘 Methodology & Overview", expanded=False):
    st.markdown(intro_text)

col1, col2 = st.columns([0.8, 9])

with col1:
    st.image(logo_path, width=60)

with col2:
    st.markdown("""
    <div style="line-height: 1.2;">
        <span style="font-size:16px; font-weight:600;">African Market Indices</span><br>
        <span style="font-size:12px; color:#9AA4AF;">
        AM100 / AM200 / AM300 Total Return Indices
        </span>
    </div>
    """, unsafe_allow_html=True)
st.markdown("""
<style>
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
    font-size: 11px;
    color: #9AA4AF;
    line-height: 1.1;
    margin-bottom: 0.15rem;
}

.kpi-value {
    font-size: 18px;
    font-weight: 700;
    line-height: 1.1;
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

st.write("Data Last Updated:", am100.index.max())

am100_df = am100.rename("Index Level").reset_index()
am200_df = am200.rename("Index Level").reset_index()
am300_df = am300.rename("Index Level").reset_index()

today = am100_df["Date"].max()

# Rolling 10Y
start_10y = today - pd.DateOffset(years=10)

# Fixed periods
start_2016 = pd.Timestamp("2016-01-01")
start_2021 = pd.Timestamp("2021-01-01")

# AM100
am100_cagr_10y = calculate_cagr(am100_df, start_10y)
am100_cagr_2016 = calculate_cagr(am100_df, start_2016)
am100_cagr_5y = calculate_cagr(am100_df, start_2021)

# AM200
am200_cagr_10y = calculate_cagr(am200_df, start_10y)
am200_cagr_2016 = calculate_cagr(am200_df, start_2016)
am200_cagr_5y = calculate_cagr(am200_df, start_2021)

# AM300
am300_cagr_10y = calculate_cagr(am300_df, start_10y)
am300_cagr_2016 = calculate_cagr(am300_df, start_2016)
am300_cagr_5y = calculate_cagr(am300_df, start_2021)


@st.cache_data
def load_history(am100_version=None, am200_version=None, am300_version=None):
    am100_hist = pd.read_excel("output/AM100_history.xlsx")
    am200_hist = pd.read_excel("output/AM200_history.xlsx")
    am300_hist = pd.read_excel("output/AM300_history.xlsx")
    return am100_hist, am200_hist, am300_hist


am100_hist, am200_hist, am300_hist = load_history(
    get_file_version("output/AM100_history.xlsx"),
    get_file_version("output/AM200_history.xlsx"),
    get_file_version("output/AM300_history.xlsx"),
)
latest_date = am100_hist["Date"].max()

am100_latest = am100_hist[am100_hist["Date"] == latest_date]
am200_latest = am200_hist[am200_hist["Date"] == latest_date]
am300_latest = am300_hist[am300_hist["Date"] == latest_date]

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

# Visual-only smoothing for plotting. This does not affect stored data or metrics.
am100_plot = am100.copy()
am200_plot = am200.copy()
am200_plot = am200_plot.interpolate()


st.caption("Key Insights")

col1, col2, col3 = st.columns(3)

col1.info("AM100: Concentrated, institutional-grade core index")
col2.info("AM200: Expansion sleeve for broader frontier exposure")
col3.info("AM300: All-share flagship total return benchmark")

factsheet_mode = st.toggle("📄 Factsheet Mode", value=False)


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

col1, col2, col3 = st.columns(3)


def color(val):
    if val is None:
        return "white"
    return "lime" if val > 0 else "red"


def styled(val):
    if val is None:
        return "-"
    return f"<span style='color:{color(val)}'>{val:.2%}</span>"


with col1:
    st.markdown("**10Y Rolling**")
    st.markdown(f"AM100: {styled(am100_cagr_10y)}", unsafe_allow_html=True)
    st.markdown(f"AM200: {styled(am200_cagr_10y)}", unsafe_allow_html=True)
    st.markdown(f"AM300: {styled(am300_cagr_10y)}", unsafe_allow_html=True)

with col2:
    st.markdown("**Since 2016**")
    st.markdown(f"AM100: {styled(am100_cagr_2016)}", unsafe_allow_html=True)
    st.markdown(f"AM200: {styled(am200_cagr_2016)}", unsafe_allow_html=True)
    st.markdown(f"AM300: {styled(am300_cagr_2016)}", unsafe_allow_html=True)

with col3:
    st.markdown("**Since 2021 (5Y)**")
    st.markdown(f"AM100: {styled(am100_cagr_5y)}", unsafe_allow_html=True)
    st.markdown(f"AM200: {styled(am200_cagr_5y)}", unsafe_allow_html=True)
    st.markdown(f"AM300: {styled(am300_cagr_5y)}", unsafe_allow_html=True)

comparison_df = pd.DataFrame(
    {
        "Metric": ["CAGR", "Volatility", "Sharpe", "Max Drawdown"],
        "AM100": [f"{cagr100:.2%}", f"{vol100:.2%}", f"{sharpe100:.2f}", f"{dd100:.2%}"],
        "AM200": [f"{cagr200:.2%}", f"{vol200:.2%}", f"{sharpe200:.2f}", f"{dd200:.2%}"],
        "AM300": [f"{cagr300:.2%}", f"{vol300:.2%}", f"{sharpe300:.2f}", f"{dd300:.2%}"],
    }
)
st.dataframe(comparison_df, use_container_width=True, hide_index=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.write("AM100")
    st.markdown(
        f"**AM100 CAGR:** <span style='{color_return(cagr100)}'>{cagr100:.2%}</span>",
        unsafe_allow_html=True,
    )
    st.metric("Sharpe Ratio", f"{sharpe100:.2f}")
    st.metric("Max Drawdown", f"{dd100:.2%}")

with col2:
    st.write("AM200")
    st.markdown(
        f"**AM200 CAGR:** <span style='{color_return(cagr200)}'>{cagr200:.2%}</span>",
        unsafe_allow_html=True,
    )
    st.metric("Sharpe Ratio", f"{sharpe200:.2f}")
    st.metric("Max Drawdown", f"{dd200:.2%}")

with col3:
    st.write("AM300")
    st.markdown(
        f"**AM300 CAGR:** <span style='{color_return(cagr300)}'>{cagr300:.2%}</span>",
        unsafe_allow_html=True,
    )
    st.metric("Sharpe Ratio", f"{sharpe300:.2f}")
    st.metric("Max Drawdown", f"{dd300:.2%}")

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

top_country_am100 = am100_latest.groupby("Country")["Weight"].sum().idxmax()
top_country_am200 = am200_latest.groupby("Country")["Weight"].sum().idxmax()
top_country_am300 = am300_latest.groupby("Country")["Weight"].sum().idxmax()
am100_country = (
    am100_latest.groupby("Country")["Weight"].sum().sort_values(ascending=False)
)
am200_country = (
    am200_latest.groupby("Country")["Weight"].sum().sort_values(ascending=False)
)
am300_country = (
    am300_latest.groupby("Country")["Weight"].sum().sort_values(ascending=False)
)
latest_am100 = am100.iloc[-1]
latest_am200 = am200.iloc[-1]
latest_am300 = am300.iloc[-1]

ticker_text = f"""
AM100: {latest_am100:.2f} ({ret100:.2%}) •
AM200: {latest_am200:.2f} ({ret200:.2%}) •
AM300: {latest_am300:.2f} ({ret300:.2%}) •
Top Country AM100: {top_country_am100} •
Top Country AM200: {top_country_am200} •
Top Country AM300: {top_country_am300}
"""

obs.append(f"AM100 is most exposed to {top_country_am100}.")
obs.append(f"AM200 shows strongest exposure to {top_country_am200}.")
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


am100_adv_usd, am100_capacity = external_capacity_metrics(am100_latest)
am200_adv_usd, am200_capacity = external_capacity_metrics(am200_latest)
am300_adv_usd, am300_capacity = external_capacity_metrics(am300_latest)

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
            "Metric": ["Average Daily Traded Value (USD)", "Investable Capacity (USD, 20%)"],
            "AM100": [f"{am100_adv_usd:,.0f}", f"{am100_capacity:,.0f}"],
            "AM200": [f"{am200_adv_usd:,.0f}", f"{am200_capacity:,.0f}"],
            "AM300": [f"{am300_adv_usd:,.0f}", f"{am300_capacity:,.0f}"],
        }
    )
    st.dataframe(capacity_display, use_container_width=True, hide_index=True)
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
                st.metric(
                    f"{name} Risk",
                    row["Rating"],
                    help=f"Multi-factor risk score: {row['Risk Score']:.2f}",
                )
                st.caption(
                    f"Score {row['Risk Score']:.2f} | Investable Capacity {row['Capacity']:,.0f}"
                )
            else:
                st.metric(f"{name} Risk", "Unavailable")

    st.markdown("### AM300 Risk Snapshot")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("AM300 Volatility", f"{vol300:.2%}")
    col2.metric("AM300 Drawdown", f"{dd300:.2%}")
    col3.metric("AM300 Sharpe", f"{sharpe300:.2f}")
    col4.metric("AM300 Risk", am300_label)

    st.markdown("## Rolling Risk")

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

    st.markdown("## Return Decomposition")
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

    allocator_cols = st.columns(3)
    for i, model_name in enumerate(MODEL_WEIGHTS):
        with allocator_cols[i]:
            st.markdown(f"### {model_name}")
            st.write(f"**Objective:** {MODEL_METADATA[model_name]['Objective']}")
            st.write(MODEL_METADATA[model_name]["Characteristics"])
            for index_name, weight in MODEL_WEIGHTS[model_name].items():
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
    st.write(
        {
            "CAGR": cagr,
            "Volatility": vol,
            "Sharpe": sharpe,
            "Drawdown": dd,
        }
    )

    st.markdown("### MPS Portfolio Builder")
    model = st.selectbox(
        "Select Portfolio",
        list(MODEL_WEIGHTS.keys()),
        key="mps_model_select",
    )
    weights = MODEL_WEIGHTS[model]
    st.write("Weights:", weights)

    port_returns = build_portfolio(allocator_returns, weights)
    port_index = build_index(port_returns)
    cagr, vol, sharpe, dd = portfolio_metrics(port_index)

    st.write(
        {
            "CAGR": cagr,
            "Volatility": vol,
            "Sharpe": sharpe,
            "Drawdown": dd,
        }
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
    frontier = simulate_random_frontier(prepare_returns_frame())
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frontier["Vol"],
            y=frontier["Return"],
            mode="markers",
            marker=dict(size=3, color="#7DD3FC", opacity=0.55),
            name="Frontier",
        )
    )
    for model_name, color in model_colors.items():
        row = model_metrics[model_metrics["Model"] == model_name].iloc[0]
        fig.add_trace(
            go.Scatter(
                x=[row["Volatility"]],
                y=[row["CAGR"]],
                mode="markers+text",
                text=[model_name],
                textposition="top center",
                marker=dict(size=9, color=color),
                name=model_name,
            )
        )
    fig.update_layout(
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font=dict(size=10, color="#CCCCCC"),
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        xaxis_title="Volatility",
        yaxis_title="Expected Return",
    )
    st.plotly_chart(fig, use_container_width=True)

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

if factsheet_mode:

    st.markdown("# AM Indices Factsheet")
    st.write("AM100 (Core) vs AM200 (Expansion) vs AM300 (All Share)")

    # -------------------------
    # KEY METRICS (TOP ROW)
    # -------------------------
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.markdown(
            f"**AM100 CAGR:** <span style='{color_return(cagr100)}'>{cagr100:.2%}</span>",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"**AM200 CAGR:** <span style='{color_return(cagr200)}'>{cagr200:.2%}</span>",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"**AM300 CAGR:** <span style='{color_return(cagr300)}'>{cagr300:.2%}</span>",
            unsafe_allow_html=True,
        )
    col4.metric("AM100 Vol", f"{vol100:.2%}")
    col5.metric("AM200 Vol", f"{vol200:.2%}")
    col6.metric("AM300 Vol", f"{vol300:.2%}")

    col7, col8, col9 = st.columns(3)
    col7.metric("AM100 Max DD", f"{dd100:.2%}")
    col8.metric("AM200 Max DD", f"{dd200:.2%}")
    col9.metric("AM300 Max DD", f"{dd300:.2%}")

    # -------------------------
    # PERFORMANCE (MAIN CHART)
    # -------------------------
    st.caption("Performance")

    fig, ax = plt.subplots(figsize=(6, 2.6))
    style_chart(fig, ax)
    ax.plot(am100_plot, label="AM100", color=AM100_COLOR, linewidth=2)
    ax.plot(am200_plot, label="AM200", color=AM200_COLOR, linewidth=2)
    ax.plot(am300, label="AM300", color=AM300_COLOR, linewidth=2)
    ax.legend(frameon=False, fontsize=10)
    fig.tight_layout()

    st.pyplot(fig, use_container_width=True)

    # -------------------------
    # COUNTRY EXPOSURE
    # -------------------------
    st.caption("Country Allocation")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**AM100**")
        fig, ax = plt.subplots(figsize=(6, 2.6))
        style_chart(fig, ax)
        ax.bar(am100_country.index, am100_country.values, color=AM100_COLOR)
        ax.tick_params(axis="x", rotation=90)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

    with col2:
        st.markdown("**AM200**")
        fig, ax = plt.subplots(figsize=(6, 2.6))
        style_chart(fig, ax)
        ax.bar(am200_country.index, am200_country.values, color=AM200_COLOR)
        ax.tick_params(axis="x", rotation=90)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

    # -------------------------
    # TOP HOLDINGS
    # -------------------------
    st.caption("Top Holdings")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**AM100 Top 5**")
        factsheet_am100 = (
            am100_latest.sort_values("Weight", ascending=False)[["Company", "Weight"]]
            .head(5)
            .reset_index(drop=True)
        )
        factsheet_am100["Weight"] = factsheet_am100["Weight"].map("{:.2%}".format)
        st.dataframe(display_with_row_numbers(factsheet_am100), use_container_width=True)

    with col2:
        st.markdown("**AM200 Top 5**")
        factsheet_am200 = (
            am200_latest.sort_values("Weight", ascending=False)[["Company", "Weight"]]
            .head(5)
            .reset_index(drop=True)
        )
        factsheet_am200["Weight"] = factsheet_am200["Weight"].map("{:.2%}".format)
        st.dataframe(display_with_row_numbers(factsheet_am200), use_container_width=True)

    # -------------------------
    # KEY OBSERVATIONS
    # -------------------------
    st.caption("Key Observations")

    st.write(f"• AM300 sits between AM100 and AM200 with CAGR {cagr300:.2%}")
    st.write(f"• AM200 CAGR ({cagr200:.2%}) vs AM100 ({cagr100:.2%})")
    st.write("• AM300 offers broader flagship exposure with balanced risk")
    st.write(f"• AM100 concentrated in {top_country_am100}")
    st.write(f"• AM200 diversified across {top_country_am200} and others")
    st.write(f"• AM300 flagship exposure led by {top_country_am300}")

    # -------------------------
    # FOOTER
    # -------------------------
    st.markdown("---")
    st.caption("Veri AM Indices • Liquidity-Based African Benchmark")

    st.stop()


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
        fig, ax = plt.subplots(figsize=(6, 2.4))
        style_chart(fig, ax)
        ax.bar(am200_country.index, am200_country.values, color=AM200_COLOR)
        ax.tick_params(axis="x", rotation=90)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

    st.markdown("### Country Exposure")

    heatmap_view = st.toggle("AM100 vs AM200", value=False)

    if heatmap_view:
        country_df = am200_country.rename_axis("Country").reset_index(name="Weight")
    else:
        country_df = am100_country.rename_axis("Country").reset_index(name="Weight")

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

    country_df["Country"] = country_df["Country"].map(country_name_map).fillna(
        country_df["Country"].str.title()
    )

    fig = px.choropleth(
        country_df,
        locations="Country",
        locationmode="country names",
        color="Weight",
        color_continuous_scale="Viridis",
    )

    fig.update_layout(
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

    st.markdown("---")

with overview_tab:
    st.markdown("## 🏦 Constituents")

    col1, col2 = st.columns(2)

    with col1:
        st.caption("AM100 Top 10")

        am100_top = (
            am100_latest.sort_values("Weight", ascending=False)[
                ["Company", "Country", "Weight"]
            ].head(10)
        )
        am100_top["Weight"] = am100_top["Weight"].map("{:.2%}".format)

        st.dataframe(display_with_row_numbers(am100_top), use_container_width=True)

    with col2:
        st.caption("AM200 Top 10")

        am200_top = (
            am200_latest.sort_values("Weight", ascending=False)[
                ["Company", "Country", "Weight"]
            ].head(10)
        )
        am200_top["Weight"] = am200_top["Weight"].map("{:.2%}".format)

        st.dataframe(display_with_row_numbers(am200_top), use_container_width=True)

    with st.expander("🔍 View Full AM100 Constituents (Top 100)"):
        am100_full = (
            am100_latest.sort_values("Rank", ascending=True)[
                ["Company", "Country", "Weight", "Rank"]
            ]
        )
        search = st.text_input("Search AM100", key="am100_search")
        if search:
            am100_full = am100_full[
                am100_full["Company"].str.contains(search, case=False)
            ]
        am100_full["Weight"] = am100_full["Weight"].map("{:.2%}".format)
        st.dataframe(display_with_row_numbers(am100_full), use_container_width=True)

    with st.expander("🔍 View Full AM200 Constituents (101–200)"):
        am200_full = (
            am200_latest.sort_values("Rank", ascending=True)[
                ["Company", "Country", "Weight", "Rank"]
            ]
        )
        search = st.text_input("Search AM200", key="am200_search")
        if search:
            am200_full = am200_full[
                am200_full["Company"].str.contains(search, case=False)
            ]
        am200_full["Weight"] = am200_full["Weight"].map("{:.2%}".format)
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
