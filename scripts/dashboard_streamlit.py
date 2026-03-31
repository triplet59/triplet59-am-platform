import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate


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


@st.cache_data
def load_country_exposure(path):
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


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(BASE_DIR, "..", "assets", "veri_logo.png")
AM100_COLOR = "#4DA3FF"
AM200_COLOR = "#FF9F1C"

st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
password = st.text_input("Enter access code", type="password")

if password != "veri2026":
    st.stop()

intro_text = """
## African Market Indices (AM100 & AM200)

### Overview

The African Market Indices platform provides a **liquidity-driven, investable benchmark system** designed to represent the most accessible equity opportunities across African markets.

The indices are constructed to reflect **real-world investability**, prioritising securities that institutional and professional investors can realistically access, trade, and scale.

---

### Index Structure

* **AM100 (Core Index)**
  A concentrated, institutional-grade index of the **100 most liquid and investable equities** across Africa.

* **AM200 (Expansion Index)**
  A broader index capturing the **next 100 securities**, offering exposure to emerging, frontier, and mid-cap opportunities.

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
        AM100 (Core) vs AM200 (Expansion)
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
def load_data():
    def load_total_return(path, label):
        if not os.path.exists(path):
            st.error(f"No data file found for {label}: {path}")
            return None

        df = pd.read_csv(path, parse_dates=["Date"])
        if "Index Level" not in df.columns:
            st.error(f"{label} file is missing required column: Index Level")
            return None

        st.sidebar.success(f"{label} loaded: {path}")
        series = df.sort_values("Date").set_index("Date")["Index Level"]
        return series

    am100 = load_total_return("output/AM100_total_return.csv", "AM100")
    am200 = load_total_return("output/AM200_total_return.csv", "AM200")

    if am100 is None or am200 is None:
        st.stop()

    return am100, am200


am100, am200 = load_data()

am100_df = am100.rename("Index Level").reset_index()
am200_df = am200.rename("Index Level").reset_index()

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


@st.cache_data
def load_history():
    am100_hist = pd.read_excel("output/AM100_history.xlsx")
    am200_hist = pd.read_excel("output/AM200_history.xlsx")
    return am100_hist, am200_hist


am100_hist, am200_hist = load_history()
latest_date = am100_hist["Date"].max()

am100_latest = am100_hist[am100_hist["Date"] == latest_date]
am200_latest = am200_hist[am200_hist["Date"] == latest_date]

# Latest returns (daily)
if len(am100) > 1:
    ret100 = am100.pct_change().iloc[-1]
else:
    ret100 = 0

if len(am200) > 1:
    ret200 = am200.pct_change().iloc[-1]
else:
    ret200 = 0


# ----------------------------
# METRICS
# ----------------------------
@st.cache_data
def load_metric_snapshot(path):
    df = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date")
    index_series = df["Index Level"]
    returns = index_series.pct_change().dropna()

    start = index_series.iloc[0]
    end = index_series.iloc[-1]
    years = (df["Date"].iloc[-1] - df["Date"].iloc[0]).days / 365

    cagr = (end / start) ** (1 / years) - 1
    vol = returns.std() * np.sqrt(252)
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0
    drawdown = (index_series / index_series.cummax()) - 1
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
    content.append(Paragraph("AM100 vs AM200 Report", styles["Title"]))
    content.append(Paragraph(f"CAGR AM100: {cagr100:.2%}", styles["Normal"]))
    content.append(Paragraph(f"CAGR AM200: {cagr200:.2%}", styles["Normal"]))

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

# Visual-only smoothing for plotting. This does not affect stored data or metrics.
am100_plot = am100.copy()
am200_plot = am200.copy()
am200_plot = am200_plot.interpolate()


st.caption("Key Insights")

col1, col2, col3 = st.columns(3)

col1.info("AM100: Concentrated, institutional-grade core index")
col2.info("AM200: Broader exposure across emerging and frontier markets")
col3.info("Liquidity-driven selection ensures real investability")

factsheet_mode = st.toggle("📄 Factsheet Mode", value=False)


# ----------------------------
# METRICS DISPLAY
# ----------------------------
cagr100, vol100, sharpe100, dd100 = load_metric_snapshot("output/AM100_total_return.csv")
cagr200, vol200, sharpe200, dd200 = load_metric_snapshot("output/AM200_total_return.csv")

col1, col2, col3, col4 = st.columns(4)

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
    <div class="kpi-label">Vol (AM100)</div>
    <div class="kpi-value">{vol100:.2%}</div>
</div>
"""
,
    unsafe_allow_html=True,
)

col4.markdown(
    f"""
<div class="kpi-card">
    <div class="kpi-label">Vol (AM200)</div>
    <div class="kpi-value">{vol200:.2%}</div>
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

with col2:
    st.markdown("**Since 2016**")
    st.markdown(f"AM100: {styled(am100_cagr_2016)}", unsafe_allow_html=True)
    st.markdown(f"AM200: {styled(am200_cagr_2016)}", unsafe_allow_html=True)

with col3:
    st.markdown("**Since 2021 (5Y)**")
    st.markdown(f"AM100: {styled(am100_cagr_5y)}", unsafe_allow_html=True)
    st.markdown(f"AM200: {styled(am200_cagr_5y)}", unsafe_allow_html=True)

col1, col2 = st.columns(2)

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
am100_country = (
    am100_latest.groupby("Country")["Weight"].sum().sort_values(ascending=False)
)
am200_country = (
    am200_latest.groupby("Country")["Weight"].sum().sort_values(ascending=False)
)
latest_am100 = am100.iloc[-1]
latest_am200 = am200.iloc[-1]

ticker_text = f"""
AM100: {latest_am100:.2f} ({ret100:.2%}) •
AM200: {latest_am200:.2f} ({ret200:.2%}) •
Top Country AM100: {top_country_am100} •
Top Country AM200: {top_country_am200}
"""

obs.append(f"AM100 is most exposed to {top_country_am100}.")
obs.append(f"AM200 shows strongest exposure to {top_country_am200}.")

for o in obs:
    st.write(f"• {o}")

st.markdown(
    f"<div class='ticker'><span>{ticker_text}</span></div>",
    unsafe_allow_html=True,
)

if factsheet_mode:

    st.markdown("# AM Indices Factsheet")
    st.write("AM100 (Core) vs AM200 (Expansion)")

    # -------------------------
    # KEY METRICS (TOP ROW)
    # -------------------------
    col1, col2, col3, col4 = st.columns(4)

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
    col3.metric("AM100 Vol", f"{vol100:.2%}")
    col4.metric("AM200 Vol", f"{vol200:.2%}")

    col5, col6 = st.columns(2)
    col5.metric("AM100 Max DD", f"{dd100:.2%}")
    col6.metric("AM200 Max DD", f"{dd200:.2%}")

    # -------------------------
    # PERFORMANCE (MAIN CHART)
    # -------------------------
    st.caption("Performance")

    fig, ax = plt.subplots(figsize=(6, 2.6))
    style_chart(fig, ax)
    ax.plot(am100_plot, label="AM100", color=AM100_COLOR, linewidth=2)
    ax.plot(am200_plot, label="AM200", color=AM200_COLOR, linewidth=2)
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

    st.write(f"• AM200 CAGR ({cagr200:.2%}) vs AM100 ({cagr100:.2%})")
    st.write("• Higher volatility in AM200 reflects broader exposure")
    st.write(f"• AM100 concentrated in {top_country_am100}")
    st.write(f"• AM200 diversified across {top_country_am200} and others")

    # -------------------------
    # FOOTER
    # -------------------------
    st.markdown("---")
    st.caption("Veri AM Indices • Liquidity-Based African Benchmark")

    st.stop()


# ----------------------------
# PERFORMANCE & RISK CHARTS
# ----------------------------
st.markdown("## Performance & Risk")

col1, col2 = st.columns(2)

with col1:
    st.caption("Index Performance")
    fig, ax = plt.subplots(figsize=(6, 2.4))
    style_chart(fig, ax)
    ax.plot(am100_plot, label="AM100", color=AM100_COLOR, linewidth=2)
    ax.plot(am200_plot, label="AM200", color=AM200_COLOR, linewidth=2)
    ax.legend(frameon=False, fontsize=10)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

with col2:
    st.caption("Log Performance")
    fig, ax = plt.subplots(figsize=(6, 2.4))
    style_chart(fig, ax)
    ax.plot(am100_plot, color=AM100_COLOR, linewidth=2)
    ax.plot(am200_plot, color=AM200_COLOR, linewidth=2)
    ax.set_yscale("log")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

st.caption("Index composition shift due to liquidity-driven rebalance")

col1, col2 = st.columns(2)

with col1:
    st.caption("Drawdown")
    fig, ax = plt.subplots(figsize=(6, 2.4))
    style_chart(fig, ax)
    ax.plot(am100 / am100.cummax() - 1, color=AM100_COLOR, linewidth=2)
    ax.plot(am200 / am200.cummax() - 1, color=AM200_COLOR, linewidth=2)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    st.caption("Drawdown calculated as peak-to-trough decline based on daily index levels.")

with col2:
    st.caption("Volatility (30D)")
    fig, ax = plt.subplots(figsize=(6, 2.4))
    style_chart(fig, ax)
    ax.plot(am100.pct_change().rolling(30).std() * np.sqrt(252), color=AM100_COLOR, linewidth=2)
    ax.plot(am200.pct_change().rolling(30).std() * np.sqrt(252), color=AM200_COLOR, linewidth=2)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)


# ----------------------------
# COUNTRY EXPOSURE
# ----------------------------
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

st.caption("Country Exposure Heatmap")

heatmap_view = st.toggle("AM100 vs AM200", value=False)

if heatmap_view:
    country_df = am200_country.rename_axis("Country").reset_index(name="Weight")
    heatmap_title = "Country Exposure - AM200"
else:
    country_df = am100_country.rename_axis("Country").reset_index(name="Weight")
    heatmap_title = "Country Exposure - AM100"

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
    color_continuous_scale="Blues",
    title=heatmap_title,
)

fig.update_layout(
    margin=dict(l=0, r=0, t=30, b=0),
    font=dict(size=10),
)

st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# TOP HOLDINGS
# ----------------------------
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

st.caption("Methodology & Rules")

st.markdown("### Methodology Flow")

st.markdown(
    """
**Data Pipeline -> Index Construction**

1. **Raw Market Data**
   - Prices, volumes, corporate actions

2. **Cleaning & Validation**
   - Outlier detection
   - Missing data handling

3. **Liquidity Scoring**
   - Traded Value x Participation^2

4. **Ranking Engine**
   - Cross-market comparability

5. **Eligibility Filters**
   - Liquidity thresholds
   - Data consistency

6. **Index Construction**
   - AM100 (Top 100)
   - AM200 (Next 100)

7. **Rebalancing**
   - Monthly
   - Buffer-controlled turnover
"""
)

col1, col2 = st.columns(2)

with col1:
    st.write("Methodology Overview")

    st.write(
        """
- Liquidity-driven selection using traded value  
- Participation-adjusted liquidity scoring  
- Monthly rebalancing  
- Country caps applied iteratively  
- No interpolation of missing data  
"""
    )

with col2:
    st.write("Qualification & Ranking Rules")

    st.write(
        """
**Eligibility:**
- Minimum trading history required
- Valid price series (no synthetic data)
- Volume or value data available

**Liquidity Model:**
- 30 valid trading days rolling window
- Liquidity = traded value × participation²
- Country/regime scaling applied

**Ranking:**
- Ranked by adjusted liquidity score
- No normalization distortions
- Full uniqueness enforced

**Selection:**
- Top 100 → AM100
- Next 100 → AM200

**Constraints:**
- Country cap: 40%
- Turnover control applied
- Monthly rebalance with buffers
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
