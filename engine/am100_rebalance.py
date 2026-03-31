import pandas as pd
import numpy as np

MASTER_FILE = "output/master.xlsx"
OUTPUT_FILE = "output/AM100_history.xlsx"
AM200_OUTPUT_FILE = "output/AM200_history.xlsx"
LIQUIDITY_RANKING_FILE = "output/liquidity_ranking.csv"
LIQUIDITY_REPORT_FILE = "output/liquidity_report.csv"
RANKING_REPORT_FILE = "output/am100_ranking.csv"
UNIVERSE_RANKED_FILE = "output/am_universe_ranked.csv"
REBALANCE_AUDIT_PATTERN = "output/rebalance_audit_{yyyymm}.csv"
REPORT_FILE = "output/am100_report.txt"
COUNTRY_EXPOSURE_FILE = "output/country_exposure.csv"

MAX_STOCK_WEIGHT = 0.10
MAX_COUNTRY_WEIGHT = 0.40
MAX_PER_COUNTRY = int(100 * MAX_COUNTRY_WEIGHT)
MIN_DOLLAR_LIQUIDITY = 5000
MIN_PARTICIPATION = 0.30
MAX_LOW_LIQ_EXPOSURE = 0.25
TARGET_N = 100
ENTRY_BUFFER = 100
EXIT_BUFFER = 120
MAX_FALLBACK_RANK = 80
RELAXED_FALLBACK_RANK = 90
MAX_TURNOVER = 0.20
PROTECTED_RANK = 60

REGIME_MAP = {
    "SOUTH AFRICA": "HIGH",
    "NIGERIA": "MID",
    "KENYA": "MID",
    "EGYPT": "MID",
    "MOROCCO": "MID",
    "RWANDA": "LOW",
    "TANZANIA": "LOW",
    "UGANDA": "LOW",
    "ZAMBIA": "LOW",
    "MALAWI": "LOW",
    "BOTSWANA": "LOW",
    "ZIMBABWE": "LOW",
    "GHANA": "LOW",
}

REGIME_WEIGHTS = {
    "HIGH": 1.0,
    "MID": 1.0,
    "LOW": 1.0,
}

COUNTRY_SCALING = {
    "SOUTH AFRICA": 1,
    "NIGERIA": 1,
    "KENYA": 1,
    "TANZANIA": 0.001,
    "UGANDA": 0.001,
    "MALAWI": 0.0001,
    "ZAMBIA": 0.001,
}


def compute_liquidity(price, volume, scale=1.0, window=30):
    traded_value = price * volume
    adj_value = traded_value * scale
    valid = adj_value.notna()

    # Rolling average of traded value using only valid trades, then realign.
    adj_rolling_value = adj_value[valid].rolling(window).mean().reindex(adj_value.index).ffill()
    adj_rolling_value = adj_rolling_value.clip(
        lower=adj_rolling_value.quantile(0.05),
        upper=adj_rolling_value.quantile(0.95),
    )

    # Trading frequency (participation ratio)
    trade_count = valid.rolling(window).sum()
    participation = trade_count / window

    # Final liquidity score
    liquidity = adj_rolling_value * (participation ** 2)

    return liquidity.ffill(), participation.ffill(), adj_rolling_value.ffill(), trade_count.ffill()


def extract_country(company_name):
    return company_name.rsplit("(", 1)[-1].rstrip(")")


def assign_regime(country):
    return REGIME_MAP.get(country, "LOW")


def normalize_by_regime(df):
    def safe_scale(x):
        values = x.to_numpy(dtype=float)

        if len(values) == 0:
            return x

        q75 = np.percentile(values, 75)

        if pd.isna(q75) or q75 == 0:
            return x * 0

        scaled = values / q75
        scaled = np.clip(scaled, 0, 5)

        return pd.Series(scaled, index=x.index)

    df["NormLiquidity"] = df.groupby(["Date", "Regime"])["RawLiquidity"].transform(safe_scale)
    return df


def apply_stock_cap(portfolio):
    weights = portfolio["Weight"].copy()

    while True:
        over_cap = weights > MAX_STOCK_WEIGHT

        if not over_cap.any():
            break

        excess = (weights[over_cap] - MAX_STOCK_WEIGHT).sum()
        weights[over_cap] = MAX_STOCK_WEIGHT

        under_cap = weights < MAX_STOCK_WEIGHT

        if under_cap.sum() == 0:
            break

        redistribution = weights[under_cap] / weights[under_cap].sum()
        weights[under_cap] += redistribution * excess

    portfolio["Weight"] = weights
    return portfolio


def apply_country_cap_iterative(portfolio, max_country_weight=MAX_COUNTRY_WEIGHT):
    for _ in range(20):
        country_weights = portfolio.groupby("Country")["Weight"].sum()
        overweight = country_weights[country_weights > max_country_weight]

        if overweight.empty:
            break

        total_excess = 0.0

        for country, weight in overweight.items():
            total_excess += weight - max_country_weight
            mask = portfolio["Country"] == country
            portfolio.loc[mask, "Weight"] *= max_country_weight / weight

        under_mask = ~portfolio["Country"].isin(overweight.index)
        under_total = portfolio.loc[under_mask, "Weight"].sum()

        if under_total > 0:
            portfolio.loc[under_mask, "Weight"] *= (1 + total_excess / under_total)

        portfolio["Weight"] /= portfolio["Weight"].sum()

    return portfolio


def apply_weight_constraints(portfolio):
    total_liquidity = portfolio["Liquidity Score"].sum()

    if total_liquidity == 0:
        return None

    portfolio["Weight"] = portfolio["Liquidity Score"] / total_liquidity
    portfolio["Country"] = portfolio["Company"].str.extract(r"\((.*?)\)")
    portfolio["Regime"] = portfolio["Country"].map(assign_regime)
    portfolio = apply_stock_cap(portfolio)
    portfolio = apply_country_cap_iterative(portfolio)

    for _ in range(10):
        regime_weights = portfolio.groupby("Regime")["Weight"].sum()
        low_liq_weight = regime_weights.get("LOW", 0.0)

        if low_liq_weight <= MAX_LOW_LIQ_EXPOSURE + 1e-6:
            break

        excess = low_liq_weight - MAX_LOW_LIQ_EXPOSURE
        low_mask = portfolio["Regime"] == "LOW"
        non_low_mask = ~low_mask

        portfolio.loc[low_mask, "Weight"] *= MAX_LOW_LIQ_EXPOSURE / low_liq_weight

        non_low_total = portfolio.loc[non_low_mask, "Weight"].sum()
        if non_low_total > 0:
            portfolio.loc[non_low_mask, "Weight"] *= (1 + excess / non_low_total)

        portfolio["Weight"] /= portfolio["Weight"].sum()

    portfolio = apply_country_cap_iterative(portfolio)
    portfolio["Weight"] /= portfolio["Weight"].sum()
    return portfolio

# ================================
# LOAD MASTER
# ================================
df = pd.read_excel(MASTER_FILE)
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
df = df.sort_values("Date")

# ================================
# IDENTIFY PRICE COLUMNS
# ================================
price_cols = [col for col in df.columns if "Price" in col]

results_all = []
am200_history = []
prev_portfolio = None
latest_universe = None
rebalance_audit_frames = []

# ================================
# MONTH-END REBALANCE DATES (FIXED)
# ================================
dates = df["Date"]

month_ends = (
    dates.drop_duplicates()
         .sort_values()
         .groupby([dates.dt.year, dates.dt.month])
         .max()
         .values
)

# ================================
# MAIN LOOP
# ================================
for rebalance_date in month_ends:

    df_cut = df[df["Date"] <= rebalance_date]

    records = []

    for col in price_cols:

        company = col.replace(" Price", "")
        company_country = extract_country(company)
        volume_col = col.replace("Price", "Volume")

        if volume_col not in df_cut.columns:
            continue

        price_series = df_cut[["Date", col]].dropna()

        if len(price_series) < 150:
            continue

        recent = price_series.tail(90)
        trading_days = recent[col].notna().sum()

        if trading_days < 15:
            continue

        valid_prices = recent[col].dropna()
        if len(valid_prices) < 10:
            continue

        latest_price = price_series[col].iloc[-1]

        merged = df_cut[[col, volume_col]].dropna().tail(30)

        if len(merged) < 10:
            continue

        price = df_cut[col]
        volume = df_cut[volume_col]
        scale = COUNTRY_SCALING.get(company_country, 1.0)
        liquidity, participation, avg_value, trade_count = compute_liquidity(
            price,
            volume,
            scale=scale,
            window=30,
        )
        liquidity_score = liquidity.dropna().iloc[-1] if liquidity.notna().any() else np.nan
        participation_score = (
            participation.dropna().iloc[-1] if participation.notna().any() else np.nan
        )
        avg_value_30d = avg_value.dropna().iloc[-1] if avg_value.notna().any() else np.nan
        trade_count_30 = trade_count.dropna().iloc[-1] if trade_count.notna().any() else np.nan

        if pd.isna(liquidity_score) or liquidity_score <= 0:
            continue

        if liquidity_score < MIN_DOLLAR_LIQUIDITY:
            continue

        if pd.isna(participation_score) or participation_score < MIN_PARTICIPATION:
            continue

        records.append({
            "Company": company,
            "Liquidity Score": liquidity_score,
            "Price": latest_price,
            "Trading Days (90d)": trading_days,
            "Participation": participation_score,
            "AvgDailyValue30d": avg_value_30d,
            "TradeCount30": trade_count_30,
        })

    if not records:
        continue

    universe = pd.DataFrame(records)
    universe["Date"] = rebalance_date
    universe["Country"] = universe["Company"].str.extract(r"\((.*?)\)")
    universe["Regime"] = universe["Country"].map(REGIME_MAP)
    universe["Regime"] = universe["Regime"].fillna("LOW")
    universe["RawLiquidity"] = universe["Liquidity Score"]
    print("\n=== TRUE LIQUIDITY TOP 10 (PRE-NORMALIZATION) ===")
    print(
        universe[["Company", "RawLiquidity"]]
        .sort_values("RawLiquidity", ascending=False)
        .head(10)
        .to_string(index=False)
    )
    universe = normalize_by_regime(universe)
    print("\n=== POST-NORMALIZATION ===")
    print(
        universe[["Company", "NormLiquidity"]]
        .sort_values("NormLiquidity", ascending=False)
        .head(10)
        .to_string(index=False)
    )
    universe["Liquidity Score"] = (
        universe["RawLiquidity"] * universe["Regime"].map(REGIME_WEIGHTS)
    )
    unique_ratio = universe["Liquidity Score"].nunique() / len(universe)
    print("\n=== UNIQUENESS CHECK ===")
    print(f"Universe size: {len(universe)}")
    print(f"Unique liquidity values: {universe['Liquidity Score'].nunique()}")
    print(f"Ratio: {unique_ratio:.2f}")
    assert unique_ratio > 0.8, "❌ Liquidity flattening detected"
    print("\n=== FINAL LIQUIDITY SCORE (USED FOR RANKING) ===")
    print(
        universe[["Company", "Liquidity Score"]]
        .sort_values("Liquidity Score", ascending=False)
        .head(10)
        .to_string(index=False)
    )
    universe = universe.sort_values("Liquidity Score", ascending=False)
    universe["Rank"] = universe["Liquidity Score"].rank(ascending=False, method="first")
    latest_universe = universe.copy()

    prev_constituents = set(prev_portfolio["Company"]) if prev_portfolio is not None else set()
    is_major_universe_change = (
        prev_portfolio is not None
        and abs(len(universe) - len(prev_constituents)) / max(len(prev_constituents), 1) > 0.30
    )

    selected = []

    for _, row in universe.sort_values("Rank").iterrows():
        company = row["Company"]
        rank = row["Rank"]

        if company in prev_constituents:
            if rank <= EXIT_BUFFER:
                selected.append(row)
        else:
            if rank <= ENTRY_BUFFER:
                selected.append(row)

        if len(selected) == TARGET_N:
            break

    if len(selected) < TARGET_N:
        selected_companies = [row["Company"] for row in selected]
        remaining = universe[~universe["Company"].isin(selected_companies)]
        remaining = remaining[remaining["Rank"] <= MAX_FALLBACK_RANK].sort_values("Rank")

        for _, row in remaining.iterrows():
            selected.append(row)
            if len(selected) == TARGET_N:
                break

    if len(selected) < TARGET_N:
        selected_companies = [row["Company"] for row in selected]
        remaining = universe[~universe["Company"].isin(selected_companies)]
        remaining = remaining[remaining["Rank"] <= RELAXED_FALLBACK_RANK].sort_values("Rank")

        for _, row in remaining.iterrows():
            selected.append(row)
            if len(selected) == TARGET_N:
                break

    portfolio = pd.DataFrame(selected)

    if portfolio.empty:
        continue

    # =========================
    # TURNOVER CONTROL MODULE
    # =========================
    new_constituents = set(portfolio["Company"])
    overlap = prev_constituents & new_constituents
    turnover = 1 - len(overlap) / TARGET_N
    print(f"{rebalance_date} turnover (pre-control): {turnover:.2%}")

    skip_turnover_control = False

    if len(prev_constituents) < TARGET_N * 0.8:
        skip_turnover_control = True

    if is_major_universe_change:
        skip_turnover_control = True

    if (turnover > MAX_TURNOVER) and not skip_turnover_control:
        entrants = list(new_constituents - prev_constituents)
        exits = list(prev_constituents - new_constituents)

        entrant_df = universe[universe["Company"].isin(entrants)].sort_values("Rank", ascending=False)
        exit_df = universe[universe["Company"].isin(exits)].sort_values("Rank", ascending=True)

        adjusted_constituents = set(new_constituents)

        for (_, weak_entrant), (_, strong_exit) in zip(entrant_df.iterrows(), exit_df.iterrows()):
            if turnover <= MAX_TURNOVER:
                break

            if weak_entrant["Rank"] <= 50:
                continue
            elif weak_entrant["Rank"] <= PROTECTED_RANK:
                if turnover < 0.30:
                    continue

            adjusted_constituents.remove(weak_entrant["Company"])
            adjusted_constituents.add(strong_exit["Company"])

            overlap = prev_constituents & adjusted_constituents
            turnover = 1 - len(overlap) / TARGET_N

        portfolio = universe[universe["Company"].isin(adjusted_constituents)].copy()
        print(f"{rebalance_date} turnover (post-control): {turnover:.2%}")

    portfolio = apply_weight_constraints(portfolio)

    if portfolio is None or portfolio.empty:
        continue

    audit_df = universe.copy()
    selected_companies = set(portfolio["Company"])
    audit_df["Selected"] = audit_df["Company"].isin(selected_companies)
    audit_df["RebalanceDate"] = rebalance_date
    rebalance_audit_frames.append(audit_df)

    portfolio = portfolio.drop_duplicates(subset=["Company"])
    portfolio["Weight"] /= portfolio["Weight"].sum()
    portfolio["Date"] = rebalance_date

    if portfolio["Company"].duplicated().any():
        raise ValueError(f"Duplicate companies detected on {rebalance_date}")

    if not np.isclose(portfolio["Weight"].sum(), 1.0):
        raise ValueError(f"Weights do not sum to 1.0 on {rebalance_date}")

    print(f"{rebalance_date} -> {len(portfolio)} companies | weight sum = {portfolio['Weight'].sum():.4f}")

    results_all.append(portfolio)
    prev_portfolio = portfolio[["Company"]].copy()

    am200 = universe[(universe["Rank"] > 100) & (universe["Rank"] <= 200)].copy()
    if not am200.empty:
        am200 = apply_weight_constraints(am200)

        if am200 is not None and not am200.empty:
            am200_snapshot = am200.copy()
            am200_snapshot = am200_snapshot.drop_duplicates(subset=["Company"])
            am200_snapshot["Weight"] /= am200_snapshot["Weight"].sum()
            am200_snapshot["Date"] = rebalance_date
            am200_history.append(am200_snapshot)

if results_all:

    final_df = pd.concat(results_all, ignore_index=True)
    final_df = final_df.drop_duplicates(subset=["Date", "Company"])
    final_output = final_df.copy()

    check = final_output.groupby("Date")["Weight"].sum()

    if not np.allclose(check.values, 1.0):
        raise ValueError("Final weights do not sum to 1.0")

    if latest_universe is not None:
        if "Liquidity Score" in final_output.columns and "Liquidity" not in final_output.columns:
            final_output = final_output.rename(columns={"Liquidity Score": "Liquidity"})
        preferred_cols = [
            "Date",
            "Company",
            "Country",
            "Weight",
            "Liquidity",
            "AvgDailyValue30d",
            "TradeCount30",
            "Rank",
        ]
        remaining_cols = [col for col in final_output.columns if col not in preferred_cols]
        final_output = final_output[preferred_cols + remaining_cols]
        latest_universe.sort_values("Rank").to_csv(LIQUIDITY_RANKING_FILE, index=False)
        latest_universe[
            ["Company", "Country", "Liquidity Score", "AvgDailyValue30d", "TradeCount30"]
        ].rename(
            columns={"Liquidity Score": "Liquidity"}
        ).sort_values("Liquidity", ascending=False).to_csv(
            LIQUIDITY_REPORT_FILE,
            index=False,
        )
        latest_universe[
            ["Rank", "Company", "Country", "Liquidity Score", "AvgDailyValue30d"]
        ].rename(
            columns={"Liquidity Score": "Liquidity"}
        ).sort_values("Rank").head(100).to_csv(
            RANKING_REPORT_FILE,
            index=False,
        )
        latest_universe.to_csv(UNIVERSE_RANKED_FILE, index=False)
    final_output.to_excel(OUTPUT_FILE, index=False)
    if am200_history:
        am200_output = pd.concat(am200_history, ignore_index=True)
        am200_output = am200_output.drop_duplicates(subset=["Date", "Company"])
        am200_output.to_excel(AM200_OUTPUT_FILE, index=False)
    country_exposure = (
        final_output.groupby(["Date", "Country"])["Weight"].sum().reset_index()
    )
    country_exposure.to_csv(COUNTRY_EXPOSURE_FILE, index=False)
    if rebalance_audit_frames:
        rebalance_audit = pd.concat(rebalance_audit_frames, ignore_index=True)
        for period, period_df in rebalance_audit.groupby(
            rebalance_audit["RebalanceDate"].dt.strftime("%Y%m")
        ):
            period_df.sort_values(["RebalanceDate", "Rank"]).to_csv(
                REBALANCE_AUDIT_PATTERN.format(yyyymm=period),
                index=False,
            )
    latest_date = final_output["Date"].max()
    latest_portfolio = final_output[final_output["Date"] == latest_date].copy()
    country_weights = (
        latest_portfolio.groupby("Country")["Weight"].sum().sort_values(ascending=False)
        if "Country" in latest_portfolio.columns
        else pd.Series(dtype=float)
    )
    with open(REPORT_FILE, "w", encoding="utf-8") as report:
        report.write("AM100 Report\n")
        report.write(f"Rebalance dates: {final_output['Date'].nunique()}\n")
        report.write(f"Latest rebalance date: {latest_date:%Y-%m-%d}\n")
        report.write(f"Latest constituents: {len(latest_portfolio)}\n")
        report.write(f"Latest weight sum: {latest_portfolio['Weight'].sum():.6f}\n")
        report.write("\nTop 10 holdings:\n")
        for _, row in latest_portfolio.sort_values("Weight", ascending=False).head(10).iterrows():
            report.write(f"{row['Company']} | {row['Weight']:.6f}\n")
        report.write("\nCountry weights:\n")
        for country, weight in country_weights.items():
            report.write(f"{country} | {weight:.6f}\n")

    print("AM100 historical index created")

else:
    print("No data generated")
