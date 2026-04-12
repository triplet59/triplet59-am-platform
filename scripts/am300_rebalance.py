import os

import numpy as np
import pandas as pd

# =========================
# AM300 PARAMETERS
# =========================
INDEX_NAME = "AM300"

TARGET_CONSTITUENTS = 300
BUFFER_MULTIPLIER = 1.2

MIN_ADV_USD = 200_000
MIN_TRADING_DAYS_90D = 60
MIN_NONZERO_DAYS_90D = 50
MIN_POSITIVE_COVERAGE = 0.80
MIN_HISTORY_YEARS = 2

MAX_STOCK_WEIGHT = 0.05
MAX_COUNTRY_WEIGHT = 0.25

MASTER_FILE = "output/master.xlsx"
OUTPUT_FILE = "output/AM300_history.xlsx"
ADV_USD_FILE = "output/AM300_adv_usd.csv"
CAPACITY_USD_FILE = "output/AM300_capacity_usd.csv"
LIQUIDITY_RANKING_FILE = "output/am300_liquidity_ranking.csv"
LIQUIDITY_REPORT_FILE = "output/am300_liquidity_report.csv"
RANKING_REPORT_FILE = "output/am300_ranking.csv"
UNIVERSE_RANKED_FILE = "output/am300_universe_ranked.csv"
REBALANCE_AUDIT_PATTERN = "output/am300_rebalance_audit_{yyyymm}.csv"
DECISION_AUDIT_FILE = "output/am300_rebalance_decision_audit.csv"
REPORT_FILE = "output/am300_report.txt"
COUNTRY_EXPOSURE_FILE = "output/am300_country_exposure.csv"
DIVIDEND_RESOLUTION_FILE = "output/dividend_resolution.csv"

MAX_PER_COUNTRY = int(TARGET_CONSTITUENTS * MAX_COUNTRY_WEIGHT)
MIN_DOLLAR_LIQUIDITY = MIN_ADV_USD
TARGET_N = TARGET_CONSTITUENTS
ENTRY_BUFFER = TARGET_CONSTITUENTS
EXIT_BUFFER = int(TARGET_CONSTITUENTS * BUFFER_MULTIPLIER)
MAX_FALLBACK_RANK = 80
RELAXED_FALLBACK_RANK = 90
MAX_TURNOVER = 0.20
PROTECTED_RANK = 60
REBALANCE_MONTHS = {3, 6, 9, 12}
MIN_HOLD_MONTHS = 3
MIN_VALID_OBSERVATIONS = 252 * MIN_HISTORY_YEARS

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
ELIGIBLE_DIVIDEND_STATUSES = {"OK", "ZERO_CONFIRMED"}

def extract_country(company_name):
    return company_name.rsplit("(", 1)[-1].rstrip(")")


def resolve_price_volume_columns(df, company):
    explicit_usd = f"{company} Price USD"
    standard_price = f"{company} Price"
    volume_col = f"{company} Volume"

    if explicit_usd in df.columns and volume_col in df.columns:
        return explicit_usd, volume_col
    if standard_price in df.columns and volume_col in df.columns:
        return standard_price, volume_col
    return None, None


def load_dividend_gate():
    if not os.path.exists(DIVIDEND_RESOLUTION_FILE):
        print(f"WARNING: dividend resolution file missing: {DIVIDEND_RESOLUTION_FILE}")
        return {}

    resolution = pd.read_csv(DIVIDEND_RESOLUTION_FILE)
    resolution["Company"] = resolution["Company"].astype(str).str.strip()
    resolution["Dividend_Status"] = resolution["Dividend_Status"].astype(str).str.strip()
    return dict(zip(resolution["Company"], resolution["Dividend_Status"]))


def build_capacity_usd_export(history_df):
    working = history_df.copy()
    working["Date"] = pd.to_datetime(working["Date"])

    adv_usd = (
        working["AvgDailyValue30dUSD"].fillna(working["AvgDailyValue30d"])
        if "AvgDailyValue30dUSD" in working.columns
        else working["AvgDailyValue30d"]
    )
    investable_usd = (
        working["InvestableCapacity20USD"].fillna(adv_usd * 0.20)
        if "InvestableCapacity20USD" in working.columns
        else adv_usd * 0.20
    )

    return (
        pd.DataFrame(
            {
                "Date": working["Date"],
                "ADV_USD": adv_usd,
                "InvestableCapacityUSD": investable_usd,
            }
        )
        .groupby("Date")[["ADV_USD", "InvestableCapacityUSD"]]
        .sum(min_count=1)
        .reset_index()
    )


def write_adv_capacity_files(capacity_df, adv_path, capacity_path):
    capacity_df[["Date", "ADV_USD"]].to_csv(adv_path, index=False)
    capacity_df[["Date", "InvestableCapacityUSD"]].to_csv(capacity_path, index=False)


def assign_regime(country):
    return REGIME_MAP.get(country, "LOW")


def build_month_ends(dates):
    month_end_series = (
        dates.drop_duplicates()
        .sort_values()
        .groupby([dates.dt.year, dates.dt.month])
        .max()
    )
    return [pd.Timestamp(x) for x in month_end_series.tolist()]


def build_rebalance_schedule(dates):
    month_ends = build_month_ends(dates)
    month_end_set = {pd.Timestamp(x) for x in month_ends}
    schedule = []

    for selection_date in month_ends:
        if selection_date.month not in REBALANCE_MONTHS:
            continue

        implementation_date = pd.Timestamp(selection_date) + pd.offsets.MonthEnd(1)
        implementation_date = pd.Timestamp(implementation_date)

        if implementation_date not in month_end_set:
            continue

        schedule.append((pd.Timestamp(selection_date), implementation_date))

    return schedule


def minimum_hold_satisfied(entry_date, implementation_date):
    if entry_date is None or pd.isna(entry_date):
        return True
    return pd.Timestamp(implementation_date) >= (
        pd.Timestamp(entry_date) + pd.DateOffset(months=MIN_HOLD_MONTHS)
    )


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
    total_liquidity = portfolio["AvgDailyValue30dUSD"].sum()

    if total_liquidity == 0:
        return None

    portfolio["Weight"] = portfolio["AvgDailyValue30dUSD"] / total_liquidity
    portfolio["Country"] = portfolio["Company"].str.extract(r"\((.*?)\)")
    portfolio["Regime"] = portfolio["Country"].map(assign_regime)
    portfolio = apply_stock_cap(portfolio)
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
am300_history = []
prev_portfolio = None
latest_universe = None
rebalance_audit_frames = []
decision_audit_frames = []
constituent_entry_dates = {}
dividend_gate = load_dividend_gate()

# ================================
# QUARTER-END OBSERVATION / NEXT MONTH-END IMPLEMENTATION
# ================================
rebalance_schedule = build_rebalance_schedule(df["Date"])

# ================================
# MAIN LOOP
# ================================
for selection_date, implementation_date in rebalance_schedule:

    df_cut = df[df["Date"] <= selection_date]

    records = []
    decision_rows = []

    for col in price_cols:

        company = col.replace(" Price", "")
        company_country = extract_country(company)
        price_col, volume_col = resolve_price_volume_columns(df_cut, company)
        decision = {
            "Company": company,
            "Country": company_country,
            "ObservationDate": selection_date,
            "ImplementationDate": implementation_date,
            "PassedDividendGate": False,
            "PassedTradingDays": False,
            "PassedADV": False,
            "PassedCoverage": False,
            "PassedPositiveCoverage": False,
            "PassedNonZeroDays": False,
            "RankAtSelection": np.nan,
            "RankAtImplementation": np.nan,
            "Selected": False,
            "RejectReason": "",
        }

        if price_col is None or volume_col is None:
            decision["RejectReason"] = "MISSING_PRICE_OR_VOLUME_COLUMN"
            decision_rows.append(decision)
            continue

        dividend_status = dividend_gate.get(company)
        decision["Dividend_Status"] = dividend_status
        if dividend_status not in ELIGIBLE_DIVIDEND_STATUSES:
            decision["RejectReason"] = "FAILED_DIVIDEND_GATE"
            decision_rows.append(decision)
            continue
        decision["PassedDividendGate"] = True

        price_series = pd.to_numeric(df_cut[price_col], errors="coerce").where(lambda s: s.gt(0))
        volume_raw = pd.to_numeric(df_cut[volume_col], errors="coerce")
        volume_series = volume_raw.where(lambda s: s.gt(0))
        positive_obs = price_series.notna() & volume_series.notna()
        observed_price_days = int(price_series.notna().sum())
        positive_coverage_pct = (
            float(positive_obs.sum() / observed_price_days) if observed_price_days else np.nan
        )
        decision["PositiveCoveragePct"] = positive_coverage_pct
        decision["CoveragePct"] = positive_coverage_pct
        observed_volume = volume_raw.where(price_series.notna())
        decision["ObservedPriceDays"] = observed_price_days
        decision["MissingVolumeDays"] = int(observed_volume.isna().sum())
        decision["ZeroVolumeDays"] = int(
            observed_volume.fillna(0).eq(0).sum() - observed_volume.isna().sum()
        )
        decision["PositiveVolumeDays"] = int(observed_volume.gt(0).sum())
        decision["PassedPositiveCoverage"] = bool(
            pd.notna(positive_coverage_pct) and positive_coverage_pct >= MIN_POSITIVE_COVERAGE
        )
        decision["PassedCoverage"] = decision["PassedPositiveCoverage"]
        decision["ValidObservationCount"] = int(positive_obs.sum())

        if not decision["PassedPositiveCoverage"]:
            decision["RejectReason"] = "FAILED_POSITIVE_VOLUME_COVERAGE"
            decision_rows.append(decision)
            continue

        if positive_obs.sum() < MIN_VALID_OBSERVATIONS:
            decision["RejectReason"] = "INSUFFICIENT_VALID_HISTORY"
            decision_rows.append(decision)
            continue

        recent_window_start = pd.Timestamp(selection_date) - pd.offsets.BDay(89)
        recent_mask = (df_cut["Date"] >= recent_window_start) & (df_cut["Date"] <= selection_date)
        recent_valid_obs = positive_obs[recent_mask]
        trading_days = int(recent_valid_obs.sum())
        decision["TradingDays90d"] = trading_days

        if trading_days < MIN_TRADING_DAYS_90D:
            decision["RejectReason"] = "FAILED_TRADING_DAYS"
            decision_rows.append(decision)
            continue
        decision["PassedTradingDays"] = True

        nonzero_days = int(volume_series[recent_mask].fillna(0).gt(0).sum())
        decision["NonZeroVolumeDays90d"] = nonzero_days
        if INDEX_NAME == "AM300" and nonzero_days < MIN_NONZERO_DAYS_90D:
            decision["RejectReason"] = "FAILED_NONZERO_DAYS"
            decision_rows.append(decision)
            continue
        decision["PassedNonZeroDays"] = True

        valid_prices = price_series[recent_mask].dropna()
        if len(valid_prices) < 10:
            decision["RejectReason"] = "MISSING_DATA_POST_LAG"
            decision_rows.append(decision)
            continue

        latest_price = price_series.dropna().iloc[-1]
        traded_value_usd = (price_series * volume_series).dropna()
        adv_usd_series = traded_value_usd.rolling(30, min_periods=30).mean()
        avg_value_usd_30d = adv_usd_series.iloc[-1] if not adv_usd_series.empty else np.nan
        decision["AvgDailyValue30dUSD"] = avg_value_usd_30d

        if pd.isna(avg_value_usd_30d) or avg_value_usd_30d < MIN_DOLLAR_LIQUIDITY:
            decision["RejectReason"] = "FAILED_ADV_AT_IMPLEMENTATION"
            decision_rows.append(decision)
            continue
        decision["PassedADV"] = True

        trade_count_30 = min(len(traded_value_usd.tail(30)), 30) if not traded_value_usd.empty else np.nan

        records.append({
            "Company": company,
            "Dividend_Status": dividend_status,
            "Liquidity Score": avg_value_usd_30d,
            "Price": latest_price,
            "Trading Days (90d)": trading_days,
            "NonZeroVolumeDays90d": nonzero_days,
            "AvgDailyValue30d": avg_value_usd_30d,
            "AvgDailyValue30dUSD": avg_value_usd_30d,
            "InvestableCapacity20USD": (
                avg_value_usd_30d * 0.20 if pd.notna(avg_value_usd_30d) else np.nan
            ),
            "TradeCount30": trade_count_30,
            "StaleVolumeDays30d": np.nan,
            "ClippedValueDays30d": np.nan,
            "MedianTradedValueUSD": np.nan,
            "TradedValueCapUSD": np.nan,
        })
        decision_rows.append(decision)

    if not records:
        if decision_rows:
            decision_audit_frames.append(pd.DataFrame(decision_rows))
        continue

    universe = pd.DataFrame(records)
    universe["Date"] = implementation_date
    universe["ObservationDate"] = selection_date
    universe["ImplementationDate"] = implementation_date
    universe["Country"] = universe["Company"].str.extract(r"\((.*?)\)")
    universe["Regime"] = universe["Country"].map(REGIME_MAP)
    universe["Regime"] = universe["Regime"].fillna("LOW")
    universe["RawLiquidity"] = universe["AvgDailyValue30dUSD"]
    universe["Liquidity Score"] = universe["AvgDailyValue30dUSD"]
    universe = universe.sort_values("AvgDailyValue30dUSD", ascending=False)
    universe["Rank"] = universe["AvgDailyValue30dUSD"].rank(ascending=False, method="first")
    latest_universe = universe.copy()
    universe_rank_map = dict(zip(universe["Company"], universe["Rank"]))

    prev_constituents = set(prev_portfolio["Company"]) if prev_portfolio is not None else set()
    is_major_universe_change = (
        prev_portfolio is not None
        and abs(len(universe) - len(prev_constituents)) / max(len(prev_constituents), 1) > 0.30
    )
    held_constituents = {
        company
        for company in prev_constituents
        if not minimum_hold_satisfied(constituent_entry_dates.get(company), implementation_date)
    }

    selected = []
    selected_companies = set()

    if held_constituents:
        held_rows = (
            universe[universe["Company"].isin(held_constituents)]
            .sort_values("Rank")
            .to_dict("records")
        )
        for row in held_rows:
            selected.append(row)
            selected_companies.add(row["Company"])

    for _, row in universe.sort_values("Rank").iterrows():
        company = row["Company"]
        rank = row["Rank"]

        if company in selected_companies:
            continue

        if company in prev_constituents:
            if rank <= EXIT_BUFFER:
                selected.append(row)
                selected_companies.add(company)
        else:
            if rank <= ENTRY_BUFFER:
                selected.append(row)
                selected_companies.add(company)

        if len(selected) == TARGET_N:
            break

    if len(selected) < TARGET_N:
        selected_company_list = [row["Company"] for row in selected]
        remaining = universe[~universe["Company"].isin(selected_company_list)]
        remaining = remaining[remaining["Rank"] <= MAX_FALLBACK_RANK].sort_values("Rank")

        for _, row in remaining.iterrows():
            selected.append(row)
            if len(selected) == TARGET_N:
                break

    if len(selected) < TARGET_N:
        selected_company_list = [row["Company"] for row in selected]
        remaining = universe[~universe["Company"].isin(selected_company_list)]
        remaining = remaining[remaining["Rank"] <= RELAXED_FALLBACK_RANK].sort_values("Rank")

        for _, row in remaining.iterrows():
            selected.append(row)
            if len(selected) == TARGET_N:
                break

    portfolio = pd.DataFrame(selected)

    if portfolio.empty:
        decision_df = pd.DataFrame(decision_rows)
        if not decision_df.empty:
            decision_df["RankAtSelection"] = decision_df["Company"].map(universe_rank_map)
            decision_df["RankAtImplementation"] = decision_df["RankAtSelection"]
            decision_df.loc[
                decision_df["RejectReason"].eq("") & decision_df["RankAtSelection"].isna(),
                "RejectReason",
            ] = "NOT_IN_QUALIFIED_UNIVERSE"
            decision_audit_frames.append(decision_df)
        continue

    # =========================
    # TURNOVER CONTROL MODULE
    # =========================
    new_constituents = set(portfolio["Company"])
    overlap = prev_constituents & new_constituents
    turnover = 1 - len(overlap) / TARGET_N
    print(
        f"selection {selection_date:%Y-%m-%d} -> implementation {implementation_date:%Y-%m-%d} turnover (pre-control): {turnover:.2%}"
    )

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
        print(
            f"selection {selection_date:%Y-%m-%d} -> implementation {implementation_date:%Y-%m-%d} turnover (post-control): {turnover:.2%}"
        )

    portfolio = apply_weight_constraints(portfolio)

    if portfolio is None or portfolio.empty:
        decision_df = pd.DataFrame(decision_rows)
        if not decision_df.empty:
            decision_df["RankAtSelection"] = decision_df["Company"].map(universe_rank_map)
            decision_df["RankAtImplementation"] = decision_df["RankAtSelection"]
            decision_df.loc[
                decision_df["RejectReason"].eq("") & decision_df["RankAtSelection"].notna(),
                "RejectReason",
            ] = "WEIGHT_CONSTRAINTS_REMOVED"
            decision_audit_frames.append(decision_df)
        continue

    audit_df = universe.copy()
    selected_companies = set(portfolio["Company"])
    audit_df["Selected"] = audit_df["Company"].isin(selected_companies)
    audit_df["ObservationDate"] = selection_date
    audit_df["RebalanceDate"] = implementation_date
    rebalance_audit_frames.append(audit_df)

    decision_df = pd.DataFrame(decision_rows)
    if not decision_df.empty:
        decision_df["RankAtSelection"] = decision_df["Company"].map(universe_rank_map)
        decision_df["RankAtImplementation"] = decision_df["RankAtSelection"]
        selected_set = set(portfolio["Company"])
        decision_df["Selected"] = decision_df["Company"].isin(selected_set)
        in_prev = decision_df["Company"].isin(prev_constituents)
        decision_df.loc[
            decision_df["RejectReason"].eq("") & decision_df["Selected"],
            "RejectReason",
        ] = "SELECTED"
        decision_df.loc[
            decision_df["RejectReason"].eq("") & (~decision_df["Selected"]) & in_prev & (decision_df["RankAtSelection"] > EXIT_BUFFER),
            "RejectReason",
        ] = "BUFFER_EXIT"
        decision_df.loc[
            decision_df["RejectReason"].eq("") & (~decision_df["Selected"]) & (~in_prev) & (decision_df["RankAtSelection"] > ENTRY_BUFFER),
            "RejectReason",
        ] = "RANK_BELOW_THRESHOLD"
        decision_df.loc[
            decision_df["RejectReason"].eq("") & (~decision_df["Selected"]),
            "RejectReason",
        ] = "NOT_SELECTED_POST_BUFFER"
        decision_audit_frames.append(decision_df)

    portfolio = portfolio.drop_duplicates(subset=["Company"])
    portfolio["Weight"] /= portfolio["Weight"].sum()
    portfolio["ObservationDate"] = selection_date
    portfolio["ImplementationDate"] = implementation_date
    portfolio["Date"] = implementation_date

    if portfolio["Company"].duplicated().any():
        raise ValueError(f"Duplicate companies detected on {implementation_date}")

    if not np.isclose(portfolio["Weight"].sum(), 1.0):
        raise ValueError(f"Weights do not sum to 1.0 on {implementation_date}")

    print(
        f"selection {selection_date:%Y-%m-%d} -> implementation {implementation_date:%Y-%m-%d} | {len(portfolio)} companies | weight sum = {portfolio['Weight'].sum():.4f}"
    )

    results_all.append(portfolio)
    current_entries = {}
    for company in portfolio["Company"]:
        current_entries[company] = constituent_entry_dates.get(company, implementation_date)
    constituent_entry_dates = current_entries
    prev_portfolio = portfolio[["Company"]].copy()

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
            "ObservationDate",
            "ImplementationDate",
            "Company",
            "Country",
            "Weight",
            "Liquidity",
            "AvgDailyValue30d",
            "AvgDailyValue30dUSD",
            "InvestableCapacity20USD",
            "TradeCount30",
            "Rank",
        ]
        remaining_cols = [col for col in final_output.columns if col not in preferred_cols]
        final_output = final_output[preferred_cols + remaining_cols]
        latest_universe.sort_values("Rank").to_csv(LIQUIDITY_RANKING_FILE, index=False)
        latest_universe[
            [
                "Company",
                "Country",
                "Liquidity Score",
                "AvgDailyValue30d",
                "AvgDailyValue30dUSD",
                "InvestableCapacity20USD",
                "TradeCount30",
            ]
        ].rename(
            columns={"Liquidity Score": "Liquidity"}
        ).sort_values("Liquidity", ascending=False).to_csv(
            LIQUIDITY_REPORT_FILE,
            index=False,
        )
        latest_universe[
            [
                "Rank",
                "Company",
                "Country",
                "Liquidity Score",
                "AvgDailyValue30d",
                "AvgDailyValue30dUSD",
                "InvestableCapacity20USD",
            ]
        ].rename(
            columns={"Liquidity Score": "Liquidity"}
        ).sort_values("Rank").head(TARGET_CONSTITUENTS).to_csv(
            RANKING_REPORT_FILE,
            index=False,
        )
        latest_universe.to_csv(UNIVERSE_RANKED_FILE, index=False)
    final_output.to_excel(OUTPUT_FILE, index=False)
    capacity_usd = build_capacity_usd_export(final_output)
    write_adv_capacity_files(capacity_usd, ADV_USD_FILE, CAPACITY_USD_FILE)
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
    if decision_audit_frames:
        decision_audit = pd.concat(decision_audit_frames, ignore_index=True)
        decision_audit.to_csv(DECISION_AUDIT_FILE, index=False)
    latest_date = final_output["Date"].max()
    latest_portfolio = final_output[final_output["Date"] == latest_date].copy()
    country_weights = (
        latest_portfolio.groupby("Country")["Weight"].sum().sort_values(ascending=False)
        if "Country" in latest_portfolio.columns
        else pd.Series(dtype=float)
    )
    with open(REPORT_FILE, "w", encoding="utf-8") as report:
        report.write(f"{INDEX_NAME} Report\n")
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

    print(f"{INDEX_NAME} historical index created")

else:
    print("No data generated")
