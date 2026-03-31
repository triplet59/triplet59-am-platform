import pandas as pd

MASTER_FILE = "output/master.xlsx"
OUTPUT_FILE = "output/AM100_history.xlsx"

MAX_STOCK_WEIGHT = 0.10
MAX_WEIGHT = 0.40
MAX_PER_COUNTRY = int(100 * MAX_WEIGHT)

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

# ================================
# GET MONTH-END DATES
# ================================
month_ends = df["Date"].drop_duplicates().sort_values().groupby(
    [df["Date"].dt.year, df["Date"].dt.month]
).last().values

# ================================
# MAIN LOOP
# ================================
for rebalance_date in month_ends:
    df_cut = df[df["Date"] <= rebalance_date]

    results = []

    for col in price_cols:
        company = col.replace(" Price", "")
        volume_col = col.replace("Price", "Volume")

        if volume_col not in df_cut.columns:
            continue

        series = df_cut[["Date", col]].dropna()

        # Filter 1: History
        if len(series) < 150:
            continue

        # Filter 2: Recent liquidity
        recent = series.tail(90)
        trading_days = recent[col].notna().sum()
        if trading_days < 15:
            continue

        valid_prices = recent[col].dropna()
        if len(valid_prices) < 10:
            continue

        # Latest price
        latest_price = series[col].iloc[-1]

        # Liquidity Score (30-day)
        merged = df_cut[[col, volume_col]].dropna().tail(30)
        if len(merged) < 10:
            continue

        merged["TradedValue"] = merged[col] * merged[volume_col]
        liquidity_score = merged["TradedValue"].mean()

        results.append(
            {
                "Company": company,
                "Liquidity Score": liquidity_score,
                "Price": latest_price,
                "Trading Days (90d)": trading_days,
            }
        )

    if len(results) == 0:
        continue

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values("Liquidity Score", ascending=False)

    # ================================
    # COUNTRY CAP
    # ================================
    selected = []
    country_counts = {}

    for _, row in result_df.iterrows():
        company = row["Company"]
        country = company.split("(")[-1].replace(")", "").strip()

        count = country_counts.get(country, 0)
        if count >= MAX_PER_COUNTRY:
            continue

        selected.append(row)
        country_counts[country] = count + 1

        if len(selected) == 100:
            break

    am100 = pd.DataFrame(selected)

    if len(am100) == 0:
        continue

    # ================================
    # WEIGHTING (ITERATIVE CAP)
    # ================================
    weights = am100["Liquidity Score"] / am100["Liquidity Score"].sum()

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

    am100["Weight"] = weights
    am100["Date"] = rebalance_date

    results_all.append(am100)

# ================================
# SAVE OUTPUT
# ================================
if len(results_all) > 0:
    final_df = pd.concat(results_all)
    final_df.to_excel(OUTPUT_FILE, index=False)
    print("✅ AM100 historical index created")
else:
    print("⚠️ No data generated")
