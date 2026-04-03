import pandas as pd

from data_loader import INDEX_POOL


VOL_CEILING = 0.30
DD_CEILING = 0.60
CAPACITY_TARGET = 50_000_000
CONCENTRATION_CEILING = 0.05


def risk_label(score):
    if score < 0.3:
        return "Low"
    elif score < 0.6:
        return "Moderate"
    else:
        return "High"


def load_latest_snapshot(path):
    df = pd.read_excel(path)
    df["Date"] = pd.to_datetime(df["Date"])
    latest = df["Date"].max()
    return df[df["Date"] == latest].copy()


def calculate_components(metrics_row, snapshot):
    volatility = float(metrics_row["Volatility"])
    max_drawdown = float(metrics_row["Max Drawdown"])
    if "InvestableCapacity20USD" in snapshot.columns:
        capacity_series = snapshot["InvestableCapacity20USD"].fillna(
            snapshot.get("AvgDailyValue30dUSD", pd.Series(index=snapshot.index, dtype=float)) * 0.20
        ).fillna(snapshot["AvgDailyValue30d"] * 0.20)
        capacity = float(capacity_series.sum())
    elif "AvgDailyValue30dUSD" in snapshot.columns:
        capacity = float(snapshot["AvgDailyValue30dUSD"].fillna(snapshot["AvgDailyValue30d"]).sum() * 0.20)
    else:
        capacity = float(snapshot["AvgDailyValue30d"].sum())
    concentration = float((snapshot["Weight"] ** 2).sum())

    vol_score = min(volatility / VOL_CEILING, 1)
    dd_score = min(abs(max_drawdown) / DD_CEILING, 1)
    liq_score = 1 - min(capacity / CAPACITY_TARGET, 1)
    concentration_score = min(concentration / CONCENTRATION_CEILING, 1)

    risk_score = (
        0.4 * vol_score
        + 0.3 * dd_score
        + 0.2 * liq_score
        + 0.1 * concentration_score
    )

    return {
        "Volatility": volatility,
        "Max Drawdown": max_drawdown,
        "Capacity": capacity,
        "Concentration": concentration,
        "Vol Score": vol_score,
        "Drawdown Score": dd_score,
        "Liquidity Score": liq_score,
        "Concentration Score": concentration_score,
        "Risk Score": risk_score,
        "Rating": risk_label(risk_score),
    }


def build_risk_table():
    rows = []
    for name in INDEX_POOL:
        metrics = pd.read_csv(f"output/{name}_metrics.csv").iloc[0]
        snapshot = load_latest_snapshot(f"output/{name}_history.xlsx")
        row = {"Index": name}
        row.update(calculate_components(metrics, snapshot))
        rows.append(row)

    return pd.DataFrame(rows)


def main():
    risk_df = build_risk_table()
    risk_df.to_csv("output/index_risk_ratings.csv", index=False)

    display_cols = [
        "Index",
        "Risk Score",
        "Rating",
        "Volatility",
        "Max Drawdown",
        "Capacity",
        "Concentration",
    ]
    print(risk_df[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
