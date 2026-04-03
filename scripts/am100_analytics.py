import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import contextlib
import io
from data_loader import load_index
from metrics import calculate_cagr
from am100_performance import (
    AM100_HISTORY_FILE,
    build_total_return_index,
    load_history,
    load_prices,
)

def calculate_sharpe(returns, risk_free_rate=0.02):
    mean_return = returns.mean() * 252
    volatility = returns.std() * np.sqrt(252)

    if volatility == 0:
        return np.nan

    return (mean_return - risk_free_rate) / volatility


df = load_index("AM100")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")
index_series = df.set_index("Date")["Index Level"]

# ================================
# METRICS
# ================================
total_return = index_series.iloc[-1] / index_series.iloc[0] - 1

cagr = calculate_cagr(index_series)

returns = index_series.pct_change().dropna()

price_only_df = pd.read_excel("output/AM100_index.xlsx")
price_only_df["Date"] = pd.to_datetime(price_only_df["Date"])
price_only_df = price_only_df.sort_values("Date")
price_only_index_series = price_only_df.set_index("Date")["Index Level"]
price_only_returns = price_only_index_series.pct_change().dropna()
price_vol = price_only_returns.std() * np.sqrt(252)
tr_vol = returns.std() * np.sqrt(252)

prices = load_prices()
history = load_history(AM100_HISTORY_FILE)
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    rebuilt_price_only_index, _ = build_total_return_index(prices, history, {})

rebuilt_price_only_index["Date"] = pd.to_datetime(rebuilt_price_only_index["Date"])
rebuilt_price_only_index = rebuilt_price_only_index.sort_values("Date")
price_only_index_series = rebuilt_price_only_index.set_index("Date")["Index Level"]
aligned = pd.concat(
    [
        price_only_index_series.rename("PriceOnly"),
        index_series.rename("TotalReturn"),
    ],
    axis=1,
    join="inner",
).dropna()
price_only_returns = aligned["PriceOnly"].pct_change().dropna()
returns = aligned["TotalReturn"].pct_change().dropna()
mean_return = returns.mean() * 252
volatility = returns.std() * np.sqrt(252)
sharpe = calculate_sharpe(returns)
vol = volatility
price_vol = price_only_returns.std() * np.sqrt(252)
tr_vol = returns.std() * np.sqrt(252)
dividend_effect = price_vol - tr_vol

am100_hist = pd.read_excel("output/AM100_history.xlsx")
am100_hist["Date"] = pd.to_datetime(am100_hist["Date"])
latest_weights = am100_hist[am100_hist["Date"] == am100_hist["Date"].max()][
    ["Company", "Weight"]
].copy()
weights_sq = (latest_weights["Weight"] ** 2).sum()

master = pd.read_csv("output/master.csv")
master["Date"] = pd.to_datetime(master["Date"], dayfirst=True, errors="coerce")
master = master.sort_values("Date")
price_cols = [f"{company} Price" for company in latest_weights["Company"]]
price_cols = [col for col in price_cols if col in master.columns]
price_panel = master[["Date"] + price_cols].set_index("Date").tail(252)
returns_df = price_panel.pct_change().dropna(how="all")
corr_matrix = returns_df.corr()
mask = ~np.eye(len(corr_matrix), dtype=bool)
avg_corr = corr_matrix.where(mask).stack().mean()
diversification_effect = (1 - avg_corr) * price_vol

volatility_driver_df = pd.DataFrame(
    {
        "Component": [
            "Price volatility",
            "Dividend dampening",
            "Diversification effect (proxy)",
            "Concentration proxy (HHI)",
            "Average correlation",
            "Total return volatility",
        ],
        "Contribution": [
            price_vol,
            -dividend_effect,
            -diversification_effect,
            weights_sq,
            avg_corr,
            tr_vol,
        ],
    }
)
volatility_driver_df.to_csv("output/am100_volatility_drivers.csv", index=False)

print("Mean return (annualised):", mean_return)
print("Daily vol:", returns.std())
print("Volatility (annualised):", volatility)
print("Sharpe:", sharpe)
print("Price-only vol:", price_vol)
print("Total return vol:", tr_vol)
print(returns.describe())
print(volatility_driver_df.to_string(index=False))

assert sharpe < 3

# ================================
# DRAWDOWN
# ================================
rolling_peak = index_series.cummax()
drawdown = index_series / rolling_peak - 1
df["Peak"] = rolling_peak.to_numpy()
df["Drawdown"] = drawdown.to_numpy()

max_drawdown = df["Drawdown"].min()

# ================================
# ROLLING VOLATILITY
# ================================
rolling_vol = returns.rolling(30).std() * np.sqrt(252)
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(rolling_vol.index, rolling_vol.values, linewidth=1.2)
ax.set_title("AM100 Rolling Volatility (30D)", fontsize=10)
fig.tight_layout()
fig.savefig("output/am100_rolling_vol.png", dpi=150)
plt.close(fig)

# ================================
# OUTPUT
# ================================
print(
    {
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": sharpe,
        "Max Drawdown": max_drawdown,
    }
)
