import numpy as np
import pandas as pd
from metrics import calculate_sharpe
FILE = "output/AM100_index.xlsx"

df = pd.read_excel(FILE)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")
am100_index_series = df["Index Level"]

# ================================
# METRICS
# ================================
total_return = am100_index_series.iloc[-1] / am100_index_series.iloc[0] - 1

years = (df["Date"].iloc[-1] - df["Date"].iloc[0]).days / 365
annual_return = (1 + total_return) ** (1 / years) - 1

returns = am100_index_series.pct_change().dropna()
mean_return = returns.mean() * 252
volatility = returns.std() * np.sqrt(252)
sharpe = calculate_sharpe(am100_index_series)

print("Mean return (annualised):", mean_return)
print("Volatility (annualised):", volatility)
print("Sharpe:", sharpe)

assert sharpe < 3

# ================================
# DRAWDOWN
# ================================
df["Peak"] = am100_index_series.cummax()
df["Drawdown"] = (am100_index_series / df["Peak"]) - 1

max_drawdown = df["Drawdown"].min()

# ================================
# OUTPUT
# ================================
print("\n📊 AM100 Analytics")
print(f"Total Return: {total_return:.2%}")
print(f"Annual Return: {annual_return:.2%}")
print(f"Volatility: {volatility:.2%}")
print(f"Sharpe Ratio: {sharpe:.2f}")
print(f"Max Drawdown: {max_drawdown:.2%}")
