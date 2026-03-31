import numpy as np
import pandas as pd

FILE = "output/AM100_index.xlsx"

df = pd.read_excel(FILE)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")

# ================================
# RETURNS
# ================================
df["Return"] = df["Index Level"].pct_change()

# ================================
# METRICS
# ================================
total_return = df["Index Level"].iloc[-1] / df["Index Level"].iloc[0] - 1

years = (df["Date"].iloc[-1] - df["Date"].iloc[0]).days / 365
annual_return = (1 + total_return) ** (1 / years) - 1

volatility = df["Return"].std() * np.sqrt(12)

sharpe = annual_return / volatility if volatility != 0 else np.nan

# ================================
# DRAWDOWN
# ================================
df["Peak"] = df["Index Level"].cummax()
df["Drawdown"] = (df["Index Level"] / df["Peak"]) - 1

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
