import numpy as np
from scripts.metrics import calculate_drawdown, calculate_sharpe


def compute_metrics(index_series):
    returns = index_series.pct_change().dropna()
    print(returns.describe())
    annual_return = returns.mean() * 252
    vol = returns.std() * np.sqrt(252)
    print(annual_return)
    print(vol)

    start = index_series.iloc[0]
    end = index_series.iloc[-1]
    years = (index_series.index[-1] - index_series.index[0]).days / 365
    cagr = (end / start) ** (1 / years) - 1
    sharpe = calculate_sharpe(index_series)
    if sharpe > 10:
        print("WARNING: Sharpe unusually high — check inputs")

    drawdown = calculate_drawdown(index_series)
    max_dd = drawdown.min()

    return {
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": sharpe,
        "Max Drawdown": max_dd,
    }
