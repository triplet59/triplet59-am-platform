import numpy as np


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
    sharpe = annual_return / vol if vol != 0 else 0
    if sharpe > 10:
        print("WARNING: Sharpe unusually high — check inputs")

    drawdown = (index_series / index_series.cummax()) - 1
    max_dd = drawdown.min()

    return {
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": sharpe,
        "Max Drawdown": max_dd,
    }
