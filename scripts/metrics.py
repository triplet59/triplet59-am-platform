import numpy as np


def calculate_cagr(series):
    start = series.iloc[0]
    end = series.iloc[-1]
    years = (series.index[-1] - series.index[0]).days / 365
    return (end / start) ** (1 / years) - 1


def calculate_sharpe(index_series, risk_free_rate=0.02):
    """
    Calculates annualised Sharpe ratio from index level series.

    Parameters:
        index_series (pd.Series): Index level time series
        risk_free_rate (float): Annual risk-free rate

    Returns:
        float: Sharpe ratio
    """

    returns = index_series.pct_change().dropna()
    if returns.empty:
        return np.nan
    if len(returns) < 30:
        return np.nan

    mean_return = returns.mean() * 252
    volatility = returns.std() * np.sqrt(252)

    if volatility == 0:
        return np.nan

    return (mean_return - risk_free_rate) / volatility


def calculate_drawdown(series):
    series = series.copy()
    peak = series.cummax()
    drawdown = (series / peak) - 1
    return drawdown
