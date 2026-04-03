import numpy as np
import pandas as pd

try:
    from data_loader import load_index
    from metrics import calculate_cagr, calculate_drawdown, calculate_sharpe
except ModuleNotFoundError:
    from scripts.data_loader import load_index
    from scripts.metrics import calculate_cagr, calculate_drawdown, calculate_sharpe

try:
    from scipy.optimize import minimize
except ModuleNotFoundError:
    minimize = None


BASE_MODEL_WEIGHTS = {
    "Conservative": {"AM100": 0.70, "AM200": 0.10, "AM300": 0.20},
    "Balanced": {"AM100": 0.50, "AM200": 0.30, "AM300": 0.20},
    "Growth": {"AM100": 0.30, "AM200": 0.50, "AM300": 0.20},
}

MODEL_WEIGHTS = BASE_MODEL_WEIGHTS.copy()

MODEL_METADATA = {
    "Conservative": {
        "Objective": "Stability + liquidity",
        "Characteristics": "Lower volatility, high liquidity exposure, conservative mandate",
    },
    "Balanced": {
        "Objective": "Stable + growth",
        "Characteristics": "Moderate risk, strong diversification, institutional default",
    },
    "Growth": {
        "Objective": "Maximise return",
        "Characteristics": "Higher volatility, higher expected return, frontier tilt",
    },
    "Aggressive": {
        "Objective": "Maximum return / aggressive allocation",
        "Characteristics": "Optimizer-led high-risk sleeve with strong frontier tilt",
    },
}

INDEX_NAMES = ["AM100", "AM200", "AM300"]
RISK_FREE_RATE = 0.02


def build_model_weight_system(high_risk_weights=None):
    weights_map = {name: weights.copy() for name, weights in BASE_MODEL_WEIGHTS.items()}
    if high_risk_weights is not None:
        weights_map["Aggressive"] = high_risk_weights.copy()
    return weights_map


def annual_volatility(returns):
    return returns.std() * np.sqrt(252)


def build_portfolio(returns, weights):
    ordered_weights = np.array([weights["AM100"], weights["AM200"], weights["AM300"]])
    ordered_returns = returns[["AM100", "AM200", "AM300"]]
    return ordered_returns.dot(ordered_weights)


def build_index(returns, base_level=1000.0):
    return (1 + returns).cumprod() * base_level


def metrics(portfolio_index):
    portfolio_index = portfolio_index.dropna()
    returns = portfolio_index.pct_change().dropna()
    return (
        calculate_cagr(portfolio_index),
        annual_volatility(returns),
        calculate_sharpe(portfolio_index),
        calculate_drawdown(portfolio_index).min(),
    )


def prepare_returns_frame(index_levels=None):
    levels = (
        load_building_block_panel()
        if index_levels is None
        else index_levels.copy().sort_index().dropna()
    )
    return levels.pct_change().dropna()


def optimizer_inputs(returns):
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    return mean_returns, cov_matrix


def portfolio_performance(weights, mean_returns, cov_matrix):
    ret = float(np.dot(weights, mean_returns))
    vol = float(np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))))
    return ret, vol


def neg_sharpe(weights, mean_returns, cov_matrix, risk_free_rate=RISK_FREE_RATE):
    ret, vol = portfolio_performance(weights, mean_returns, cov_matrix)
    if vol == 0:
        return np.inf
    return -((ret - risk_free_rate) / vol)


def negative_return(weights, mean_returns, cov_matrix=None):
    del cov_matrix
    return -float(np.dot(weights, mean_returns))


def portfolio_volatility_for_target_return(weights, mean_returns, cov_matrix, target_return):
    del target_return
    _, vol = portfolio_performance(weights, mean_returns, cov_matrix)
    return vol


def optimize_weights(
    mean_returns,
    cov_matrix,
    objective,
    risk_free_rate=RISK_FREE_RATE,
    target_return=None,
):
    n_assets = len(mean_returns)
    init = np.array([1 / n_assets] * n_assets)
    bounds = tuple((0, 1) for _ in range(n_assets))
    constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]

    if target_return is not None:
        constraints.append(
            {"type": "eq", "fun": lambda x: np.dot(x, mean_returns) - target_return}
        )

    if minimize is None:
        return brute_force_optimize(
            mean_returns,
            cov_matrix,
            objective,
            risk_free_rate=risk_free_rate,
            target_return=target_return,
        )

    if objective == "max_sharpe":
        opt = minimize(
            neg_sharpe,
            init,
            args=(mean_returns, cov_matrix, risk_free_rate),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
    elif objective == "max_return":
        opt = minimize(
            negative_return,
            init,
            args=(mean_returns, cov_matrix),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
    elif objective == "min_vol":
        opt = minimize(
            portfolio_volatility_for_target_return,
            init,
            args=(mean_returns, cov_matrix, target_return),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
    else:
        raise ValueError(f"Unknown optimizer objective: {objective}")

    if not opt.success:
        raise RuntimeError(f"Optimization failed: {opt.message}")

    return opt.x


def brute_force_optimize(
    mean_returns,
    cov_matrix,
    objective,
    risk_free_rate=RISK_FREE_RATE,
    target_return=None,
    step=0.01,
):
    grid = np.arange(0, 1 + step, step)
    best_weights = None
    best_score = None
    tolerance = step * 1.5

    for w1 in grid:
        for w2 in grid:
            w3 = 1 - w1 - w2
            if w3 < -1e-9 or w3 > 1 + 1e-9:
                continue
            weights = np.array([w1, w2, max(0.0, w3)])
            if not np.isclose(weights.sum(), 1.0, atol=1e-6):
                continue

            ret, vol = portfolio_performance(weights, mean_returns, cov_matrix)

            if objective == "max_sharpe":
                score = -neg_sharpe(weights, mean_returns, cov_matrix, risk_free_rate)
                if best_score is None or score > best_score:
                    best_score = score
                    best_weights = weights
            elif objective == "max_return":
                score = ret
                if best_score is None or score > best_score:
                    best_score = score
                    best_weights = weights
            elif objective == "min_vol":
                if target_return is None:
                    raise ValueError("target_return is required for min_vol optimization")
                if abs(ret - target_return) > tolerance:
                    continue
                score = vol
                if best_score is None or score < best_score:
                    best_score = score
                    best_weights = weights
            else:
                raise ValueError(f"Unknown optimizer objective: {objective}")

    if best_weights is None:
        raise RuntimeError(f"Brute force optimization failed for objective: {objective}")

    return best_weights


def weights_to_dict(weights):
    return {name: float(weight) for name, weight in zip(INDEX_NAMES, weights)}


def optimize_max_sharpe_portfolio(returns, risk_free_rate=RISK_FREE_RATE):
    mean_returns, cov_matrix = optimizer_inputs(returns)
    optimal_weights = optimize_weights(
        mean_returns, cov_matrix, "max_sharpe", risk_free_rate=risk_free_rate
    )
    portfolio_returns = returns.dot(optimal_weights)
    portfolio_index = build_index(portfolio_returns)
    expected_return, expected_vol = portfolio_performance(
        optimal_weights, mean_returns, cov_matrix
    )
    sharpe = (
        (expected_return - risk_free_rate) / expected_vol if expected_vol != 0 else np.nan
    )
    return {
        "Strategy": "Optimized Max Sharpe",
        "Weights": weights_to_dict(optimal_weights),
        "Expected Return": expected_return,
        "Expected Volatility": expected_vol,
        "Expected Sharpe": sharpe,
        "Index": portfolio_index,
        "Returns": portfolio_returns,
    }


def optimize_high_risk_portfolio(returns):
    mean_returns, cov_matrix = optimizer_inputs(returns)
    optimal_weights = optimize_weights(mean_returns, cov_matrix, "max_return")
    portfolio_returns = returns.dot(optimal_weights)
    portfolio_index = build_index(portfolio_returns)
    expected_return, expected_vol = portfolio_performance(
        optimal_weights, mean_returns, cov_matrix
    )
    return {
        "Strategy": "High Risk",
        "Weights": weights_to_dict(optimal_weights),
        "Expected Return": expected_return,
        "Expected Volatility": expected_vol,
        "Expected Sharpe": (
            (expected_return - RISK_FREE_RATE) / expected_vol if expected_vol != 0 else np.nan
        ),
        "Index": portfolio_index,
        "Returns": portfolio_returns,
    }


def build_efficient_frontier(returns, points=25):
    mean_returns, cov_matrix = optimizer_inputs(returns)
    min_target = float(mean_returns.min())
    max_target = float(mean_returns.max())
    frontier_rows = []

    for target in np.linspace(min_target, max_target, points):
        weights = optimize_weights(
            mean_returns,
            cov_matrix,
            "min_vol",
            target_return=target,
        )
        expected_return, expected_vol = portfolio_performance(
            weights, mean_returns, cov_matrix
        )
        row = {
            "Target Return": target,
            "Expected Return": expected_return,
            "Expected Volatility": expected_vol,
            "Sharpe": (
                (expected_return - RISK_FREE_RATE) / expected_vol
                if expected_vol != 0
                else np.nan
            ),
        }
        row.update({f"Weight {name}": weight for name, weight in zip(INDEX_NAMES, weights)})
        frontier_rows.append(row)

    return pd.DataFrame(frontier_rows)


def simulate_random_frontier(returns, simulations=5000, seed=42):
    mean_returns, cov_matrix = optimizer_inputs(returns)
    rng = np.random.default_rng(seed)
    rows = []

    for _ in range(simulations):
        weights = rng.random(len(INDEX_NAMES))
        weights /= weights.sum()
        expected_return, expected_volatility = portfolio_performance(
            weights, mean_returns, cov_matrix
        )
        rows.append(
            {
                "Vol": expected_volatility,
                "Return": expected_return,
                "Weight AM100": weights[0],
                "Weight AM200": weights[1],
                "Weight AM300": weights[2],
            }
        )

    return pd.DataFrame(rows)


def build_model_portfolio_system(model_levels=None, model_weights=None):
    if model_weights is None:
        returns = prepare_returns_frame()
        high_risk_result = optimize_high_risk_portfolio(returns)
        weights_map = build_model_weight_system(high_risk_result["Weights"])
    else:
        weights_map = model_weights

    if model_levels is None:
        model_levels, _ = build_model_series(model_weights=weights_map)

    rows = []
    for model_name, weights in weights_map.items():
        cagr, volatility, sharpe, drawdown = metrics(model_levels[model_name])
        rows.append(
            {
                "Portfolio": model_name,
                "Category": "Strategic Model",
                "AM100": weights["AM100"],
                "AM200": weights["AM200"],
                "AM300": weights["AM300"],
                "CAGR": cagr,
                "Volatility": volatility,
                "Sharpe": sharpe,
                "Drawdown": drawdown,
            }
        )
    return pd.DataFrame(rows)


def load_building_block_panel(index_names=None):
    names = index_names or INDEX_NAMES
    panel = {}
    for name in names:
        df = load_index(name).copy()
        df["Date"] = pd.to_datetime(df["Date"])
        panel[name] = df.set_index("Date")["Index Level"]
    return pd.DataFrame(panel).sort_index().dropna()


def build_model_series(index_levels=None, model_weights=None, base_level=1000.0):
    levels = (
        load_building_block_panel()
        if index_levels is None
        else index_levels.copy().sort_index().dropna()
    )
    weights_map = model_weights or MODEL_WEIGHTS

    returns = levels.pct_change().dropna()
    model_levels = {}
    model_returns = {}

    for model_name, weights in weights_map.items():
        weighted_returns = build_portfolio(returns, weights)
        model_returns[model_name] = weighted_returns
        model_levels[model_name] = build_index(weighted_returns, base_level)

    return pd.DataFrame(model_levels), pd.DataFrame(model_returns)


def build_capacity_map():
    capacity_map = {}
    for name in INDEX_NAMES:
        history = pd.read_excel(f"output/{name}_history.xlsx")
        history["Date"] = pd.to_datetime(history["Date"])
        latest = history["Date"].max()
        latest_snapshot = history[history["Date"] == latest]
        if "InvestableCapacity20USD" in latest_snapshot.columns:
            capacity_map[name] = float(
                latest_snapshot["InvestableCapacity20USD"]
                .fillna(latest_snapshot.get("AvgDailyValue30dUSD", pd.Series(index=latest_snapshot.index, dtype=float)) * 0.20)
                .fillna(latest_snapshot["AvgDailyValue30d"] * 0.20)
                .sum()
            )
        elif "AvgDailyValue30dUSD" in latest_snapshot.columns:
            capacity_map[name] = float(
                latest_snapshot["AvgDailyValue30dUSD"].fillna(latest_snapshot["AvgDailyValue30d"]).sum() * 0.20
            )
        else:
            capacity_map[name] = float(latest_snapshot["AvgDailyValue30d"].sum())
    return capacity_map


def build_model_metrics_table(
    model_levels=None,
    model_returns=None,
    capacity_map=None,
    model_weights=None,
):
    if model_levels is None or model_returns is None:
        model_levels, model_returns = build_model_series(model_weights=model_weights)

    capacities = capacity_map or build_capacity_map()
    weights_map = model_weights or MODEL_WEIGHTS
    rows = []

    for model_name in model_levels.columns:
        level_series = model_levels[model_name].dropna()
        cagr, volatility, sharpe, max_drawdown = metrics(level_series)
        estimated_capacity = sum(
            capacities[index_name] * weight
            for index_name, weight in weights_map[model_name].items()
        )
        rows.append(
            {
                "Model": model_name,
                "Objective": MODEL_METADATA[model_name]["Objective"],
                "Characteristics": MODEL_METADATA[model_name]["Characteristics"],
                "CAGR": cagr,
                "Volatility": volatility,
                "Sharpe": sharpe,
                "Max Drawdown": max_drawdown,
                "Latest Level": float(level_series.iloc[-1]),
                "Estimated Capacity": estimated_capacity,
            }
        )

    return pd.DataFrame(rows)


def build_model_weight_table(model_weights=None):
    weights_map = model_weights or MODEL_WEIGHTS
    rows = []
    for model_name, weights in weights_map.items():
        row = {"Model": model_name}
        row.update(weights)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    returns = prepare_returns_frame()
    optimizer_result = optimize_max_sharpe_portfolio(returns)
    high_risk_result = optimize_high_risk_portfolio(returns)
    global MODEL_WEIGHTS
    MODEL_WEIGHTS = build_model_weight_system(high_risk_result["Weights"])
    model_levels, model_returns = build_model_series(model_weights=MODEL_WEIGHTS)
    metrics = build_model_metrics_table(
        model_levels,
        model_returns,
        model_weights=MODEL_WEIGHTS,
    )
    weights = build_model_weight_table(MODEL_WEIGHTS)
    frontier = build_efficient_frontier(returns)
    random_frontier = simulate_random_frontier(returns)
    mps = build_model_portfolio_system(model_levels, MODEL_WEIGHTS)

    model_levels.reset_index().to_csv("output/model_portfolio_index_levels.csv", index=False)
    model_returns.reset_index().to_csv("output/model_portfolio_returns.csv", index=False)
    metrics.to_csv("output/model_portfolio_metrics.csv", index=False)
    weights.to_csv("output/model_portfolio_weights.csv", index=False)
    frontier.to_csv("output/model_portfolio_efficient_frontier.csv", index=False)
    random_frontier.to_csv("output/model_portfolio_random_frontier.csv", index=False)
    mps.to_csv("output/model_portfolio_system.csv", index=False)
    pd.DataFrame(
        [
            {
                "Strategy": optimizer_result["Strategy"],
                **optimizer_result["Weights"],
                "Expected Return": optimizer_result["Expected Return"],
                "Expected Volatility": optimizer_result["Expected Volatility"],
                "Expected Sharpe": optimizer_result["Expected Sharpe"],
            },
            {
                "Strategy": high_risk_result["Strategy"],
                **high_risk_result["Weights"],
                "Expected Return": high_risk_result["Expected Return"],
                "Expected Volatility": high_risk_result["Expected Volatility"],
                "Expected Sharpe": high_risk_result["Expected Sharpe"],
            },
        ]
    ).to_csv("output/model_portfolio_optimizer.csv", index=False)
    optimizer_result["Index"].rename("Optimized Max Sharpe").reset_index().to_csv(
        "output/model_portfolio_optimized_index.csv", index=False
    )
    high_risk_result["Index"].rename("High Risk").reset_index().to_csv(
        "output/model_portfolio_high_risk_index.csv", index=False
    )

    print("Model portfolios created:")
    print(metrics[["Model", "CAGR", "Volatility", "Sharpe", "Max Drawdown"]].to_string(index=False))
    print("\nOptimized portfolios:")
    print(
        pd.DataFrame(
            [
                {
                    "Strategy": optimizer_result["Strategy"],
                    **optimizer_result["Weights"],
                    "Expected Return": optimizer_result["Expected Return"],
                    "Expected Volatility": optimizer_result["Expected Volatility"],
                    "Expected Sharpe": optimizer_result["Expected Sharpe"],
                },
                {
                    "Strategy": high_risk_result["Strategy"],
                    **high_risk_result["Weights"],
                    "Expected Return": high_risk_result["Expected Return"],
                    "Expected Volatility": high_risk_result["Expected Volatility"],
                    "Expected Sharpe": high_risk_result["Expected Sharpe"],
                },
            ]
        ).to_string(index=False)
    )


if __name__ == "__main__":
    main()
