# Grid search script for strategy hyperparameters.
# Adjust the grids below to match the strategies you want to tune.

import sys
import os
import datetime
import itertools
import numpy as np
import pandas as pd

# Add project root to sys.path so you can import from data, strategies, evaluation, and backtesting modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from data import DataLoader
from backtesting import Backtester
from evaluation import compute_strategy_metrics
from strategies import (
    MeanVarianceOptimizationStrategy,
    MomentumStrategy,
    RiskParityStrategy,
    CVaRStrategy,
    MaxSharpeStrategy,
    MaxSortinoStrategy,
    BlackLittermanMVO,
    ValueAveragingStrategy,
    ValueOpportunityStrategy,
    DualMomentumStrategy,
    TrendFollowingStrategy,
    VolatilityTargetingStrategy
)


def expand_grid(param_grid):
    if not param_grid:
        return [{}]
    keys = list(param_grid.keys())
    values = []
    for key in keys:
        val = param_grid[key]
        values.append(val if isinstance(val, (list, tuple)) else [val])
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def is_better(score, best_score, higher_is_better=True):
    if score is None or np.isnan(score):
        return False
    if best_score is None or np.isnan(best_score):
        return True
    return score > best_score if higher_is_better else score < best_score


#####################################################################################
##### User Inputs ###################################################################

# Universe and data
tickers = [
    "AMZN",     # Amazon
    "GOOGL",    # Alphabet 
    "MSFT",     # Microsoft
    "TSM",      # TSMC
    #"NVDA",     # Nvidia
    "AMD",      # AMD
    
    # Good performers
    #"NVDA",    # Nvidia (Unique case, ignore for evaluation)
    # "AAPL",     # Apple
    # "AMD",      # AMD
    # "MU",       # Micron
    # "TSM",      # TSMC ADR
    # "JPM",      # JPMorgan
    #"NBIS",     # Nebius Group (no historical data, ignore for evaluation)
    #"ANET",     # Arista Networks
    
    # Underperforming 
    # "INTC",       # Intel 
    # "BA",         # Boeing
    # "MA",         # Mastercard
    # "ADBE",       # Adobe
    # "VZ",         # Verizon
]
start_date = (datetime.date.today() - datetime.timedelta(days=365 * 10)).strftime("%Y-%m-%d")
end_date = (datetime.date.today() - datetime.timedelta(days=365 * 3)).strftime("%Y-%m-%d")
interval = "1mo"

# Portfolio settings
initial_capital = 0
monthly_cash = 2_000
higher_is_better = True  # for the score metric
rolling_window = 12  # months used for warm-up/lookback in backtesting

# Metric used to pick best parameters
score_metric = "CAGR"  # e.g., "CAGR", "Sharpe", "Sortino", "Max Drawdown"
risk_free_rate = 0.0125

# Parameter grids per strategy
strategy_configs = {
    "MVO": {
        "class": MeanVarianceOptimizationStrategy,
         "param_grid": {
            "risk_aversion": [0.1, 0.2, 0.5, 1.0],
            "lookback": [3, 6, 9, 12, -1],
        },
    },
    "Momentum": {
        "class": MomentumStrategy,
        "param_grid": {
            "lookback": [3, 4, 5, 6, 7, 8, 9, 12],
            "vol_threshold": [0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
        },
    },
    "RiskParity": {
        "class": RiskParityStrategy,
        "param_grid": {
            "lookback": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, -1],
        },
    },
    "MaxSharpe": {
        "class": MaxSharpeStrategy,
        "param_grid": {
            "lookback": [3, 4, 5, 6, 7, 8, 9, 12, -1],
        },
    },
    "MaxSortino": {
        "class": MaxSortinoStrategy,
        "param_grid": {
            "lookback": [3, 4, 5, 6, 7, 8, 9, 12, -1],
        },
    },
    "CVaR": {
        "class": CVaRStrategy,
        "param_grid": {
            "alpha": [0.9, 0.95, 0.99],
        },
    },
    "ValueAvg": {
        "class": ValueAveragingStrategy,
        "param_grid": {
            "target_growth_rate": [0.01, 0.02, 0.03],
        },
    },
    "ValueOpp": {
        "class": ValueOpportunityStrategy,
        "param_grid": {
            "lookback_long": [9, 12],
            "lookback_short": [1, 3],
            "top_k": [0.3, 0.5],
        },
    },
    "Dual": {
        "class": DualMomentumStrategy,
        "param_grid": {
            "lookback": [1, 2, 3, 4, 5, 6, 7, 8, 9, 12],
            "top_fraction": [0.2, 0.3, 0.4, 0.5, 0.6],
            "absolute_threshold": [0.0],
            "weighting": ["equal", "momentum"],
        },
    },
    "Trend": {
        "class": TrendFollowingStrategy,
        "param_grid": {
            "short_window": [1, 2, 3],
            "long_window": [10, 12],
        },
    },
    "VolTarget": {
        "class": VolatilityTargetingStrategy,
        "param_grid": {
            "target_vol": [0.08, 0.10, 0.12],
            "lookback": [6, 12],
        },
    },
}

#####################################################################################

# Load prices once
loader = DataLoader(tickers, start=start_date, end=end_date, interval=interval)
prices = loader.fetch_prices()

initial_allocation = np.array([initial_capital / len(tickers)] * len(tickers))

#####################################################################################
# Grid search per strategy (prints each strategy separately)

def run_grid_search(strategy_name):
    config = strategy_configs[strategy_name]
    strategy_class = config["class"]
    grid = expand_grid(config["param_grid"])

    rows = []
    best_score = None
    best_row = None

    for params in grid:
        strategy = strategy_class(tickers, **params)
        bt = Backtester(
            strategies={strategy_name: strategy},
            prices=prices,
            initial_allocation=initial_allocation,
            monthly_cash=monthly_cash,
            rolling_window=rolling_window,
        )
        results = bt.run()

        metrics = compute_strategy_metrics(
            results,
            monthly_cash=monthly_cash,
            benchmarks=None,
            risk_free_rate=risk_free_rate,
        )
        score = float(metrics.loc[strategy_name, score_metric])

        row = {"Strategy": strategy_name, "Score": score}
        row.update(params)
        rows.append(row)

        if is_better(score, best_score, higher_is_better=higher_is_better):
            best_score = score
            best_row = row

    results_df = pd.DataFrame(rows)
    print(f"\n{strategy_name} runs (sorted):")
    print(results_df.sort_values(["Score"], ascending=not higher_is_better).to_string(index=False))

    if best_row is not None:
        best_df = pd.DataFrame([best_row])
        print(f"\nBest {strategy_name} parameters:")
        print(best_df.to_string(index=False))


strategies_to_run = [
    "Momentum",
    "MVO",
    "RiskParity",
    "MaxSharpe",
    "MaxSortino",
    "CVaR",
    "Dual",
    "Trend",
    #"VolTarget",
    "ValueOpp",
]

for name in strategies_to_run:
    run_grid_search(name)
