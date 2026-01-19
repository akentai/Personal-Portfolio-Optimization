# Import necessary libraries and modules
import sys
import os
import datetime
import numpy as np
import pandas as pd

# Add parent directory to sys.path so you can import from data, strategies, evaluation, and backtesting modules
sys.path.append(os.path.abspath(".."))

from strategies import (
    MaxSharpeStrategy,
    MaxSortinoStrategy, 
    MomentumStrategy,
    RiskParityStrategy,
    EqualWeightStrategy,
    MeanVarianceOptimizationStrategy,
    #CvxPortfolioStrategy,
    BlackLittermanMVO,
    ValueAveragingStrategy,
    MinVarianceStrategy,
    CVaRStrategy,
    ValueOpportunityStrategy,
    DualMomentumStrategy, 
    TrendFollowingStrategy,
    MPCStrategy
)

from data import DataLoader

#####################################################################################
##### User Inputs ###################################################################


# 1a) Define your universe and fetch prices 
# Stocks
tickers = [
    "MSFT",     # Microsoft
    "AMZN",     # Amazon
    "META",     # Meta
    "GOOG",     # Alphabet 

    "NVDA",     # Nvidia
    #"AAPL",     # Apple
    "AMD",      # AMD
    "MU",       # Micron
    #"SNDK",     # Sandisk
    "TSM",      # TSMC ADR
    "NBIS",

    "GEV",      # GE Vernova
    "JPM",      # JPMorgan
]

# ETFs
# tickers = [
#     "SPY", 
#     "QQQ",
#     "SMH"
# ]

# 1b) Load historical prices for the tickers
# You can adjust the start date and interval as needed
# Now it's 3 years of monthly data
start_date = (datetime.date.today() - datetime.timedelta(days=365 * 3)).strftime("%Y-%m-%d")
loader  = DataLoader(tickers, start=start_date, interval='1mo')
prices = loader.fetch_prices()

# 2) Set up current portfolio and new capital
n = len(tickers)
current_portfolio = [1000] * n
current_portfolio = np.array(current_portfolio, dtype=float)

# 3) How much new cash to add each rebalance period?
monthly_cash = 1_000

######################################################################################

strategies = [
    #('RiskParity', RiskParityStrategy(tickers, lookback=2)),
    ('Momentum', MomentumStrategy(tickers, lookback=9, vol_threshold=0.3)),
    #('MVO', MeanVarianceOptimizationStrategy(tickers, risk_aversion=0.2, lookback=9)),
    ('MaxSharpe', MaxSharpeStrategy(tickers, lookback=3)),
    #('MaxSortino', MaxSortinoStrategy(tickers, lookback=3)),
    ('ValueOpp', ValueOpportunityStrategy(tickers, lookback_long=9, lookback_short=3, top_k=0.3)),
    #('Trend', TrendFollowingStrategy(tickers, short_window=3, long_window=12)),
    ('Dual', DualMomentumStrategy(tickers, lookback=3, top_fraction=0.6, weighting='equal')),
]

results = {}
for name, strat in strategies:
    df = strat.optimize(
        current_portfolio = current_portfolio, 
        new_capital       = monthly_cash, 
        price_history     = prices, 
        returns_history   = prices.ffill().pct_change(fill_method=None).dropna()
    )
    results[name] = df
    print(f"\n{name}:")
    print(df.round(2).to_string())
    print("")

# Aggregate: average the New Allocation across strategies
allocations = [results[name]['New Allocation'] for name in results]
avg_allocation = sum(allocations) / len(allocations)

# Impose minimum investment of 100
min_invest = 100

# Create a summary DataFrame
total_value = sum(results[list(results.keys())[0]]['Current Portfolio']) + sum(avg_allocation)
ensemble_df = pd.DataFrame({
    'Current Portfolio': results[list(results.keys())[0]]['Current Portfolio'],  # same for all
    'New Allocation': avg_allocation,
    'New Portfolio': results[list(results.keys())[0]]['Current Portfolio'],  # approximate
    'New Weights': (results[list(results.keys())[0]]['Current Portfolio']) / total_value if total_value > 0 else 0
}, index=tickers)

print("\nEnsemble (Average Allocation):")
print(ensemble_df.round(2).to_string())
print("")
