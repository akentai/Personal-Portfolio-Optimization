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
    MeanReversionTrendStrategy,
    TimeSeriesMeanReversionStrategy
)

from data import DataLoader

#####################################################################################
##### User Inputs ###################################################################


# 1a) Define your universe and fetch prices 
# Stocks
tickers = [
    # Standard
    "AMZN",     # Amazon
    "GOOGL",    # Alphabet 
    "MSFT",     # Microsoft
    "TSM",      # TSM
    "NVDA",     # Nvidia
    "AMD",      # AMD
]

# De-duplicate tickers while preserving order (duplicate symbols can break alignment)
tickers = list(dict.fromkeys(tickers))

#ETFs
# tickers = [
#     "SPY", 
#     "QQQ",
#     "SMH"
# ]

# tickers = ['META', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSM']

# 1b) Load historical prices for the tickers
# You can adjust the start date and interval as needed
# Now it's 3 years of monthly data
start_date = (datetime.date.today() - datetime.timedelta(days=365 * 3)).strftime("%Y-%m-%d")
loader  = DataLoader(tickers, start=start_date, interval='1mo')
prices = loader.fetch_prices()
prices = prices.apply(pd.to_numeric, errors="coerce")
prices = prices.loc[:, ~prices.columns.duplicated()]
prices = prices.reindex(columns=tickers)

# 2) Set up current portfolio and new capital
n = len(tickers)
current_portfolio = [1000] * n
current_portfolio = np.array(current_portfolio, dtype=float)

# 3) How much new cash to add each rebalance period?
monthly_cash = 1_000

######################################################################################

strategies = [
    ('Momentum', MomentumStrategy(tickers, lookback=4, vol_threshold=0.4, diversification=False)),
    ('DualMomentum', DualMomentumStrategy(tickers, lookback=12, top_fraction=0.5, weighting="equal")),
    ('MeanRevTrend', MeanReversionTrendStrategy(
        tickers,
        mean_reversion_lookback=6,
        skip_recent=0,
        top_n=3
    )),
    ('TimeSeriesMean', TimeSeriesMeanReversionStrategy(tickers,mean_reversion_lookback=1,history_lookback=6,top_n=3))
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
allocations_df = pd.concat(allocations, axis=1)
allocations_df = allocations_df.apply(pd.to_numeric, errors="coerce")
avg_allocation = allocations_df.mean(axis=1)

# Impose minimum investment of 100
min_invest = 100

# Create a summary DataFrame
total_value = (
    results[list(results.keys())[0]]['Current Portfolio'].sum()
    + avg_allocation.sum()
)
ensemble_df = pd.DataFrame({
    'Current Portfolio': results[list(results.keys())[0]]['Current Portfolio'],  # same for all
    'New Allocation': avg_allocation,
    'New Portfolio': results[list(results.keys())[0]]['Current Portfolio'],  # approximate
    'New Weights': (results[list(results.keys())[0]]['Current Portfolio']) / total_value if total_value > 0 else 0
}, index=tickers)

print("\nEnsemble (Average Allocation):")
print(ensemble_df.round(2).to_string())
print("")
