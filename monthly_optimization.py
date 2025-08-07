# Import necessary libraries and modules
import sys
import os
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
    ValueOpportunityStrategy
)

from data import DataLoader


#####################################################################################
##### User Inputs ###################################################################


# 1a) Define your universe and fetch prices (incl. SPY for benchmark)
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']

# 1b) Load historical prices for the tickers
# You can adjust the start date and interval as needed
loader  = DataLoader(tickers, start='2010-01-01', interval='1mo')
prices = loader.fetch_prices()

# 2) Set up current portfolio and new capital
n = len(tickers)
current_portfolio = [0] * n

# 3) How much new cash to add each rebalance period?
monthly_cash = 2_000

######################################################################################

momentum = MomentumStrategy(tickers)
df = momentum.optimize(
    current_portfolio = current_portfolio, 
    new_capital       = monthly_cash, 
    price_history     = prices, 
    returns_history   = prices.pct_change().dropna()
)

print("")
print(df.round(1).to_string())
print("")
