# Import necessary libraries and modules
import sys
import os
import datetime

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
# Now it's 3 years of monthly data
start_date = (datetime.date.today() - datetime.timedelta(days=365 * 3)).strftime("%Y-%m-%d")
loader  = DataLoader(tickers, start=start_date, interval='1mo')
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
    returns_history   = prices.ffill().pct_change(fill_method=None).dropna()
)

print("")
print(df.round(2).to_string())
print("")
