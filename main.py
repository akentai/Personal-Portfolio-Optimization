from data import (
    DataLoader,
    build_cash_benchmark,
    build_rf_benchmark,
    build_spy_benchmark,
    build_etf_benchmark,
)

from backtesting import Backtester

from strategies import (
    MaxSharpeStrategy,
    MaxSortinoStrategy, 
    MomentumStrategy,
    RiskParityStrategy,
    EqualWeightStrategy,
    MeanVarianceOptimizationStrategy,
    CvxPortfolioStrategy,
    BlackLittermanMVO,
    ValueAveragingStrategy,
    MinVarianceStrategy,
    CVaRStrategy
)

from evaluation import (
    plot_all_strategies_cumulative,
    plot_strategy,
    plot_drawdowns,
    plot_rolling_metrics,
    plot_risk_return_scatter,
    plot_time_weighted_returns,
    compute_strategy_metrics
)

import numpy as np
import pandas as pd

# 1) Define your universe and fetch prices (incl. SPY for benchmark)
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
#tickers = ['SPY', 'QQQ', 'VYM']
loader  = DataLoader(tickers, start='2010-01-01', interval='1mo')
prices = loader.fetch_prices()

# 2) Set up initial allocation: e.g. $100k equally split across the 4 stocks
n = len(tickers)
initial_capital    = 0
initial_allocation = np.array([initial_capital / n] * n)

# 3) How much new cash to add each rebalance period?
monthly_cash = 2_000

# 4) Instantiate your strategies
strategies = {
    'EqualWeight': EqualWeightStrategy(tickers),
    'MVO':         MeanVarianceOptimizationStrategy(tickers),
    'RiskParity':  RiskParityStrategy(tickers),
    'MaxSharpe':   MaxSharpeStrategy(tickers),
    'MaxSortino':  MaxSortinoStrategy(tickers),
    'Momentum':    MomentumStrategy(tickers, lookback=6),
    'MomentumDiv': MomentumStrategy(tickers, lookback=6, diversification=True),
    'MinVol':      MinVarianceStrategy(tickers),
    'BL':          BlackLittermanMVO(tickers, implied_weights=[0.25, 0.25, 0.25, 0.25]),
    'CVaR':        CVaRStrategy(tickers),
    'cvx':         CvxPortfolioStrategy(tickers),
    'ValueAvg':    ValueAveragingStrategy(tickers)
}

# 5) Wire up and run the backtest
bt = Backtester(
    strategies         = strategies,
    prices             = prices,
    initial_allocation = initial_allocation,
    monthly_cash       = monthly_cash,
)

#results = bt.run()