import numpy as np
import pandas as pd
from .loader import DataLoader

# Cash benchmark: assumes monthly deposits with no returns
def build_cash_benchmark(dates, initial_capital, monthly_cash):
    n = len(dates)
    return pd.Series(initial_capital + monthly_cash * (np.arange(n) + 1), index=dates, name='Cash')


# Risk-free benchmark: deposits grow at a fixed risk-free rate (default 1.25% annually for ABN AMRO, 2025)
def build_rf_benchmark(dates, initial_capital, monthly_cash, rf_rate=0.0125):
    rf_monthly = rf_rate / 12
    values = []
    v = initial_capital
    for _ in dates:
        v *= (1 + rf_monthly)  # compound existing capital
        v += monthly_cash      # add new deposit
        values.append(v)
    return pd.Series(values, index=dates, name='Risk Free')


# SPY benchmark: simulate monthly investment in SPY ETF (tracks S&P 500)
def build_spy_benchmark(dates, initial_capital, monthly_cash):
    start = dates.min().strftime('%Y-%m-%d')
    end = (dates.max() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    spy = DataLoader(['SPY'], start, end, '1mo').fetch_prices()['SPY'].reindex(dates).ffill().values

    values = []
    v = initial_capital
    for i in range(len(dates)):
        if i > 0:
            v *= spy[i] / spy[i - 1]  # simulate SPY growth
        v += monthly_cash            # add new deposit
        values.append(v)
    return pd.Series(values, index=dates, name='SPY')


# Custom ETF benchmark: simulate a weighted portfolio of ETFs
def build_etf_benchmark(dates, initial_capital, monthly_cash, etfs, weights):
    assert np.isclose(sum(weights), 1.0), "ETF weights must sum to 1"
    prices = DataLoader(etfs, dates.min().strftime('%Y-%m-%d'), dates.max().strftime('%Y-%m-%d'), '1mo').fetch_prices()
    prices = prices.reindex(dates).ffill()

    holdings = np.zeros(len(etfs))
    values = []

    v = initial_capital
    for i in range(len(dates)):
        if i > 0:
            returns = prices.iloc[i] / prices.iloc[i - 1]
            holdings *= returns.values  # apply ETF growth
        v += monthly_cash
        holdings += monthly_cash * np.array(weights)  # invest proportionally
        values.append(holdings.sum())
    
    label = '-'.join(etfs)
    name = f"{label} ({', '.join([f'{int(w*100)}%' for w in weights])})"
    return pd.Series(values, index=dates, name=name)
