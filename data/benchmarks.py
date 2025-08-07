import numpy as np
import pandas as pd
from .loader import DataLoader

# Cash benchmark: assumes monthly deposits with no returns
def build_cash_benchmark(
    dates: pd.DatetimeIndex,
    initial_capital: float,
    monthly_cash: float
) -> pd.Series:
    """
    Cash-only benchmark: no growth, just accumulating contributions.
    """
    n = len(dates)
    t = np.arange(n)
    values = initial_capital + monthly_cash * (t + 1)

    return pd.Series(values, index=dates, name='Cash')



# Risk-free benchmark: deposits grow at a fixed risk-free rate (default 1.25% annually for ABN AMRO, 2025)
def build_rf_benchmark(
    dates: pd.DatetimeIndex,
    initial_capital: float,
    monthly_cash: float,
    rf_rate: float = 0.0125,
) -> pd.Series:
    """
    Risk-free benchmark: compounds at monthly RF rate + contributions.
    """
    rf_monthly = rf_rate / 12
    n = len(dates)
    values = np.zeros(n)
    v = initial_capital

    for i in range(n):
        if i > 0:
            v *= (1 + rf_monthly)
        v += monthly_cash
        values[i] = v

    return pd.Series(values, index=dates, name='Risk Free')


# SPY benchmark: simulate monthly investment in SPY ETF (tracks S&P 500)
def build_spy_benchmark(
    dates: pd.DatetimeIndex,
    initial_capital: float,
    monthly_cash: float
) -> pd.Series:
    """
    SPY-only benchmark: fetches SPY prices internally and applies returns + monthly contributions.
    """
    # Determine date range from backtest index
    start = dates.min().strftime('%Y-%m-%d')
    # include the last date
    end = (dates.max() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')

    # Fetch SPY monthly-adjusted closes
    loader = DataLoader(['SPY'], start=start, end=end, interval='1mo')
    spy_data = loader.fetch_prices()

    # Align SPY to backtest dates
    spy = spy_data.reindex(dates).ffill()
    spy = spy.to_numpy().squeeze()

    # Build the equity curve with monthly cash injections
    values = np.zeros(len(dates))
    v = initial_capital
    for i in range(len(dates)):
        if i > 0:
            r = spy[i] / spy[i-1]
            v *= r
        v += monthly_cash
        values[i] = v

    return pd.Series(values, index=dates, name='S&P 500')


# Custom benchmark: simulate a weighted portfolio of ETFs (works with stocks too)
def build_custom_benchmark(
    dates: pd.DatetimeIndex,
    initial_capital: float,
    monthly_cash: float,
    etfs: list,
    weights: list
) -> pd.DataFrame:
    """
    Builds an equity curve for a benchmark of multiple ETFs.

    Parameters:
        dates: Backtest dates at monthly frequency
        initial_capital: starting portfolio value
        monthly_cash: total new capital per period
        etfs: list of ETF tickers (e.g. ['SPY', 'QQQ', 'VYM'])
        weights: allocation percentages (must sum to 1)

    Returns:
        DataFrame with columns for each ETF and 'Total' portfolio value.
    """

    if sum(weights) != 1.0:
        raise Exception("Weights do not sum up to 1")

    # Determine date range from backtest index
    start = dates.min().strftime('%Y-%m-%d')
    # include the last date
    end = (dates.max() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')

    # Fetch SPY monthly-adjusted closes
    loader = DataLoader(etfs, start=start, end=end, interval='1mo')
    prices_data = loader.fetch_prices()

    # Align SPY to backtest dates
    prices = prices_data.reindex(dates).ffill()
    prices = prices.to_numpy().squeeze()

    # 2. Initialize portfolio values and holdings in €
    values = pd.DataFrame(index=dates, columns=etfs)
    holdings = np.zeros(len(etfs))
    cash = initial_capital

    for i, date in enumerate(dates):
        if i > 0:
            returns = prices[i] / prices[i-1]
            holdings = holdings * returns  # update exposure
        cash += monthly_cash
        for idx in range(len(etfs)):
            alloc_cash = monthly_cash * weights[idx]
            change = alloc_cash  # add money
            holdings[idx] += change
        values.loc[date] = holdings

    values['Total'] = values.sum(axis=1)

    # Naming convention
    label = '-'.join(etfs)
    weight_str = ', '.join([f"{int(w * 100)}%" for w in weights])
    name = f"{label} ({weight_str})"

    return pd.Series(values['Total'], index=dates, name=name)