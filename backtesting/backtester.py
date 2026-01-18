import pandas as pd
import numpy as np

class Backtester:
    """
    Runs backtests for any number of strategies.
    
    Parameters:
    - strategies: dict of {name: BaseStrategy instance}
    - prices: pd.DataFrame of prices (assets in columns)
    - initial_allocation: np.ndarray of starting $ per asset
    - monthly_cash: float, amount to inject each period
    - rolling_window: int or None, size of the lookback window
                      used to compute historical data for strategy optimization
    """
    def __init__(
        self,
        strategies: dict,
        prices: pd.DataFrame,
        initial_allocation: np.ndarray,
        monthly_cash: float,
        rolling_window: int = 12  
    ):
        self.strategies = strategies
        self.prices = prices
        self.tickers = prices.columns.tolist()
        self.returns = prices.ffill().pct_change(fill_method=None).dropna()
        self.dates = self.returns.index
        self.initial_allocation = initial_allocation
        self.monthly_cash = monthly_cash
        self.rolling_window = rolling_window

    def run(self) -> pd.DataFrame:
        """
        Executes the backtest for each strategy over time.

        Returns:
        Dictionary keyed by strategy name, with values:
        {
            'asset_values':  DataFrame(dates x tickers) — dollar value per asset,
            'allocations':   DataFrame(dates x tickers) — new capital allocations,
            'weights':       DataFrame(dates x tickers) — portfolio weights,
            'total_values':  Series(dates) — total portfolio value over time
        }
        """
        if not self.strategies:
            raise ValueError('No strategies provided.')

        # 1) Initialize data storage buffers per strategy
        buffers = {}
        for name in self.strategies:
            buffers[name] = {
                'curr':     self.initial_allocation.copy(),  # Current portfolio ($ exposure per asset)
                'vals':     [],  # New portfolio values
                'allocs':   [],  # New allocations (new capital)
                'weights':  [],  # Portfolio weights
                'total':    []   # Total portfolio value
            }

        n = len(self.dates)
        start = self.rolling_window
        dates = self.dates[start:]  # Dates to iterate over after warmup window

        # 2) Time evolution loop
        for idx in range(start, n):
            # Slice rolling history for strategy use
            hist_start   = max(0, idx - self.rolling_window)
            price_hist   = self.prices.iloc[hist_start : idx + 1]
            returns_hist = self.returns.iloc[hist_start : idx + 1]

            # Compute asset price growth between last 2 steps
            prev_price = price_hist.iloc[-2]
            curr_price = price_hist.iloc[-1]
            growth = (curr_price / prev_price)

            # Run each strategy
            for name, strat in self.strategies.items():
                buf = buffers[name]

                # 3) Simulate market growth of current portfolio
                curr_holdings = buf['curr'] * growth  # simulate market return
                curr_holdings = np.array(curr_holdings)

                # 4) Ask strategy to compute new allocation using new capital
                df = strat.optimize(
                    current_portfolio=curr_holdings,
                    new_capital=self.monthly_cash,
                    price_history=price_hist,
                    returns_history=returns_hist
                )

                # 5) Extract results from strategy output
                new_allocs = np.asarray(df['New Allocation'].values, dtype=int)  # how new cash is allocated
                new_vals   = np.asarray(df['New Portfolio'].values, dtype=int)   # total new value of portfolio
                new_wts    = np.asarray(df['New Weights'].values, dtype=float)   # new portfolio weights
                # Subtract transaction fees (1.75EUR per trade)
                new_vals   = new_vals - (1.75 * (new_allocs > 0).astype(int))
                total_val  = new_vals.sum()  # total portfolio value

                # 6) Store current step values
                buf['allocs'].append(new_allocs)  
                buf['vals'].append(new_vals)
                buf['weights'].append(new_wts)
                buf['total'].append(total_val)

                # 7) Update portfolio state for next month
                buf['curr'] = new_vals

        # 8) Convert buffers to DataFrames for each strategy
        results = {}
        for name, buf in buffers.items():
            idxs = dates
            results[name] = {
                'asset_values': pd.DataFrame(buf['vals'],     index=idxs, columns=self.tickers),
                'allocations' : pd.DataFrame(buf['allocs'],   index=idxs, columns=self.tickers),
                'weights'     : pd.DataFrame(buf['weights'],  index=idxs, columns=self.tickers),
                'total_values': pd.Series(buf['total'],       index=idxs, name='Total Value')
            }

        return results