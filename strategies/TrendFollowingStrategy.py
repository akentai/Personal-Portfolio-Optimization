from .BaseStrategy import BaseStrategy
import numpy as np
import pandas as pd


class TrendFollowingStrategy(BaseStrategy):
    """
    Trend-Following Strategy using a moving-average filter.
    Invest when price is above its long-term average, or when a short
    moving average is above a long moving average.
    """
    def __init__(
        self,
        tickers,
        name="TrendFollowing",
        long_window=10,
        short_window=3,
        **params
    ):
        super().__init__(tickers, name)
        self.long_window = long_window
        self.short_window = short_window

    def optimize(self, current_portfolio, new_capital, price_history, returns_history, **kwargs):
        current_portfolio = np.asarray(current_portfolio, dtype=float)
        V0 = np.sum(current_portfolio)
        Vt = V0 + new_capital
        n_assets = len(self.tickers)

        prices = price_history[self.tickers]
        # Long-term moving average (monthly data by default).
        long_ma = prices.rolling(window=self.long_window, min_periods=1).mean().iloc[-1]

        if self.short_window is not None:
            # Cross-over signal: short MA above long MA.
            short_ma = prices.rolling(window=self.short_window, min_periods=1).mean().iloc[-1]
            signal = short_ma > long_ma
        else:
            # Filter signal: price above long MA.
            last_price = prices.iloc[-1]
            signal = last_price > long_ma

        if signal.sum() == 0:
            # No uptrends: hold cash (no new allocation).
            weights = np.zeros(n_assets)
        else:
            # Equal-weight only the assets in an uptrend.
            weights = (signal.astype(float) / signal.sum()).values

        target_portfolio = weights * Vt
        allocation = np.maximum(target_portfolio - current_portfolio, 0)

        if allocation.sum() > new_capital and allocation.sum() > 0:
            allocation = allocation / allocation.sum() * new_capital

        new_portfolio = current_portfolio + allocation
        new_weights = new_portfolio / new_portfolio.sum() if new_portfolio.sum() > 0 else np.zeros_like(new_portfolio)

        return pd.DataFrame({
            'Current Portfolio': current_portfolio,
            'New Allocation': allocation,
            'New Portfolio': new_portfolio,
            'New Weights': new_weights,
            'Unused': new_capital - allocation.sum()
        }, index=self.tickers)
