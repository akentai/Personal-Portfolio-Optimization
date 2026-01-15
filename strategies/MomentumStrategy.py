from .BaseStrategy import BaseStrategy
import numpy as np
import pandas as pd

class MomentumStrategy(BaseStrategy):
    """
    Allocates more capital to assets with recent positive momentum,
    optionally smoothed and diversified.

    Parameters:
    - lookback: number of months to compute momentum over (default: 6)
    - diversification: blend signal with equal weight to avoid concentration
    """
    def __init__(self, tickers, name="Momentum", lookback=6, diversification=False, vol_threshold=0.1, **params):
        super().__init__(tickers, name)
        self.lookback = lookback
        self.diversification = diversification
        self.vol_threshold = vol_threshold

    def optimize(self, current_portfolio, new_capital, price_history, returns_history, **kwargs):
        V0 = np.sum(current_portfolio)
        Vt = V0 + new_capital

        # 1. Compute momentum signal: average return over lookback period
        momentum_returns = returns_history[-self.lookback:].mean()

        # 2. Ignore assets with negative momentum
        momentum_returns[momentum_returns < 0] = 0

        # 3. Remove high volatility assets
        vol = returns_history[-self.lookback:].std()
        momentum_returns[vol > self.vol_threshold] = 0

        # 4. Normalize or fallback to equal weights
        equal_weights = np.full(len(self.tickers), 1 / len(self.tickers))
        if momentum_returns.sum() == 0:
            weights = equal_weights
        else:
            weights = (momentum_returns / momentum_returns.sum()).values

        # 5. Optional diversification with uniform portfolio
        if self.diversification:
            weights = 0.7 * weights + 0.3 * equal_weights

        # 6. Convert to portfolio targets
        target_portfolio = weights * Vt
        allocation = np.maximum(target_portfolio - current_portfolio, 0)

        # 7. Rescale allocation to not exceed new capital
        if allocation.sum() > new_capital:
            allocation = allocation / allocation.sum() * new_capital

        new_portfolio = current_portfolio + allocation

        return pd.DataFrame({
            'Current Portfolio': current_portfolio,
            'New Allocation': allocation,
            'New Portfolio': new_portfolio,
            'New Weights': weights,
            'Unused': new_capital - allocation.sum()
        }, index=self.tickers)
