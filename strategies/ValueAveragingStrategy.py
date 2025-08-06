from .BaseStrategy import BaseStrategy
import numpy as np
import pandas as pd

class ValueAveragingStrategy(BaseStrategy):
    """
    Value Averaging strategy injects capital to reach a growing target portfolio value.
    Each month aims to bring the portfolio closer to a target trajectory.

    Parameters:
    - target_growth_rate: monthly target growth rate (e.g., 0.02 for 2%)
    """
    def __init__(self, tickers, name=None, target_growth_rate=0.02, **params):
        super().__init__(tickers, name)
        self.target_growth_rate = target_growth_rate
        self.t = 0  # internal time step

    def optimize(self, current_portfolio, new_capital, price_history, returns_history, **kwargs):
        V0 = np.sum(current_portfolio)  # current portfolio value

        # Compute target value assuming linear injection and growth
        self.t += 1
        Vtarget = new_capital * self.t * (1 + self.target_growth_rate)

        # How much to invest to reach the target
        D = max(0, Vtarget - V0)
        D = min(D, new_capital)  # can’t invest more than available

        # Equal allocation across assets
        n_assets = len(self.tickers)
        equal_alloc = D / n_assets if D > 0 else 0
        target_portfolio = current_portfolio + equal_alloc

        # Compute allocation needed
        allocation = np.maximum(target_portfolio - current_portfolio, 0)

        # Enforce capital constraint
        if allocation.sum() > new_capital:
            allocation = allocation / allocation.sum() * new_capital

        # Update portfolio and weights
        new_portfolio = current_portfolio + allocation
        total_value = new_portfolio.sum()
        weights = new_portfolio / total_value if total_value > 0 else np.zeros_like(new_portfolio)

        return pd.DataFrame({
            'Current Portfolio': current_portfolio,
            'New Allocation': allocation,
            'New Portfolio': new_portfolio,
            'New Weights': weights,
            'Unused': new_capital - allocation.sum()
        }, index=self.tickers)
