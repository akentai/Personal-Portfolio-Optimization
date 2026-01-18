from .BaseStrategy import BaseStrategy
import numpy as np
import pandas as pd
from scipy.optimize import minimize

class MaxSortinoStrategy(BaseStrategy):
    """
    Optimizes portfolio to maximize the Sortino Ratio (return / downside risk).
    - Penalizes only negative volatility (downside deviation).
    - Enforces no selling and full investment.
    """
    def __init__(self, tickers, name="MaxSortino", lookback=-1, **params):
        super().__init__(tickers, name)
        self.lookback = lookback

    def optimize(self, current_portfolio, new_capital, price_history, returns_history, **kwargs):
        V0 = current_portfolio.sum()
        Vt = V0 + new_capital

        reutnrs_history = returns_history if self.lookback == -1 else returns_history.tail(self.lookback)
        mu = returns_history.mean().values
        downside = returns_history.copy()
        downside[downside > 0] = 0
        downside_std = downside.std().values  # Only negative deviations
        n = len(mu)

        min_weights = current_portfolio / Vt  # Enforce buy-only constraint

        # 1. Objective: negative Sortino ratio
        def sortino_neg(w):
            port_return = np.dot(mu, w)
            downside_risk = np.sqrt(np.dot((w * downside_std), (w * downside_std)))
            return -port_return / downside_risk if downside_risk > 0 else np.inf

        # 2. Constraints: fully invested + no selling
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},          # ∑w_i = 1
            {"type": "ineq", "fun": lambda w: w - min_weights}       # w_i ≥ current / Vt
        ]
        bounds = [(0.0, 1.0)] * n
        x0 = np.full(n, 1 / n)

        # 3. Solve optimization (not convex)
        result = minimize(sortino_neg, x0, method="SLSQP", bounds=bounds, constraints=constraints)

        weights = result.x
        target_portfolio = weights * Vt
        allocation = np.maximum(target_portfolio - current_portfolio, 0)
        new_portfolio = current_portfolio + allocation

        return pd.DataFrame({
            'Current Portfolio': current_portfolio,
            'New Allocation': allocation,
            'New Portfolio': new_portfolio,
            'New Weights': weights,
            'Unused': new_capital - allocation.sum()
        }, index=self.tickers)
