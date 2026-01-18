from .BaseStrategy import BaseStrategy
import numpy as np
import pandas as pd
from scipy.optimize import minimize

class MaxSharpeStrategy(BaseStrategy):
    """
    Mean-Variance Optimization Strategy to Maximize Sharpe Ratio.
    - Uses expected returns and covariance matrix from historical data.
    - Applies long-only constraint and enforces minimum weights to prevent selling.
    """
    def __init__(self, tickers, name="MaxSharpe", lookback=-1, **params):
        super().__init__(tickers, name)
        self.lookback = lookback

    def optimize(self, current_portfolio, new_capital, price_history, returns_history, **kwargs):
        V0 = current_portfolio.sum()
        Vt = V0 + new_capital

        returns_history = returns_history if self.lookback == -1 else returns_history.tail(self.lookback)
        mu = returns_history.mean().values              # Expected returns
        cov = returns_history.cov().values              # Covariance matrix
        n = len(mu)
        min_weights = current_portfolio / Vt            # Buy-only constraint

        # 1. Define objective: Negative Sharpe Ratio
        def sharpe_neg(w):
            port_return = np.dot(mu, w)
            port_vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))
            return -port_return / port_vol if port_vol > 0 else np.inf

        # 2. Constraints: fully invested + no selling
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},          # ∑w_i = 1
            {"type": "ineq", "fun": lambda w: w - min_weights}       # w_i ≥ current / Vt
        ]
        bounds = [(0.0, 1.0)] * n
        x0 = np.full(n, 1 / n)

        # 3. Solve optimization (not convex)
        result = minimize(sharpe_neg, x0, method="SLSQP", bounds=bounds, constraints=constraints)

        weights = result.x
        target_portfolio = weights * Vt
        allocation = np.maximum(target_portfolio - current_portfolio, 0)
        new_portfolio = current_portfolio + allocation

        return pd.DataFrame({
            'Current Portfolio': current_portfolio,
            'New Allocation': allocation,
            'New Portfolio': new_portfolio,
            'New Weights': weights,
            'Unused': new_capital - allocation.sum()  # Remaining cash not used
        }, index=self.tickers)
