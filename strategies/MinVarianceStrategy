from .BaseStrategy import BaseStrategy
import cvxpy as cp
import pandas as pd
import numpy as np

class MinVarianceStrategy(BaseStrategy):
    """
    Minimum Variance Portfolio Strategy:
    Allocates capital to minimize portfolio variance
    given historical return covariance.
    """
    def __init__(self, tickers, name=None):
        super().__init__(tickers, name)

    def optimize(self, current_portfolio, new_capital, price_history, returns_history, **kwargs):
        # Compute covariance matrix of returns
        cov = returns_history.cov().values
        cov = 0.5 * (cov + cov.T)  # Ensure symmetry

        n = len(cov)
        Vt = np.sum(current_portfolio) + new_capital

        # Optimization variables and objective
        w = cp.Variable(n)
        objective = cp.Minimize(cp.quad_form(w, cov))

        # Constraint: weights must sum to 1, be long-only,
        # and be no less than current allocation proportion
        min_weights = current_portfolio / Vt
        constraints = [
            cp.sum(w) == 1,
            w >= min_weights
        ]

        # Solve the problem
        problem = cp.Problem(objective, constraints)
        problem.solve()

        weights = w.value
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
