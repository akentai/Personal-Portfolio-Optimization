from .BaseStrategy import BaseStrategy
import cvxpy as cp
import numpy as np
import pandas as pd

class CVaRStrategy(BaseStrategy):
    def __init__(self, tickers, name="CVaR", alpha=0.95):
        super().__init__(tickers, name)
        self.alpha = alpha

    def optimize(self, current_portfolio, new_capital, price_history, returns_history, **kwargs):
        X = returns_history.values
        T, n = X.shape
        Vt = np.sum(current_portfolio) + new_capital

        w = cp.Variable(n)
        z = cp.Variable(T)
        VaR = cp.Variable()

        loss = -X @ w
        constraints = [
            cp.sum(w) == 1,
            w >= current_portfolio / Vt,
            z >= 0,
            z >= loss - VaR
        ]
        objective = cp.Minimize(VaR + (1 / ((1 - self.alpha) * T)) * cp.sum(z))
        cp.Problem(objective, constraints).solve()

        weights = w.value
        allocation = (weights * Vt - current_portfolio).round(0)
        new_portfolio = current_portfolio + allocation

        return pd.DataFrame({
            'Current Portfolio': current_portfolio,
            'New Allocation': allocation,
            'New Portfolio': new_portfolio,
            'New Weights': weights,
        }, index=self.tickers)