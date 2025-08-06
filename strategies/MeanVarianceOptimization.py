from .BaseStrategy import BaseStrategy
import cvxpy as cp
import numpy as np 
import pandas as pd

class MeanVarianceOptimizationStrategy(BaseStrategy):
    """
    Classic Mean-Variance Optimization Strategy.
    Maximizes expected return minus a penalty for risk, scaled by a risk aversion parameter.
    """

    def __init__(self, tickers, risk_aversion=1.0, fractional_shares=True, backtest=False):
        self.tickers = tickers
        # MVO constants
        self.risk_aversion = risk_aversion
        # CP vs MILP
        # For now only CP is supported
        self.fractional_shares = fractional_shares # assumed to be True for this strategy
        self.backtest = backtest

    def optimize(self, current_portfolio: np.ndarray, new_capital: float, price_history: pd.DataFrame, returns_history: pd.DataFrame):
        """
        Parameters:

        Returns:
        - Asset quantity to buy after optimal buy-only rebalancing
        """
        mu = returns_history.mean().values
        cov = returns_history.cov().values
        cov = 0.5 * (cov + cov.T)

        # Current portfolio
        A = current_portfolio
        B = new_capital
        V0 = np.sum(A)
        Vt = V0 + B

        # Optimization Problem
        n = len(mu)
        w = cp.Variable(n)

        # Mean-variance objective: maximize return - risk_aversion * variance
        expected_return = mu @ w
        risk            = cp.quad_form(w, cov)
        objective       = cp.Maximize(expected_return - self.risk_aversion * risk)

        # constraints:
        #    • fully invested: ∑ wᵢ = 1
        #    • buy‐only:       wᵢ ≥ A_expᵢ / Vt  (i.e. exposureᵢ ≥ current exposure)
        constraints = [
          cp.sum(w) == 1,
          w >= A / Vt
        ]
        prob = cp.Problem(objective, constraints)
        prob.solve()

        weights = w.value.round(3)
        allocation = (w.value * Vt - A).round(0).astype(int)
        new_portoflio = (current_portfolio + allocation).round(0).astype(int)

        if self.backtest:
            return current_portfolio, allocation, new_portoflio, w.value

        return pd.DataFrame({
            'Current Portfolio': current_portfolio,
            'New Allocation': allocation,
            'New Portfolio': new_portoflio,
            'New Weights': weights
        }, index=self.tickers)