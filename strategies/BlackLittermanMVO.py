from .BaseStrategy import BaseStrategy
import cvxpy as cp
import numpy as np
import pandas as pd

class BlackLittermanMVO(BaseStrategy):
    """
    Black-Litterman Mean-Variance Optimizer (simplified version).

    Attributes:
        implied_weights: Investor's prior belief about asset weights.
        tau: Scalar controlling the weight of the prior (lower tau = more confidence in prior).
        risk_aversion: Coefficient penalizing risk in the objective.
    """
    def __init__(self, tickers, name=None, implied_weights=None, tau=0.05, risk_aversion=1.0):
        super().__init__(tickers, name)

        if implied_weights is None:
            implied_weights = np.full(len(tickers), 1 / len(tickers))

        if len(implied_weights) != len(tickers):
            raise ValueError("Number of implied weights must match the number of tickers.")

        self.implied_weights = np.array(implied_weights)
        self.tau = tau
        self.risk_aversion = risk_aversion

    def optimize(self, current_portfolio, new_capital, price_history, returns_history, **kwargs):
        # Compute covariance matrix and ensure it's symmetric
        cov = returns_history.cov().values
        cov = 0.5 * (cov + cov.T)

        V0 = current_portfolio.sum()
        Vt = V0 + new_capital

        # Compute equilibrium expected returns: π = γ * Σ * w_mkt
        pi = self.risk_aversion * cov @ self.implied_weights

        # Simplified Black-Litterman: assume no views, so posterior mean = π
        mu_bl = pi

        # Optimization: maximize muᵀw - γ * wᵀΣw
        n = len(mu_bl)
        w = cp.Variable(n)

        objective = cp.Maximize(mu_bl @ w - self.risk_aversion * cp.quad_form(w, cov))
        constraints = [
            cp.sum(w) == 1,                       # Fully invested
            w >= current_portfolio / Vt           # No selling
        ]

        problem = cp.Problem(objective, constraints)
        problem.solve()

        weights = w.value
        target_portfolio = weights * Vt
        allocation = np.maximum(target_portfolio - current_portfolio, 0)
        new_portfolio = current_portfolio + allocation

        import cvxpy as cp
import numpy as np
import pandas as pd

class BlackLittermanMVO(BaseStrategy):
    """
    Black-Litterman Mean-Variance Optimizer (simplified version).

    Attributes:
        implied_weights: Investor's prior belief about asset weights.
        tau: Scalar controlling the weight of the prior (lower tau = more confidence in prior).
        risk_aversion: Coefficient penalizing risk in the objective.
    """
    def __init__(self, tickers, name=None, implied_weights=None, tau=0.05, risk_aversion=1.0):
        super().__init__(tickers, name)

        if implied_weights is None:
            implied_weights = np.full(len(tickers), 1 / len(tickers))

        if len(implied_weights) != len(tickers):
            raise ValueError("Number of implied weights must match the number of tickers.")

        self.implied_weights = np.array(implied_weights)
        self.tau = tau
        self.risk_aversion = risk_aversion

    def optimize(self, current_portfolio, new_capital, price_history, returns_history, **kwargs):
        # Compute covariance matrix and ensure it's symmetric
        cov = returns_history.cov().values
        cov = 0.5 * (cov + cov.T)

        V0 = current_portfolio.sum()
        Vt = V0 + new_capital

        # Compute equilibrium expected returns: π = γ * Σ * w_mkt
        pi = self.risk_aversion * cov @ self.implied_weights

        # Simplified Black-Litterman: assume no views, so posterior mean = π
        mu_bl = pi

        # Optimization: maximize muᵀw - γ * wᵀΣw
        n = len(mu_bl)
        w = cp.Variable(n)

        objective = cp.Maximize(mu_bl @ w - self.risk_aversion * cp.quad_form(w, cov))
        constraints = [
            cp.sum(w) == 1,                       # Fully invested
            w >= current_portfolio / Vt           # No selling
        ]

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