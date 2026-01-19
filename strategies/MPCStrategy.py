from .BaseStrategy import BaseStrategy
import cvxpy as cp
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

class MPCStrategy(BaseStrategy):
    """
    Model Predictive Control inspired strategy.
    Forecasts expected returns over a time horizon using ARIMA, then optimizes portfolio.
    """

    def __init__(self, tickers, name="MPC", risk_aversion=1.0, horizon=3):
        super().__init__(tickers, name)
        self.risk_aversion = risk_aversion
        self.horizon = horizon

    def optimize(self, current_portfolio: np.ndarray, new_capital: float, price_history: pd.DataFrame, returns_history: pd.DataFrame):
        """
        Forecast expected returns using ARIMA over horizon, then MVO.
        """
        n = len(self.tickers)
        mu_forecast = np.zeros(n)

        # Suppress statsmodels warnings
        warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
        warnings.filterwarnings("ignore", category=ConvergenceWarning, module="statsmodels")

        for i, ticker in enumerate(self.tickers):
            returns = returns_history[ticker].dropna()
            if len(returns) < 5:  # minimum for ARIMA
                mu_forecast[i] = returns.mean()
                continue
            try:
                # Set frequency to avoid warning
                returns = returns.copy()
                returns.index = pd.date_range(start=returns.index[0], periods=len(returns), freq='MS')
                model = ARIMA(returns, order=(1, 0, 1))
                model_fit = model.fit()
                forecast = model_fit.forecast(steps=self.horizon)
                mu_forecast[i] = forecast.mean()  # average forecasted return
            except:
                mu_forecast[i] = returns.mean()  # fallback

        cov = returns_history.cov().values
        cov = 0.5 * (cov + cov.T)
        cov += 1e-6 * np.eye(n)  # regularization

        # Current portfolio
        A = current_portfolio
        B = new_capital
        V0 = np.sum(A)
        Vt = V0 + B

        # Optimization Problem
        w = cp.Variable(n)

        # Mean-variance objective with forecasted mu
        expected_return = mu_forecast @ w
        risk = cp.quad_form(w, cov)
        objective = cp.Maximize(expected_return - self.risk_aversion * risk)

        # constraints
        constraints = [
            cp.sum(w) == 1,
            w >= A / Vt
        ]
        prob = cp.Problem(objective, constraints)
        prob.solve()

        weights = w.value.round(3)
        allocation = (w.value * Vt - A).round(0).astype(int)
        new_portfolio = (current_portfolio + allocation).round(0).astype(int)

        return pd.DataFrame({
            'Current Portfolio': current_portfolio,
            'New Allocation': allocation,
            'New Portfolio': new_portfolio,
            'New Weights': weights
        }, index=self.tickers)