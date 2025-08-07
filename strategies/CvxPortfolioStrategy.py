import numpy as np
import pandas as pd
import cvxportfolio as cvx
from cvxportfolio.data import UserProvidedMarketData
from .BaseStrategy import BaseStrategy

class CvxPortfolioStrategy(BaseStrategy):
    def __init__(
        self,
        tickers,
        risk_aversion: float = 2.5,
        forecast_risk_aversion: float = 0.5,
        backtest: bool = False,
    ):
        self.tickers = tickers
        self.risk_aversion = risk_aversion
        self.forecast_risk_aversion = forecast_risk_aversion
        self.backtest = backtest

    def optimize(
        self,
        current_portfolio: np.ndarray,  # current $ exposures
        new_capital: float,
        price_history: pd.DataFrame,     # indexed by timestamp
        returns_history: pd.DataFrame   # indexed by timestamp
    ) -> pd.DataFrame:
        #
        V0 = current_portfolio.sum()
        Vt = V0 + new_capital

        # 1) load and preprocess data
        returns = returns_history.tz_localize("UTC")

        # 2) prepare holdings and market data
        tickers = list(returns.columns)
        h0 = pd.Series([0]*len(tickers) + [new_capital],
                      index=tickers + ['cash'])
        md = cvx.UserProvidedMarketData(returns=returns, cash_key='cash')

        # 3) Objective & constraints
        GAMMA = self.risk_aversion
        KAPPA = self.forecast_risk_aversion
        objective = cvx.ReturnsForecast() - GAMMA * (
                    cvx.FullCovariance() + KAPPA * cvx.RiskForecastError()
                )

        w_min = pd.Series(current_portfolio / Vt, index=self.tickers)
        constraints = [
            cvx.LongOnly(),              # no short positions
            cvx.LeverageLimit(1.0),      # sum weights ≤ 1 each period
            cvx.MinWeights(limit=w_min)  # w_i ≥ current_exposure_i/Vt
        ]

        # 4) Optimization
        policy = cvx.MultiPeriodOptimization(
            objective=objective,
            constraints=constraints,
            planning_horizon=12,
            include_cash_return=False
        )

        t0 = md.trading_calendar()[-1]
        u_seq, _, _ = policy.execute(h0, md, t=t0)

        # 5) Results
        allocations = u_seq.values[:-1]
        exposures = current_portfolio + allocations
        weights = exposures / exposures.sum()

        return pd.DataFrame({
            'Current Portfolio': current_portfolio,
            'New Allocation': allocations,
            'New Portfolio': exposures,
            'New Weights': weights
        }, index=self.tickers)