from .BaseStrategy import BaseStrategy
import numpy as np
import pandas as pd


class VolatilityTargetingStrategy(BaseStrategy):
    """
    Volatility Targeting Strategy:
    Scales new investment so the portfolio's expected volatility
    is closer to a target level (annualized).
    """
    def __init__(
        self,
        tickers,
        name="VolatilityTargeting",
        target_vol=0.12,
        lookback=12,
        weighting="equal",
        periods_per_year=12,
        **params
    ):
        super().__init__(tickers, name)
        self.target_vol = target_vol
        self.lookback = lookback
        self.weighting = weighting
        self.periods_per_year = periods_per_year

    def optimize(self, current_portfolio, new_capital, price_history, returns_history, **kwargs):
        current_portfolio = np.asarray(current_portfolio, dtype=float)
        V0 = np.sum(current_portfolio)
        n_assets = len(self.tickers)

        # Use trailing returns to estimate risk over the lookback window.
        window = returns_history[self.tickers].tail(self.lookback)

        if self.weighting == "inv_vol":
            # Inverse-volatility base weights reduce risk concentration.
            vol = window.std()
            inv_vol = 1 / vol.replace(0, np.nan)
            inv_vol = inv_vol.fillna(0.0)
            if inv_vol.sum() > 0:
                base_weights = (inv_vol / inv_vol.sum()).values
            else:
                base_weights = np.full(n_assets, 1 / n_assets)
        else:
            # Default to equal-weight base portfolio.
            base_weights = np.full(n_assets, 1 / n_assets)

        # Annualize the covariance to align with target_vol.
        cov = window.cov().values
        cov = 0.5 * (cov + cov.T)
        cov = cov * self.periods_per_year

        # Scale new capital to hit target volatility without selling.
        port_vol = float(np.sqrt(base_weights.T @ cov @ base_weights))
        scale = min(1.0, self.target_vol / port_vol) if port_vol > 0 else 1.0

        allocatable = new_capital * scale
        # Unused cash remains uninvested if target volatility is lower than current risk.
        Vt = V0 + allocatable

        target_portfolio = base_weights * Vt
        allocation = np.maximum(target_portfolio - current_portfolio, 0)

        if allocation.sum() > allocatable and allocation.sum() > 0:
            allocation = allocation / allocation.sum() * allocatable

        new_portfolio = current_portfolio + allocation
        new_weights = new_portfolio / new_portfolio.sum() if new_portfolio.sum() > 0 else np.zeros_like(new_portfolio)

        return pd.DataFrame({
            'Current Portfolio': current_portfolio,
            'New Allocation': allocation,
            'New Portfolio': new_portfolio,
            'New Weights': new_weights,
            'Unused': new_capital - allocation.sum()
        }, index=self.tickers)
