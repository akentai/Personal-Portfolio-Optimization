from .BaseStrategy import BaseStrategy
import numpy as np
import pandas as pd

class RiskParityStrategy(BaseStrategy):
    def __init__(self, tickers, name=None, lookback=-1):
        super().__init__(tickers, name)
        self.lookback = lookback

    def optimize(self, current_portfolio, new_capital, price_history, returns_history, **kwargs):

        # Current portfolio
        A = current_portfolio
        B = new_capital
        V0 = np.sum(A)
        Vt = V0 + B

        # 1. Compute inverse volatility weights
        vol = returns_history.std() if self.lookback == -1 else returns_history.tail(self.lookback).std()
    
        inv_vol = 1 / vol.replace(0, np.nan)  # Avoid division by zero
        w = inv_vol / inv_vol.sum()

        # 2. Target portfolio allocation
        target_portfolio = Vt * w
        allocation = np.maximum(0, target_portfolio - current_portfolio)
        
        # 3. Rescale allocation to not exceed new capital
        if allocation.sum() > new_capital:
            allocation = allocation / allocation.sum() * new_capital

        # 4. Compute new portfolio values
        new_portfolio = current_portfolio + allocation
        weights = new_portfolio / new_portfolio.sum() if new_portfolio.sum() > 0 else np.zeros_like(new_portfolio)

        return pd.DataFrame({
            'Current Portfolio': current_portfolio,
            'New Allocation': allocation,
            'New Portfolio': new_portfolio,
            'New Weights': weights,
            'Unused': new_capital - allocation.sum()
        }, index=self.tickers)