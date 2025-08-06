import numpy as np
import pandas as pd
from .BaseStrategy import BaseStrategy  

class EqualWeightStrategy(BaseStrategy):
    """
    Simple Equal Weighting Strategy.
    Distributes available capital equally across all assets.
    """

    def __init__(self, tickers, name=None):
        super().__init__(tickers, name)

    def optimize(
        self,
        current_portfolio: np.ndarray,
        new_capital: float,
        price_history: pd.DataFrame,
        returns_history: pd.DataFrame,
        **kwargs
    ) -> pd.DataFrame:

        n = len(current_portfolio)
        allocation = np.full(n, new_capital / n)  # equal allocation to each asset
        new_portfolio = current_portfolio + allocation
        weights = new_portfolio / new_portfolio.sum()

        return pd.DataFrame({
            'Current Portfolio': current_portfolio,
            'New Allocation': allocation,
            'New Portfolio': new_portfolio,
            'New Weights': weights.round(3)
        }, index=self.tickers)
