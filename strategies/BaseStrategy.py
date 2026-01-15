from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

class BaseStrategy(ABC):
    """Interface for all strategies."""
    
    @abstractmethod
    def __init__(self, tickers, name, **params):
        """Store any strategy-specific parameters."""
        self.tickers = tickers
        self.name = name

    @abstractmethod
    def optimize(
        self,
        current_portfolio: np.ndarray,
        new_capital: float,
        price_history: pd.DataFrame,
        returns_history: pd.DataFrame,
        **kwargs
    ):
        """
        Given current_weights, cash, and history, compute new allocations.
        Strategies can decide to use prices, returns, covariances, macro series, etc.
        """
        raise NotImplementedError