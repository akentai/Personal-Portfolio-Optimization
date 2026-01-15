from .BaseStrategy import BaseStrategy
import numpy as np
import pandas as pd


class DualMomentumStrategy(BaseStrategy):
    """
    Dual/Relative Momentum Strategy:
    - Absolute filter: keep assets with trailing return above a threshold.
    - Relative filter: allocate to top-ranked assets by momentum.
    """
    def __init__(
        self,
        tickers,
        name="DualMomentum",
        lookback=12,
        top_n=None,
        top_fraction=0.4,
        absolute_threshold=0.0,
        weighting="equal",
        **params
    ):
        super().__init__(tickers, name)
        self.lookback = lookback
        self.top_n = top_n
        self.top_fraction = top_fraction
        self.absolute_threshold = absolute_threshold
        self.weighting = weighting

    def optimize(self, current_portfolio, new_capital, price_history, returns_history, **kwargs):
        current_portfolio = np.asarray(current_portfolio, dtype=float)
        V0 = np.sum(current_portfolio)
        Vt = V0 + new_capital
        n_assets = len(self.tickers)

        weights = np.zeros(n_assets)
        window = returns_history[self.tickers].tail(self.lookback)
        
        
        # Total return over lookback for absolute + relative momentum.
        momentum = (1 + window).prod() - 1
        eligible = momentum[momentum > self.absolute_threshold]

        if not eligible.empty:
            # Select the top assets by momentum.
            if self.top_n is not None:
                k = min(self.top_n, len(eligible))
            else:
                fraction = self.top_fraction if self.top_fraction is not None else 1.0
                k = max(1, int(len(self.tickers) * fraction))
                k = min(k, len(eligible))

            selected = eligible.sort_values(ascending=False).iloc[:k]

            if self.weighting == "momentum":
                # Momentum-weighted allocation among selected assets.
                scores = selected.clip(lower=0)
                if scores.sum() > 0:
                    selected_weights = scores / scores.sum()
                else:
                    selected_weights = pd.Series(1 / len(selected), index=selected.index)
            else:
                # Equal-weight allocation among selected assets.
                selected_weights = pd.Series(1 / len(selected), index=selected.index)

            weight_series = pd.Series(0.0, index=self.tickers)
            weight_series[selected_weights.index] = selected_weights.values
            weights = weight_series.values

        target_portfolio = weights * Vt
        allocation = np.maximum(target_portfolio - current_portfolio, 0)

        if allocation.sum() > new_capital and allocation.sum() > 0:
            allocation = allocation / allocation.sum() * new_capital

        new_portfolio = current_portfolio + allocation
        new_weights = new_portfolio / new_portfolio.sum() if new_portfolio.sum() > 0 else np.zeros_like(new_portfolio)

        return pd.DataFrame({
            'Current Portfolio': current_portfolio,
            'New Allocation': allocation,
            'New Portfolio': new_portfolio,
            'New Weights': new_weights,
            'Unused': new_capital - allocation.sum()
        }, index=self.tickers)
