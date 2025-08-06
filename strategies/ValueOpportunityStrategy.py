from .BaseStrategy import BaseStrategy
import numpy as np
import pandas as pd

class ValueOpportunityStrategy(BaseStrategy):
    """
    ChatGPT Query
    -------------
    Can you suggest another strategy which takes advantage of good stocks being low and focusing more there (exploiting market opportunities)
    
    ChatGPT Response
    ----------------
    Value Opportunity Strategy:
    Momentum filter: Take top 50% of tickers based on 12-month return.
    Short-term reversion filter: Within those, pick assets with negative 1-month return (i.e., they’ve dipped).
    Score = -1-month return × 12-month return — so the higher the 12mo return and lower the 1mo return, the better.
    """
    
    def __init__(self, tickers, name=None, lookback_long=12, lookback_short=1, top_k=0.5, **params):
        super().__init__(tickers, name)
        self.lookback_long = lookback_long
        self.lookback_short = lookback_short
        self.top_k = top_k  # Top % of performers to consider

    def optimize(self, current_portfolio, new_capital, price_history, returns_history, **kwargs):
        V0 = np.sum(current_portfolio)
        Vt = V0 + new_capital

        # 1. Compute long-term (quality) and short-term (dip) returns
        R_long = returns_history[-self.lookback_long:].mean()
        R_short = returns_history[-self.lookback_short:].mean()

        # 2. Filter top-k performers by long-term return
        k = int(len(self.tickers) * self.top_k)
        top_quality = R_long.sort_values(ascending=False).iloc[:k].index

        # 3. Combine: score = long_term * (- short_term)
        combined_score = (R_long[top_quality] * (-R_short[top_quality])).clip(lower=0)

        if combined_score.sum() == 0:
            weights = np.full(len(self.tickers), 1 / len(self.tickers))
        else:
            weights = pd.Series(0, index=self.tickers)
            weights[top_quality] = combined_score / combined_score.sum()
            weights = weights.values

        target_portfolio = weights * Vt
        allocation = np.maximum(target_portfolio - current_portfolio, 0)

        if allocation.sum() > new_capital:
            allocation = allocation / allocation.sum() * new_capital

        new_portfolio = current_portfolio + allocation

        return pd.DataFrame({
            'Current Portfolio': current_portfolio,
            'New Allocation': allocation,
            'New Portfolio': new_portfolio,
            'New Weights': new_portfolio / new_portfolio.sum(),
            'Unused': np.minimum(0, target_portfolio - current_portfolio)
        }, index=self.tickers)
