from .BaseStrategy import BaseStrategy
import numpy as np
import pandas as pd


class TimeSeriesMeanReversionStrategy(BaseStrategy):
    """
    Time-series mean reversion:
    Score assets by how much their recent return underperforms
    their own EMA of returns, then allocate to top-N laggards.
    """
    def __init__(
        self,
        tickers,
        name="TimeSeriesMeanReversion",
        mean_reversion_lookback=3,
        history_lookback=12,
        top_n=None,
        allow_sells=False,
        **params
    ):
        super().__init__(tickers, name)
        self.mean_reversion_lookback = mean_reversion_lookback
        self.history_lookback = history_lookback
        self.top_n = top_n
        self.allow_sells = allow_sells

    def _zscore(self, series: pd.Series) -> pd.Series:
        series = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
        std = series.std(ddof=0)
        if std == 0 or np.isnan(std):
            return pd.Series(0.0, index=series.index)
        return (series - series.mean()) / std

    def _normalize(self, weights: np.ndarray) -> np.ndarray:
        total = np.sum(weights)
        if total <= 0:
            return np.full(len(weights), 1 / len(weights))
        return weights / total

    def _time_series_score(self, returns_history: pd.DataFrame) -> pd.Series:
        returns = returns_history.reindex(columns=self.tickers)

        if self.mean_reversion_lookback and len(returns) >= self.mean_reversion_lookback:
            window = returns.tail(self.mean_reversion_lookback)
            recent = (1 + window).prod() - 1
        else:
            recent = returns.mean() if len(returns) > 0 else pd.Series(0.0, index=self.tickers)

        if self.history_lookback and len(returns) > 0:
            ema = returns.ewm(span=self.history_lookback, adjust=False).mean().iloc[-1]
        else:
            ema = returns.mean() if len(returns) > 0 else pd.Series(0.0, index=self.tickers)

        # Higher score when recent return is below the EMA of returns.
        return ema - recent

    def optimize(self, current_portfolio, new_capital, price_history, returns_history, **kwargs):
        current_portfolio = np.asarray(current_portfolio, dtype=float)
        V0 = np.sum(current_portfolio)
        Vt = V0 + new_capital

        raw_score = self._time_series_score(returns_history)
        score = self._zscore(raw_score).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        score = score - score.min()

        if self.top_n is not None:
            k = max(1, min(int(self.top_n), len(self.tickers)))
            selected = score.sort_values(ascending=False).iloc[:k].index
        else:
            selected = score.index

        selected_scores = score.loc[selected]
        if selected_scores.sum() <= 0:
            raw_weights = np.full(len(selected_scores), 1 / len(selected_scores))
        else:
            raw_weights = (selected_scores / selected_scores.sum()).values

        weights = pd.Series(0.0, index=self.tickers)
        weights.loc[selected] = self._normalize(raw_weights)

        target_portfolio = weights.values * Vt
        if self.allow_sells:
            allocation = target_portfolio - current_portfolio
        else:
            allocation = np.maximum(target_portfolio - current_portfolio, 0)
            if allocation.sum() > new_capital and allocation.sum() > 0:
                allocation = allocation / allocation.sum() * new_capital

        new_portfolio = current_portfolio + allocation
        new_weights = new_portfolio / new_portfolio.sum() if new_portfolio.sum() > 0 else np.zeros_like(new_portfolio)

        return pd.DataFrame({
            "Current Portfolio": current_portfolio,
            "New Allocation": allocation,
            "New Portfolio": new_portfolio,
            "New Weights": new_weights,
            "Unused": new_capital - allocation.sum()
        }, index=self.tickers)
