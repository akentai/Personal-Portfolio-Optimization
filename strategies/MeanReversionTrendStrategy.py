from .BaseStrategy import BaseStrategy
import numpy as np
import pandas as pd


class MeanReversionTrendStrategy(BaseStrategy):
    """
    Mean-reversion allocation: buy recent laggards (undervalued).
    Constraints are applied to the selected assets only.
    """
    def __init__(
        self,
        tickers,
        name="MeanReversion",
        mean_reversion_lookback=6,
        skip_recent=0,
        top_n=None,
        allow_sells=False,
        **params
    ):
        super().__init__(tickers, name)
        self.mean_reversion_lookback = mean_reversion_lookback
        self.skip_recent = skip_recent
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

    def _mean_reversion_score(self, returns_history: pd.DataFrame) -> pd.Series:
        returns = returns_history.reindex(columns=self.tickers)
        if self.mean_reversion_lookback and len(returns) >= self.mean_reversion_lookback + self.skip_recent:
            if self.skip_recent > 0:
                window = returns.iloc[-(self.mean_reversion_lookback + self.skip_recent):-self.skip_recent]
            else:
                window = returns.tail(self.mean_reversion_lookback)
            # Undervaluation proxy: lower cumulative return -> higher score
            cumulative = (1 + window).prod() - 1
            return -cumulative
        if len(returns) > 0:
            return -returns.mean()
        return pd.Series(0.0, index=self.tickers)

    def optimize(self, current_portfolio, new_capital, price_history, returns_history, **kwargs):
        current_portfolio = np.asarray(current_portfolio, dtype=float)
        V0 = np.sum(current_portfolio)
        Vt = V0 + new_capital

        # Score laggards higher (negative cumulative return -> higher score).
        mean_rev_raw = self._mean_reversion_score(returns_history)
        mean_rev_score = self._zscore(mean_rev_raw).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        # Shift to non-negative for proportional weighting.
        combined = mean_rev_score - mean_rev_score.min()

        # Optionally limit to top-N most undervalued names.
        if self.top_n is not None:
            k = max(1, min(int(self.top_n), len(self.tickers)))
            selected = combined.sort_values(ascending=False).iloc[:k].index
        else:
            selected = combined.index

        selected_scores = combined.loc[selected]
        # Fallback to equal weights if all scores are flat/zero.
        if selected_scores.sum() <= 0:
            raw_weights = np.full(len(selected_scores), 1 / len(selected_scores))
        else:
            raw_weights = (selected_scores / selected_scores.sum()).values

        weights = pd.Series(0.0, index=self.tickers)
        weights.loc[selected] = self._normalize(raw_weights)

        target_portfolio = weights.values * Vt
        # Buy-only by default: allocate only new capital unless allow_sells is True.
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
