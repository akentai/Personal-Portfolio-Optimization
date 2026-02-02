from .BaseStrategy import BaseStrategy
import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception:  # pragma: no cover - optional dependency
    yf = None


class QualityValueMomentumStrategy(BaseStrategy):
    """
    Combines valuation, quality, and momentum into a composite score.
    - Valuation: lower multiples score higher (PE, PB, EV/EBITDA).
    - Quality: higher ROE/margins/revenue growth and lower debt score higher.
    - Momentum: medium-term price strength (skip most recent month).

    If fundamentals are missing, the model falls back to available signals.
    """
    def __init__(
        self,
        tickers,
        name="QVM",
        lookback=12,
        skip_recent=1,
        value_weight=0.35,
        quality_weight=0.35,
        momentum_weight=0.30,
        rebalance_month=1,
        allow_sells=True,
        fundamentals=None,
        **params
    ):
        super().__init__(tickers, name)
        self.lookback = lookback
        self.skip_recent = skip_recent
        self.value_weight = value_weight
        self.quality_weight = quality_weight
        self.momentum_weight = momentum_weight
        self.rebalance_month = rebalance_month
        self.allow_sells = allow_sells
        self._fundamentals = fundamentals
        self._fundamentals_cache = None

    def _zscore(self, series: pd.Series) -> pd.Series:
        series = series.astype(float)
        std = series.std(ddof=0)
        if std == 0 or np.isnan(std):
            return pd.Series(0.0, index=series.index)
        return (series - series.mean()) / std

    def _clean_positive_series(self, series: pd.Series) -> pd.Series:
        series = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
        series = series.where(series > 0)
        median = series.median()
        return series.fillna(median)

    def _fetch_fundamentals(self) -> pd.DataFrame:
        if self._fundamentals_cache is not None:
            return self._fundamentals_cache

        if self._fundamentals is not None:
            if isinstance(self._fundamentals, pd.DataFrame):
                fundamentals = self._fundamentals.copy()
            else:
                fundamentals = pd.DataFrame(self._fundamentals)
            fundamentals = fundamentals.reindex(self.tickers)
            self._fundamentals_cache = fundamentals
            return fundamentals

        if yf is None:
            fundamentals = pd.DataFrame(index=self.tickers)
            self._fundamentals_cache = fundamentals
            return fundamentals

        records = {}
        for ticker in self.tickers:
            try:
                info = yf.Ticker(ticker).info or {}
            except Exception:
                info = {}
            records[ticker] = {
                "trailingPE": info.get("trailingPE"),
                "forwardPE": info.get("forwardPE"),
                "priceToBook": info.get("priceToBook"),
                "enterpriseToEbitda": info.get("enterpriseToEbitda"),
                "returnOnEquity": info.get("returnOnEquity"),
                "profitMargins": info.get("profitMargins"),
                "operatingMargins": info.get("operatingMargins"),
                "revenueGrowth": info.get("revenueGrowth"),
                "debtToEquity": info.get("debtToEquity"),
            }
        fundamentals = pd.DataFrame.from_dict(records, orient="index")
        self._fundamentals_cache = fundamentals
        return fundamentals

    def _compute_value_score(self, fundamentals: pd.DataFrame) -> pd.Series:
        value_fields = ["trailingPE", "forwardPE", "priceToBook", "enterpriseToEbitda"]
        components = []
        for field in value_fields:
            if field not in fundamentals.columns:
                continue
            series = self._clean_positive_series(fundamentals[field])
            if series.isna().all():
                continue
            components.append(self._zscore(-np.log(series)))
        if not components:
            return pd.Series(0.0, index=self.tickers)
        return pd.concat(components, axis=1).mean(axis=1)

    def _compute_quality_score(self, fundamentals: pd.DataFrame) -> pd.Series:
        quality_fields = ["returnOnEquity", "profitMargins", "operatingMargins", "revenueGrowth"]
        components = []
        for field in quality_fields:
            if field not in fundamentals.columns:
                continue
            series = pd.to_numeric(fundamentals[field], errors="coerce").replace([np.inf, -np.inf], np.nan)
            series = series.fillna(series.median())
            if series.isna().all():
                continue
            components.append(self._zscore(series))

        if "debtToEquity" in fundamentals.columns:
            debt = self._clean_positive_series(fundamentals["debtToEquity"])
            if not debt.isna().all():
                components.append(self._zscore(-debt))

        if not components:
            return pd.Series(0.0, index=self.tickers)
        return pd.concat(components, axis=1).mean(axis=1)

    def _compute_momentum_score(self, returns_history: pd.DataFrame, price_history: pd.DataFrame) -> pd.Series:
        returns_history = returns_history.reindex(columns=self.tickers)
        price_history = price_history.reindex(columns=self.tickers)

        window = self.lookback
        skip = self.skip_recent

        momentum_raw = None
        if len(returns_history) >= window + skip and window > 0:
            if skip > 0:
                window_returns = returns_history.iloc[-(window + skip):-skip]
            else:
                window_returns = returns_history.iloc[-window:]
            momentum_raw = (1 + window_returns).prod() - 1
        elif len(price_history) >= window + skip + 1 and window > 0:
            end_idx = -1 - skip if skip > 0 else -1
            start_idx = end_idx - window
            momentum_raw = price_history.iloc[end_idx] / price_history.iloc[start_idx] - 1
        elif len(returns_history) > 0:
            momentum_raw = returns_history.mean()

        if momentum_raw is None:
            return pd.Series(0.0, index=self.tickers)
        return self._zscore(momentum_raw.fillna(momentum_raw.median()))

    def _rebalance_now(self, price_history: pd.DataFrame) -> bool:
        if not self.allow_sells or self.rebalance_month is None:
            return False
        if price_history is None or len(price_history) == 0:
            return False
        try:
            last_date = pd.to_datetime(price_history.index[-1])
            return last_date.month == self.rebalance_month
        except Exception:
            return False

    def optimize(self, current_portfolio, new_capital, price_history, returns_history, **kwargs):
        V0 = np.sum(current_portfolio)
        Vt = V0 + new_capital

        fundamentals = kwargs.get("fundamentals")
        if fundamentals is None:
            fundamentals = self._fetch_fundamentals()
        fundamentals = fundamentals.reindex(self.tickers)

        value_score = self._compute_value_score(fundamentals)
        quality_score = self._compute_quality_score(fundamentals)
        momentum_score = self._compute_momentum_score(returns_history, price_history)

        signal_weights = {
            "value": self.value_weight,
            "quality": self.quality_weight,
            "momentum": self.momentum_weight,
        }

        active = []
        for name, series in [
            ("value", value_score),
            ("quality", quality_score),
            ("momentum", momentum_score),
        ]:
            if series.isna().all():
                signal_weights[name] = 0.0
            else:
                active.append(name)

        total_weight = sum(signal_weights.values())
        if total_weight == 0:
            signal_weights = {"value": 0.0, "quality": 0.0, "momentum": 1.0}
            total_weight = 1.0

        for key in signal_weights:
            signal_weights[key] = signal_weights[key] / total_weight

        composite = (
            signal_weights["value"] * value_score.fillna(0.0)
            + signal_weights["quality"] * quality_score.fillna(0.0)
            + signal_weights["momentum"] * momentum_score.fillna(0.0)
        )
        composite = composite - composite.min()

        if composite.sum() == 0:
            weights = np.full(len(self.tickers), 1 / len(self.tickers))
        else:
            weights = (composite / composite.sum()).values

        target_portfolio = weights * Vt
        if self._rebalance_now(price_history):
            allocation = target_portfolio - current_portfolio
        else:
            allocation = np.maximum(target_portfolio - current_portfolio, 0)
            if allocation.sum() > new_capital:
                allocation = allocation / allocation.sum() * new_capital

        new_portfolio = current_portfolio + allocation

        return pd.DataFrame({
            "Current Portfolio": current_portfolio,
            "New Allocation": allocation,
            "New Portfolio": new_portfolio,
            "New Weights": new_portfolio / new_portfolio.sum(),
            "Unused": new_capital - allocation.sum()
        }, index=self.tickers)
