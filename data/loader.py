import yfinance as yf
import pandas as pd
from datetime import datetime

class DataLoader:
    """
    Loads price series for assets, benchmarks, and macro series (e.g., inflation).
    """
    def __init__(self, tickers: list, start: str = '2010-01-01', end: str = None, interval: str = '1mo'):
        self.tickers    = tickers
        self.start      = start
        self.end        = end or datetime.today().strftime('%Y-%m-%d')
        self.interval   = interval

    def fetch_prices(self) -> pd.DataFrame:
        data = yf.download(
            tickers=self.tickers,
            start=self.start,
            end=self.end,
            interval=self.interval,
            auto_adjust=True,
            progress=False
        )['Close']

        return data.ffill()