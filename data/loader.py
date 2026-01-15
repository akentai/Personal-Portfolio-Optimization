import yfinance as yf
import pandas as pd
from datetime import datetime

class DataLoader:
    """
    Loads price series for assets, benchmarks, and macro series (e.g., inflation).
    """
    def __init__(
        self, 
        tickers: list, 
        start: str = '2010-01-01', 
        end: str = datetime.today().strftime('%Y-%m-%d'), 
        interval: str = '1mo',
        currency: str = 'EUR'
    ):
        self.tickers    = tickers
        self.start      = start
        self.end        = end 
        self.interval   = interval
        self.currency   = currency

    def fetch_prices(self) -> pd.DataFrame:
        data = yf.download(
            tickers=self.tickers,
            start=self.start,
            end=self.end,
            interval=self.interval,
            auto_adjust=True,
            progress=False
        )['Close']

        # Ensure we have a DataFrame, even if single ticker request returns Series
        if isinstance(data, pd.Series):
            data = data.to_frame()
        
        # YFinance returns prices in USD, convert to target currency if needed
        if self.currency != 'USD':
            data = self.convert_to_currency(data)

        return data.ffill()

    def convert_to_currency(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Converts price data to the target currency using historical FX rates.
        """
        # Determine correct FX ticker (e.g., 'EURUSD=X')
        fx_ticker = f"{self.currency}USD=X"
        
        fx_data = yf.download(
            tickers=fx_ticker,
            start=self.start,
            end=self.end,
            interval=self.interval,
            auto_adjust=True,
            progress=False
        )['Close']

        # Resample/Align FX data to match asset prices dates
        fx_data.reindex(data.index).ffill()
        
        data.div(fx_data, axis=0)
        return data