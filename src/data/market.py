"""Market data fetchers for stock prices and fundamentals."""

import os
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel

load_dotenv()


class MarketData(BaseModel):
    """Market data container."""

    symbol: str
    data: pd.DataFrame
    last_updated: datetime

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True

    @property
    def latest_close(self) -> float:
        """Get latest closing price."""
        return float(self.data["Close"].iloc[-1])

    @property
    def latest_volume(self) -> float:
        """Get latest volume."""
        return float(self.data["Volume"].iloc[-1])


class MarketDataFetcher:
    """Fetch market data from Alpha Vantage or yfinance."""

    def __init__(self, use_alpha_vantage: bool = True) -> None:
        """Initialize market data fetcher.

        Args:
            use_alpha_vantage: Use Alpha Vantage (True) or yfinance (False)
        """
        self.use_alpha_vantage = use_alpha_vantage

        if use_alpha_vantage:
            api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
            if not api_key:
                msg = "ALPHA_VANTAGE_API_KEY not set in environment"
                raise ValueError(msg)
            self.ts = TimeSeries(key=api_key, output_format="pandas")
            logger.info("Initialized Alpha Vantage client")
        else:
            logger.info("Using yfinance for market data")

    def fetch_daily(
        self,
        symbol: str,
        period_days: int = 90,
    ) -> MarketData:
        """Fetch daily OHLCV data.

        Args:
            symbol: Stock ticker symbol
            period_days: Number of days of historical data

        Returns:
            MarketData with OHLCV dataframe
        """
        logger.info(f"Fetching {period_days} days of data for {symbol}")

        if self.use_alpha_vantage:
            return self._fetch_alpha_vantage(symbol)
        return self._fetch_yfinance(symbol, period_days)

    def _fetch_alpha_vantage(self, symbol: str) -> MarketData:
        """Fetch from Alpha Vantage API."""
        try:
            data, _ = self.ts.get_daily(symbol=symbol, outputsize="compact")

            data = data.sort_index()
            data.columns = ["Open", "High", "Low", "Close", "Volume"]
            data.index.name = "Date"

            logger.info(f"Fetched {len(data)} rows from Alpha Vantage")

            return MarketData(
                symbol=symbol,
                data=data,
                last_updated=datetime.now(),
            )
        except Exception as e:
            logger.error(f"Alpha Vantage fetch failed: {e}")
            raise

    def _fetch_yfinance(self, symbol: str, period_days: int) -> MarketData:
        """Fetch from yfinance."""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=period_days)

            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)

            if data.empty:
                msg = f"No data returned for {symbol}"
                raise ValueError(msg)

            data.index.name = "Date"
            logger.info(f"Fetched {len(data)} rows from yfinance")

            return MarketData(
                symbol=symbol,
                data=data,
                last_updated=datetime.now(),
            )
        except Exception as e:
            logger.error(f"yfinance fetch failed: {e}")
            raise

    def fetch_intraday(self, symbol: str, interval: str = "5min") -> MarketData:
        """Fetch intraday data.

        Args:
            symbol: Stock ticker symbol
            interval: Time interval (1min, 5min, 15min, 30min, 60min)

        Returns:
            MarketData with intraday OHLCV dataframe
        """
        if not self.use_alpha_vantage:
            msg = "Intraday data only available with Alpha Vantage"
            raise ValueError(msg)

        logger.info(f"Fetching intraday data for {symbol} ({interval})")

        try:
            data, _ = self.ts.get_intraday(symbol=symbol, interval=interval, outputsize="compact")

            data = data.sort_index()
            data.columns = ["Open", "High", "Low", "Close", "Volume"]
            data.index.name = "DateTime"

            logger.info(f"Fetched {len(data)} intraday rows")

            return MarketData(
                symbol=symbol,
                data=data,
                last_updated=datetime.now(),
            )
        except Exception as e:
            logger.error(f"Intraday fetch failed: {e}")
            raise

    def __repr__(self) -> str:
        """String representation."""
        source = "Alpha Vantage" if self.use_alpha_vantage else "yfinance"
        return f"MarketDataFetcher(source={source})"
