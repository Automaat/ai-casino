"""Market data fetchers for stock prices and fundamentals."""

import os
from datetime import UTC, date, datetime, timedelta

import pandas as pd
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.cache.historical import HistoricalCache
from src.metrics.execution import timed_operation

load_dotenv()

HTTP_RETRY = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception) & retry_if_not_exception_type(ValueError),
    reraise=True,
    before_sleep=lambda retry_state: logger.warning(
        f"Retry {retry_state.attempt_number} after {retry_state.outcome.exception()}"
    ),
)


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

    def __init__(
        self,
        use_alpha_vantage: bool = True,
        historical_cache: HistoricalCache | None = None,
    ) -> None:
        """Initialize market data fetcher.

        Args:
            use_alpha_vantage: Use Alpha Vantage (True) or yfinance (False)
            historical_cache: Optional permanent cache for OHLCV data
        """
        self.use_alpha_vantage = use_alpha_vantage
        self._cache = historical_cache

        if use_alpha_vantage:
            api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
            if not api_key:
                msg = "ALPHA_VANTAGE_API_KEY not set in environment"
                raise ValueError(msg)
            self.ts = TimeSeries(key=api_key, output_format="pandas")
            logger.info("Initialized Alpha Vantage client")
        else:
            logger.info("Using yfinance for market data")

    @staticmethod
    def _previous_business_day() -> date:
        """Get the previous business day (Mon-Fri)."""
        today = datetime.now(UTC).date()
        weekday = today.weekday()
        if weekday == 0:  # Monday → Friday
            return today - timedelta(days=3)
        if weekday == 6:  # Sunday → Friday
            return today - timedelta(days=2)
        return today - timedelta(days=1)

    def _try_cache(self, symbol: str, period_days: int) -> MarketData | None:
        """Try to serve from cache if fresh enough.

        Args:
            symbol: Stock ticker
            period_days: Required history depth

        Returns:
            MarketData from cache or None
        """
        if not self._cache:
            return None

        last_date = self._cache.get_last_ohlcv_date(symbol)
        if last_date is None:
            return None

        prev_bday = self._previous_business_day()
        cached_rows = self._cache.get_ohlcv_count(symbol)

        if last_date >= prev_bday and cached_rows >= period_days:
            df = self._cache.get_ohlcv(symbol)
            logger.info(f"Cache hit for {symbol} ({cached_rows} rows, last={last_date})")
            return MarketData(symbol=symbol, data=df.tail(period_days), last_updated=datetime.now())

        return None

    def _store_to_cache(self, symbol: str, df: pd.DataFrame) -> None:
        """Store closed-day rows to cache.

        Args:
            symbol: Stock ticker
            df: OHLCV dataframe
        """
        if not self._cache or df.empty:
            return

        today = datetime.now(UTC).date()
        closed_rows = df[df.index.map(lambda d: d.date() if hasattr(d, "date") else d) < today]
        if not closed_rows.empty:
            self._cache.store_ohlcv(symbol, closed_rows)

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

        cached = self._try_cache(symbol, period_days)
        if cached:
            return cached

        source = "alpha_vantage" if self.use_alpha_vantage else "yfinance"
        with timed_operation("market_data_fetch", source=source):
            if self.use_alpha_vantage:
                result = self._fetch_alpha_vantage(symbol)
            else:
                result = self._fetch_yfinance(symbol, period_days)

        self._store_to_cache(symbol, result.data)
        return result

    @HTTP_RETRY
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

    @HTTP_RETRY
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

    @HTTP_RETRY
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
