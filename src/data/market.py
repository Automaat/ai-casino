"""Market data fetchers for stock prices and fundamentals."""

import asyncio
import os
import zoneinfo
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
from src.strategies.timeframe import MultiTimeframeData, Timeframe

load_dotenv()

ET_TIMEZONE = zoneinfo.ZoneInfo("America/New_York")
INTRADAY_CACHE_TTL_MINUTES = 15

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

    @staticmethod
    def _is_market_hours() -> bool:
        """Check if currently within market hours (4am-8pm ET)."""
        now = datetime.now(ET_TIMEZONE)
        return 4 <= now.hour < 20

    def _try_intraday_cache(self, symbol: str, interval: str) -> pd.DataFrame | None:
        """Try to serve intraday data from cache if fresh enough.

        Args:
            symbol: Stock ticker
            interval: Time interval (60min, 15min)

        Returns:
            DataFrame from cache or None
        """
        if not self._cache:
            return None

        last_dt = self._cache.get_last_intraday_datetime(symbol, interval)
        if last_dt is None:
            return None

        now = datetime.now(UTC)
        if last_dt.tzinfo is None:
            last_dt = last_dt.replace(tzinfo=UTC)

        age_minutes = (now - last_dt).total_seconds() / 60

        if self._is_market_hours():
            if age_minutes < INTRADAY_CACHE_TTL_MINUTES:
                df = self._cache.get_ohlcv_intraday(symbol, interval)
                logger.info(f"Intraday cache hit for {symbol} ({interval}, age={age_minutes:.1f}m)")
                return df
        else:
            df = self._cache.get_ohlcv_intraday(symbol, interval)
            if not df.empty:
                logger.info(f"Using cached intraday data outside market hours ({symbol}, {interval})")
                return df

        return None

    def _store_intraday_to_cache(self, symbol: str, interval: str, df: pd.DataFrame) -> None:
        """Store intraday bars to cache.

        Args:
            symbol: Stock ticker
            interval: Time interval (60min, 15min)
            df: OHLCV dataframe
        """
        if not self._cache or df.empty:
            return

        now = datetime.now(UTC)
        cutoff_time = now - timedelta(minutes=INTRADAY_CACHE_TTL_MINUTES)

        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is None:
            cutoff_for_filter = cutoff_time.replace(tzinfo=None)
        else:
            cutoff_for_filter = cutoff_time

        closed_rows = df[df.index < cutoff_for_filter] if self._is_market_hours() else df

        if not closed_rows.empty:
            self._cache.store_ohlcv_intraday(symbol, interval, closed_rows)

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

        cached = self._try_intraday_cache(symbol, interval)
        if cached is not None and not cached.empty:
            return MarketData(symbol=symbol, data=cached, last_updated=datetime.now())

        logger.info(f"Fetching intraday data for {symbol} ({interval})")

        try:
            data, _ = self.ts.get_intraday(symbol=symbol, interval=interval, outputsize="compact")

            data = data.sort_index()
            data.columns = ["Open", "High", "Low", "Close", "Volume"]
            data.index.name = "DateTime"

            logger.info(f"Fetched {len(data)} intraday rows")

            self._store_intraday_to_cache(symbol, interval, data)

            return MarketData(
                symbol=symbol,
                data=data,
                last_updated=datetime.now(),
            )
        except Exception as e:
            logger.error(f"Intraday fetch failed: {e}")
            raise

    async def fetch_multi_timeframe(
        self,
        symbol: str,
        timeframes: list[Timeframe] | None = None,
        period_days: int = 90,
    ) -> MultiTimeframeData:
        """Fetch market data for multiple timeframes in parallel.

        Args:
            symbol: Stock ticker symbol
            timeframes: List of timeframes to fetch (defaults to [DAILY, HOURLY])
            period_days: Number of days of historical data for daily timeframe

        Returns:
            MultiTimeframeData with data for each timeframe

        Raises:
            ValueError: If no timeframe data could be fetched
        """
        if timeframes is None:
            timeframes = [Timeframe.DAILY, Timeframe.HOURLY]

        logger.info(f"Fetching multi-timeframe data for {symbol}: {timeframes}")

        async def fetch_timeframe(tf: Timeframe) -> tuple[Timeframe, pd.DataFrame | None]:
            """Fetch data for a single timeframe."""
            try:
                if tf == Timeframe.DAILY:
                    result = await asyncio.to_thread(self.fetch_daily, symbol, period_days)
                    return (tf, result.data)
                if tf == Timeframe.HOURLY:
                    result = await asyncio.to_thread(self.fetch_intraday, symbol, "60min")
                    return (tf, result.data)
                if tf == Timeframe.FIFTEEN_MIN:
                    result = await asyncio.to_thread(self.fetch_intraday, symbol, "15min")
                    return (tf, result.data)
                logger.warning(f"Unsupported timeframe: {tf}")
                return (tf, None)
            except Exception as e:
                logger.warning(f"Failed to fetch {tf} data for {symbol}: {e}")
                return (tf, None)

        results = await asyncio.gather(*[fetch_timeframe(tf) for tf in timeframes])

        timeframe_dict = {}
        for tf, data in results:
            if data is not None and not data.empty:
                timeframe_dict[tf] = data
            else:
                logger.warning(f"No data available for {tf}, skipping")

        if not timeframe_dict:
            msg = f"No timeframe data could be fetched for {symbol}"
            logger.error(msg)
            raise ValueError(msg)

        logger.info(f"Successfully fetched {len(timeframe_dict)}/{len(timeframes)} timeframes for {symbol}")

        return MultiTimeframeData(
            symbol=symbol,
            timeframes=timeframe_dict,
            last_updated=datetime.now(),
        )

    def fetch_overnight_futures(self, symbols: list[str]) -> dict[str, float]:
        """Fetch overnight futures % change.

        Args:
            symbols: Futures symbols (e.g., ["ES=F", "NQ=F"])

        Returns:
            Dict mapping symbol to % change from previous close

        Raises:
            ValueError: If no data available
        """
        logger.info(f"Fetching overnight futures: {symbols}")
        results = {}

        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="2d")

                if data.empty or len(data) < 2:
                    logger.warning(f"Insufficient data for {symbol}, skipping")
                    continue

                prev_close = data["Close"].iloc[-2]
                current_price = data["Close"].iloc[-1]
                pct_change = ((current_price - prev_close) / prev_close) * 100

                results[symbol] = round(pct_change, 2)
                logger.debug(f"{symbol}: {pct_change:+.2f}%")

            except Exception as e:
                logger.warning(f"Failed to fetch {symbol}: {e}")
                continue

        if not results:
            msg = "No futures data available"
            logger.error(msg)
            raise ValueError(msg)

        return results

    def __repr__(self) -> str:
        """String representation."""
        source = "Alpha Vantage" if self.use_alpha_vantage else "yfinance"
        return f"MarketDataFetcher(source={source})"
