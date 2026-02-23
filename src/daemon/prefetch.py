"""After-hours data prefetching and caching."""

import asyncio
import hashlib
import time
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field

from src.cache.memory import MemoryTTLCache
from src.data.fundamental import FundamentalDataFetcher
from src.data.market import MarketData, MarketDataFetcher
from src.data.news import NewsArticle, NewsFetcher

# Cache TTLs in seconds
MARKET_DATA_TTL = 57600  # 16 hours — until next close
NEWS_TTL = 14400  # 4 hours
FUNDAMENTALS_TTL = 86400  # 24 hours

# Alpha Vantage rate limit: 5 req/min → 13s between calls for safety
AV_RATE_LIMIT_SLEEP = 13


class PrefetchResult(BaseModel):
    """Result of prefetching data for a single symbol."""

    symbol: str
    market_data: bool
    news: bool
    fundamentals: bool
    duration_ms: float


class PrefetchReport(BaseModel):
    """Report of a complete prefetch run."""

    timestamp: datetime
    results: list[PrefetchResult] = Field(default_factory=list)
    finbert_ready: bool = False
    api_connectivity: dict[str, bool] = Field(default_factory=dict)
    total_duration_seconds: float = 0.0


class DataPrefetcher:
    """Orchestrates after-hours data prefetching into a shared cache."""

    def __init__(
        self,
        market_fetcher: MarketDataFetcher,
        news_fetcher: NewsFetcher,
        fundamental_fetcher: FundamentalDataFetcher,
    ) -> None:
        """Initialize data prefetcher.

        Args:
            market_fetcher: Market data fetcher instance
            news_fetcher: News fetcher instance
            fundamental_fetcher: Fundamental data fetcher instance
        """
        self._market_fetcher = market_fetcher
        self._news_fetcher = news_fetcher
        self._fundamental_fetcher = fundamental_fetcher
        self._cache = MemoryTTLCache()
        logger.info("DataPrefetcher initialized")

    def _cache_key(self, prefix: str, symbol: str) -> str:
        """Generate SHA256 cache key.

        Args:
            prefix: Data type prefix (market, news, fundamentals)
            symbol: Stock ticker symbol

        Returns:
            Truncated SHA256 hash string
        """
        raw = f"{prefix}:{symbol}"
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    async def prefetch_symbol(self, symbol: str) -> PrefetchResult:
        """Fetch and cache all data types for one symbol.

        Fetches market data, news, and fundamentals sequentially.
        Sleeps between Alpha Vantage calls (market + fundamentals) for rate limiting.

        Args:
            symbol: Stock ticker symbol

        Returns:
            PrefetchResult with success/fail per data type
        """
        start = time.perf_counter()
        market_ok = False
        news_ok = False
        fundamentals_ok = False

        # Market data (uses AV or yfinance) - offload blocking I/O
        try:
            market_data = await asyncio.to_thread(self._market_fetcher.fetch_daily, symbol)
            key = self._cache_key("market", symbol)
            cached_data = {
                "symbol": market_data.symbol,
                "data": market_data.data.to_json(orient="split", date_format="iso"),
                "last_updated": market_data.last_updated.isoformat(),
            }
            await asyncio.to_thread(self._cache.set, key, cached_data, expire=MARKET_DATA_TTL)
            market_ok = True
            logger.debug(f"Prefetched market data for {symbol}")
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to prefetch market data for {symbol}: {e}")

        # News (uses Marketaux, no AV rate limit concern)
        try:
            articles = await self._news_fetcher.afetch_company_news(symbol)

            key = self._cache_key("news", symbol)
            await asyncio.to_thread(
                self._cache.set,
                key,
                [a.model_dump(mode="json") for a in articles],
                expire=NEWS_TTL,
            )
            news_ok = True
            logger.debug(f"Prefetched {len(articles)} news articles for {symbol}")
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to prefetch news for {symbol}: {e}")

        # Rate limit sleep before fundamentals (both market and fundamentals use Alpha Vantage)
        await asyncio.sleep(AV_RATE_LIMIT_SLEEP)

        # Fundamentals (uses Alpha Vantage) - offload blocking I/O
        try:
            fundamentals = await asyncio.to_thread(self._fundamental_fetcher.fetch_overview, symbol)
            key = self._cache_key("fundamentals", symbol)
            await asyncio.to_thread(self._cache.set, key, fundamentals, expire=FUNDAMENTALS_TTL)
            fundamentals_ok = True
            logger.debug(f"Prefetched fundamentals for {symbol}")
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to prefetch fundamentals for {symbol}: {e}")

        duration_ms = (time.perf_counter() - start) * 1000
        return PrefetchResult(
            symbol=symbol,
            market_data=market_ok,
            news=news_ok,
            fundamentals=fundamentals_ok,
            duration_ms=duration_ms,
        )

    async def prefetch_watchlist(self, symbols: list[str]) -> PrefetchReport:
        """Prefetch data for all symbols sequentially with rate limiting.

        Args:
            symbols: List of stock ticker symbols

        Returns:
            PrefetchReport with per-symbol results
        """
        start = time.perf_counter()
        results: list[PrefetchResult] = []

        for i, symbol in enumerate(symbols):
            logger.info(f"Prefetching {symbol} ({i + 1}/{len(symbols)})")
            result = await self.prefetch_symbol(symbol)
            results.append(result)

        total_duration = time.perf_counter() - start
        return PrefetchReport(
            timestamp=datetime.now(UTC),
            results=results,
            total_duration_seconds=total_duration,
        )

    def warm_finbert(self) -> bool:
        """Trigger FinBERT model load to warm the singleton.

        Returns:
            True if FinBERT loaded successfully
        """
        try:
            from src.models.sentiment import get_finbert_sentiment

            get_finbert_sentiment()
            logger.info("FinBERT model warmed up")
            return True
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to warm FinBERT: {e}")
            return False

    def check_api_key_presence(self) -> dict[str, bool]:
        """Check if API keys are present in environment.

        Returns:
            Dict mapping service name to key presence status
        """
        import os

        presence: dict[str, bool] = {}

        # Alpha Vantage
        av_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        presence["alpha_vantage"] = bool(av_key)

        # Marketaux
        mx_key = os.getenv("MARKETAUX_API_KEY")
        presence["marketaux"] = bool(mx_key)

        return presence

    def get_cached_market_data(self, symbol: str) -> MarketData | None:
        """Retrieve cached market data for a symbol.

        Args:
            symbol: Stock ticker symbol

        Returns:
            MarketData if cached, None otherwise
        """
        key = self._cache_key("market", symbol)
        cached = self._cache.get(key)
        if cached is None:
            return None

        try:
            from io import StringIO

            if not isinstance(cached, dict):
                msg = f"Expected dict, got {type(cached).__name__}"
                raise TypeError(msg)
            required_fields = ["data", "symbol", "last_updated"]
            missing = [f for f in required_fields if f not in cached]
            if missing:
                msg = f"Missing required fields: {missing}"
                raise TypeError(msg)

            df = pd.read_json(StringIO(cached["data"]), orient="split")
            return MarketData(
                symbol=cached["symbol"],
                data=df,
                last_updated=datetime.fromisoformat(cached["last_updated"]),
            )
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to deserialize cached market data for {symbol}: {e}")
            return None

    def get_cached_news(self, symbol: str) -> list[NewsArticle] | None:
        """Retrieve cached news articles for a symbol.

        Args:
            symbol: Stock ticker symbol

        Returns:
            List of NewsArticle if cached, None otherwise
        """
        key = self._cache_key("news", symbol)
        cached = self._cache.get(key)
        if cached is None:
            return None

        try:
            if not isinstance(cached, list):
                msg = f"Expected list, got {type(cached).__name__}"
                raise TypeError(msg)
            return [NewsArticle.model_validate(a) for a in cached]
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to deserialize cached news for {symbol}: {e}")
            return None

    def get_cached_fundamentals(self, symbol: str) -> dict[str, Any] | None:
        """Retrieve cached fundamental data for a symbol.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Fundamentals dict if cached, None otherwise
        """
        key = self._cache_key("fundamentals", symbol)
        cached = self._cache.get(key)
        if cached is None:
            return None
        try:
            if not isinstance(cached, dict):
                msg = f"Expected dict, got {type(cached).__name__}"
                raise TypeError(msg)
            return cached
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to deserialize cached fundamentals for {symbol}: {e}")
            return None

    def clear_cache(self) -> None:
        """Clear all prefetch cache data."""
        self._cache.clear()
        logger.info("Cleared prefetch cache")

    async def aclose(self) -> None:
        """Close HTTP clients."""
        await self._news_fetcher.aclose()
        logger.debug("DataPrefetcher HTTP clients closed")

    def __repr__(self) -> str:
        """Return string representation."""
        return "DataPrefetcher()"
