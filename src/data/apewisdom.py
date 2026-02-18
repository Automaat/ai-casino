"""ApeWisdom API fetcher — WSB trending tickers with mentions and ranks."""

from __future__ import annotations

import time

import httpx
from loguru import logger
from pydantic import BaseModel


class ApeWisdomTicker(BaseModel):
    """Single trending ticker from ApeWisdom."""

    rank: int
    ticker: str
    name: str
    mentions: int
    upvotes: int
    rank_24h_ago: int
    mentions_24h_ago: int

    def __repr__(self) -> str:
        """String representation."""
        return f"ApeWisdomTicker({self.ticker} rank={self.rank} mentions={self.mentions})"


class ApeWisdomFetcher:
    """Fetches trending tickers from ApeWisdom with in-memory TTL cache."""

    BASE_URL = "https://apewisdom.io/api/v1.0/filter/all-stocks"

    def __init__(self, cache_ttl: int = 300) -> None:
        """Initialize fetcher.

        Args:
            cache_ttl: Cache time-to-live in seconds (default 5min)
        """
        self._cache_ttl = cache_ttl
        self._cache: list[ApeWisdomTicker] = []
        self._cache_time: float = 0.0

    def _is_cache_valid(self) -> bool:
        """Check if cached data is still fresh."""
        return bool(self._cache) and (time.monotonic() - self._cache_time) < self._cache_ttl

    def fetch_trending(self) -> list[ApeWisdomTicker]:
        """Fetch all trending tickers (cached).

        Returns:
            List of trending tickers sorted by rank
        """
        if self._is_cache_valid():
            return self._cache

        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(self.BASE_URL)
                resp.raise_for_status()

            data = resp.json()
            results = data.get("results", [])

            tickers = []
            for item in results:
                try:
                    tickers.append(
                        ApeWisdomTicker(
                            rank=item.get("rank", 0),
                            ticker=item.get("ticker", ""),
                            name=item.get("name", ""),
                            mentions=item.get("mentions", 0),
                            upvotes=item.get("upvotes", 0),
                            rank_24h_ago=item.get("rank_24h_ago", 0),
                            mentions_24h_ago=item.get("mentions_24h_ago", 0),
                        )
                    )
                except Exception:
                    logger.opt(exception=True).warning(f"Failed to parse ApeWisdom ticker: {item}")

            self._cache = tickers
            self._cache_time = time.monotonic()
            logger.debug(f"ApeWisdom fetched {len(tickers)} trending tickers")
            return tickers

        except httpx.HTTPError as e:
            logger.opt(exception=True).warning(f"ApeWisdom API error: {e}")
            return self._cache  # Return stale cache on error

    def get_ticker(self, symbol: str) -> ApeWisdomTicker | None:
        """Lookup a specific ticker from cached trending list.

        Args:
            symbol: Stock ticker symbol

        Returns:
            ApeWisdomTicker if found in trending, None otherwise
        """
        tickers = self.fetch_trending()
        symbol_upper = symbol.upper()
        for t in tickers:
            if t.ticker.upper() == symbol_upper:
                return t
        return None

    def __repr__(self) -> str:
        """String representation."""
        return f"ApeWisdomFetcher(cached={len(self._cache)}, ttl={self._cache_ttl}s)"
