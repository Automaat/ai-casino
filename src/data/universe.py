"""Stock universe fetcher for S&P 500 and NASDAQ 100."""

import hashlib
from datetime import datetime
from pathlib import Path

import httpx
from bs4 import BeautifulSoup
from diskcache import Cache
from loguru import logger
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

HTTP_RETRY = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception) & retry_if_not_exception_type(ValueError),
    reraise=True,
    before_sleep=lambda retry_state: logger.warning(
        f"Retry {retry_state.attempt_number} after {retry_state.outcome.exception()}"
    ),
)

UNIVERSE_CACHE_TTL = 604800  # 7 days

SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
NASDAQ100_URL = "https://en.wikipedia.org/wiki/Nasdaq-100"


class StockInfo(BaseModel):
    """Stock information from universe."""

    symbol: str
    name: str
    sector: str
    industry: str


class StockUniverse(BaseModel):
    """Stock universe container."""

    name: str
    stocks: list[StockInfo]
    fetched_at: datetime


class StockUniverseFetcher:
    """Fetch stock lists from Wikipedia with caching."""

    def __init__(self, cache_dir: str | None = None) -> None:
        """Initialize stock universe fetcher.

        Args:
            cache_dir: Cache directory path. Defaults to data/cache/universe/
        """
        self._cache_dir = Path(cache_dir or "data/cache/universe")
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache = Cache(str(self._cache_dir))
        logger.info(f"Initialized StockUniverseFetcher (cache_dir={self._cache_dir})")

    def _cache_key(self, universe_name: str) -> str:
        """Generate cache key from universe name.

        Args:
            universe_name: Name of the universe

        Returns:
            Cache key string
        """
        return hashlib.sha256(universe_name.encode()).hexdigest()[:32]

    @HTTP_RETRY
    def fetch_sp500(self) -> StockUniverse:
        """Fetch S&P 500 stock list.

        Returns:
            StockUniverse with S&P 500 stocks
        """
        logger.info("Fetching S&P 500 universe")

        cache_key = self._cache_key("SP500")
        cached = self._cache.get(cache_key)
        if cached:
            logger.debug("Cache hit for S&P 500")
            return StockUniverse.model_validate(cached)

        stocks = self._scrape_sp500()
        universe = StockUniverse(name="SP500", stocks=stocks, fetched_at=datetime.now())

        self._cache.set(cache_key, universe.model_dump(), expire=UNIVERSE_CACHE_TTL)
        logger.info(f"Fetched {len(stocks)} S&P 500 stocks")
        return universe

    @HTTP_RETRY
    def fetch_nasdaq100(self) -> StockUniverse:
        """Fetch NASDAQ 100 stock list.

        Returns:
            StockUniverse with NASDAQ 100 stocks
        """
        logger.info("Fetching NASDAQ 100 universe")

        cache_key = self._cache_key("NASDAQ100")
        cached = self._cache.get(cache_key)
        if cached:
            logger.debug("Cache hit for NASDAQ 100")
            return StockUniverse.model_validate(cached)

        stocks = self._scrape_nasdaq100()
        universe = StockUniverse(name="NASDAQ100", stocks=stocks, fetched_at=datetime.now())

        self._cache.set(cache_key, universe.model_dump(), expire=UNIVERSE_CACHE_TTL)
        logger.info(f"Fetched {len(stocks)} NASDAQ 100 stocks")
        return universe

    def fetch_combined(self) -> StockUniverse:
        """Fetch combined S&P 500 + NASDAQ 100 (deduplicated).

        Returns:
            StockUniverse with combined stocks
        """
        logger.info("Fetching combined universe")

        cache_key = self._cache_key("COMBINED")
        cached = self._cache.get(cache_key)
        if cached:
            logger.debug("Cache hit for combined universe")
            return StockUniverse.model_validate(cached)

        sp500 = self.fetch_sp500()
        nasdaq100 = self.fetch_nasdaq100()

        seen = set()
        combined_stocks = []
        for stock in sp500.stocks + nasdaq100.stocks:
            if stock.symbol not in seen:
                seen.add(stock.symbol)
                combined_stocks.append(stock)

        universe = StockUniverse(name="COMBINED", stocks=combined_stocks, fetched_at=datetime.now())

        self._cache.set(cache_key, universe.model_dump(), expire=UNIVERSE_CACHE_TTL)
        logger.info(f"Fetched {len(combined_stocks)} combined stocks (deduplicated)")
        return universe

    def _scrape_sp500(self) -> list[StockInfo]:
        """Scrape S&P 500 list from Wikipedia.

        Returns:
            List of StockInfo
        """
        with httpx.Client(timeout=30.0) as client:
            response = client.get(SP500_URL)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            table = soup.find("table", {"id": "constituents"})
            if not table:
                msg = "S&P 500 table not found on Wikipedia"
                raise ValueError(msg)

            stocks = []
            for row in table.find_all("tr")[1:]:  # Skip header
                cols = row.find_all("td")
                if len(cols) >= 4:
                    symbol = cols[0].get_text(strip=True).replace(".", "-")  # BRK.B -> BRK-B
                    name = cols[1].get_text(strip=True)
                    sector = cols[3].get_text(strip=True)
                    industry = cols[4].get_text(strip=True) if len(cols) > 4 else ""

                    stocks.append(StockInfo(symbol=symbol, name=name, sector=sector, industry=industry))

            return stocks

    def _scrape_nasdaq100(self) -> list[StockInfo]:
        """Scrape NASDAQ 100 list from Wikipedia.

        Returns:
            List of StockInfo
        """
        with httpx.Client(timeout=30.0) as client:
            response = client.get(NASDAQ100_URL)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            table = soup.find("table", {"id": "constituents"})
            if not table:
                msg = "NASDAQ 100 table not found on Wikipedia"
                raise ValueError(msg)

            stocks = []
            for row in table.find_all("tr")[1:]:  # Skip header
                cols = row.find_all("td")
                if len(cols) >= 3:
                    name = cols[0].get_text(strip=True)
                    symbol = cols[1].get_text(strip=True).replace(".", "-")
                    sector = cols[2].get_text(strip=True)
                    industry = cols[3].get_text(strip=True) if len(cols) > 3 else ""

                    stocks.append(StockInfo(symbol=symbol, name=name, sector=sector, industry=industry))

            return stocks

    def clear_cache(self) -> None:
        """Clear all cached universes."""
        self._cache.clear()
        logger.info("Cleared universe cache")

    def __repr__(self) -> str:
        """String representation."""
        return f"StockUniverseFetcher(cache_dir={self._cache_dir})"
