"""Stock universe fetcher for S&P 500 and NASDAQ 100."""

import csv
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import yfinance as yf
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

if TYPE_CHECKING:
    from src.daemon.config import LiquidityFilterConfig

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
ISHARES_RUSSELL3000_URL = "https://www.ishares.com/us/products/239714/ishares-russell-3000-etf/1467271812596.ajax?fileType=csv&fileName=IWV_holdings&dataType=fund"

# User-Agent to avoid 403 from Wikipedia/iShares
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


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

    @HTTP_RETRY
    def fetch_russell3000(self) -> StockUniverse:
        """Fetch Russell 3000 stock list (unfiltered).

        Returns:
            StockUniverse with Russell 3000 stocks
        """
        logger.info("Fetching Russell 3000 universe")

        cache_key = self._cache_key("RUSSELL3000")
        cached = self._cache.get(cache_key)
        if cached:
            logger.debug("Cache hit for Russell 3000")
            return StockUniverse.model_validate(cached)

        stocks = self._scrape_ishares_russell3000()
        universe = StockUniverse(name="RUSSELL3000", stocks=stocks, fetched_at=datetime.now())

        self._cache.set(cache_key, universe.model_dump(), expire=UNIVERSE_CACHE_TTL)
        logger.info(f"Fetched {len(stocks)} Russell 3000 stocks")
        return universe

    def _fetch_batch_metadata(  # noqa: C901, PLR0912
        self,
        symbols: list[str],
        batch_size: int = 50,
    ) -> dict[str, dict[str, float]]:
        """Fetch price, volume, market cap for symbols in batches.

        Args:
            symbols: List of stock symbols
            batch_size: Number of symbols per batch

        Returns:
            Dict mapping symbol to {price, avg_volume, market_cap}
        """
        logger.info(f"Fetching metadata for {len(symbols)} symbols")

        # Validate symbols before sending to yfinance
        invalid_symbols = [s for s in symbols if " " in s or len(s) > 6]
        if invalid_symbols:
            logger.error(f"INVALID SYMBOLS (spaces/too long): {invalid_symbols[:10]}")

        all_data: dict[str, dict[str, float]] = {}

        # Stage 1: Batch fetch OHLCV data for price and volume
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i : i + batch_size]
            logger.debug(f"Batch symbols (first 5): {batch[:5]}")

            try:
                batch_num = i // batch_size + 1
                total_batches = (len(symbols) + batch_size - 1) // batch_size
                logger.debug(f"Fetching OHLCV batch {batch_num}/{total_batches}")
                data = yf.download(batch, period="3mo", progress=False, group_by="ticker", threads=True)

                if data.empty:
                    continue

                # Handle single symbol vs multiple symbols response structure
                if len(batch) == 1:
                    symbol = batch[0]
                    if not data.empty and "Close" in data.columns:
                        price = float(data["Close"].iloc[-1])
                        avg_volume = float(data["Volume"].tail(30).mean())
                        all_data[symbol] = {"price": price, "avg_volume": avg_volume, "market_cap": 0.0}
                else:
                    for symbol in batch:
                        try:
                            if symbol not in data.columns.get_level_values(0):
                                continue
                            symbol_data = data[symbol]
                            if symbol_data.empty or "Close" not in symbol_data.columns:
                                continue
                            price = float(symbol_data["Close"].iloc[-1])
                            avg_volume = float(symbol_data["Volume"].tail(30).mean())
                            all_data[symbol] = {"price": price, "avg_volume": avg_volume, "market_cap": 0.0}
                        except Exception as e:
                            logger.debug(f"Failed to parse {symbol}: {e}")
                            continue

            except Exception as e:
                logger.opt(exception=True).warning(f"Failed to fetch OHLCV batch: {e}")
                continue

        logger.info(f"Stage 1: Fetched OHLCV for {len(all_data)} symbols")

        # Stage 2: Fetch market cap in parallel for survivors
        def fetch_market_cap(symbol: str) -> tuple[str, float]:
            try:
                ticker = yf.Ticker(symbol)
                market_cap = ticker.info.get("marketCap", 0)
                return symbol, float(market_cap) if market_cap else 0.0
            except Exception as e:
                logger.debug(f"Failed to fetch market cap for {symbol}: {e}")
                return symbol, 0.0

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(fetch_market_cap, sym): sym for sym in all_data}
            for completed, future in enumerate(as_completed(futures), start=1):
                if completed % 100 == 0:
                    logger.debug(f"Stage 2: Fetched market cap {completed}/{len(all_data)}")
                try:
                    symbol, market_cap = future.result()
                    all_data[symbol]["market_cap"] = market_cap
                except Exception as e:
                    logger.debug(f"Failed to get market cap result: {e}")

        logger.info(f"Stage 2: Fetched market cap for {len(all_data)} symbols")
        return all_data

    @HTTP_RETRY
    def fetch_us_liquid(self, filters: LiquidityFilterConfig) -> StockUniverse:
        """Fetch US liquid stocks (Russell 3000 filtered by liquidity).

        Args:
            filters: Liquidity filter configuration

        Returns:
            StockUniverse with filtered stocks (~500-1500 depending on filters)
        """
        # Import here to avoid circular dependency
        from src.daemon.config import LiquidityFilterConfig  # noqa: F401

        # Generate cache key from filter config hash
        filter_hash = hashlib.sha256(
            f"{filters.min_market_cap}_{filters.min_avg_volume}_{filters.price_range}".encode()
        ).hexdigest()[:16]
        cache_key = self._cache_key(f"US_LIQUID_{filter_hash}")

        # Check cache
        cached = self._cache.get(cache_key)
        if cached:
            logger.debug("Cache hit for US_LIQUID")
            return StockUniverse.model_validate(cached)

        logger.info("Fetching US_LIQUID universe (filtered Russell 3000)")

        # Fetch unfiltered Russell 3000
        russell3000 = self.fetch_russell3000()
        symbols = [s.symbol for s in russell3000.stocks]

        # Fetch metadata and filter in two stages
        metadata = self._fetch_batch_metadata(symbols)

        # Apply filters
        filtered_stocks = []
        for stock in russell3000.stocks:
            if stock.symbol not in metadata:
                continue

            meta = metadata[stock.symbol]

            # Price range filter
            price = meta["price"]
            min_price, max_price = filters.price_range
            if not (min_price <= price <= max_price):
                continue

            # Volume filter
            if meta["avg_volume"] < filters.min_avg_volume:
                continue

            # Market cap filter
            if meta["market_cap"] < filters.min_market_cap:
                continue

            filtered_stocks.append(stock)

        universe = StockUniverse(
            name="US_LIQUID",
            stocks=filtered_stocks,
            fetched_at=datetime.now(),
        )

        # Cache for 7 days
        self._cache.set(cache_key, universe.model_dump(), expire=UNIVERSE_CACHE_TTL)
        logger.info(
            f"Filtered to {len(filtered_stocks)} liquid stocks "
            f"from {len(russell3000.stocks)} Russell 3000 stocks"
        )

        return universe

    def _scrape_sp500(self) -> list[StockInfo]:
        """Scrape S&P 500 list from Wikipedia.

        Returns:
            List of StockInfo
        """
        headers = {"User-Agent": USER_AGENT}
        with httpx.Client(timeout=30.0, headers=headers) as client:
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
                    raw_symbol = cols[0].get_text(strip=True)
                    raw_name = cols[1].get_text(strip=True)
                    symbol = raw_symbol.replace(".", "-")  # BRK.B -> BRK-B
                    name = raw_name
                    sector = cols[3].get_text(strip=True)
                    industry = cols[4].get_text(strip=True) if len(cols) > 4 else ""

                    # Validate: symbol should not contain spaces and should be short
                    if " " in symbol or len(symbol) > 6:
                        logger.warning(
                            f"S&P500 INVALID: col0='{raw_symbol}' col1='{raw_name}' -> "
                            f"symbol='{symbol}' name='{name}'"
                        )
                        continue

                    stocks.append(StockInfo(symbol=symbol, name=name, sector=sector, industry=industry))

            return stocks

    def _scrape_nasdaq100(self) -> list[StockInfo]:
        """Scrape NASDAQ 100 list from Wikipedia.

        Returns:
            List of StockInfo
        """
        headers = {"User-Agent": USER_AGENT}
        with httpx.Client(timeout=30.0, headers=headers) as client:
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
                    raw_name = cols[0].get_text(strip=True)
                    raw_symbol = cols[1].get_text(strip=True)
                    name = raw_name
                    symbol = raw_symbol.replace(".", "-")
                    sector = cols[2].get_text(strip=True)
                    industry = cols[3].get_text(strip=True) if len(cols) > 3 else ""

                    # Validate: symbol should not contain spaces and should be short
                    if " " in symbol or len(symbol) > 6:
                        logger.warning(
                            f"NASDAQ100 INVALID: col0='{raw_name}' col1='{raw_symbol}' -> "
                            f"symbol='{symbol}' name='{name}'"
                        )
                        continue

                    stocks.append(StockInfo(symbol=symbol, name=name, sector=sector, industry=industry))

            return stocks

    def _scrape_ishares_russell3000(self) -> list[StockInfo]:
        """Scrape Russell 3000 list from iShares IWV ETF holdings CSV.

        Returns:
            List of StockInfo
        """
        headers = {"User-Agent": USER_AGENT}
        with httpx.Client(timeout=30.0, headers=headers) as client:
            response = client.get(ISHARES_RUSSELL3000_URL)
            response.raise_for_status()

            # Parse CSV (skip first 10 rows which are metadata)
            lines = response.text.strip().split("\n")[10:]
            stocks = []

            # Use csv.reader to properly handle quoted fields with commas
            reader = csv.reader(lines)
            for parts in reader:
                if len(parts) < 4:
                    continue

                # Columns: Ticker, Name, Sector, Asset Class, Market Value, Weight, ...
                ticker = parts[0].strip()
                name = parts[1].strip()
                sector = parts[2].strip()

                # Skip header row and non-equity entries
                if ticker in ("Ticker", "-", "") or sector == "-":
                    continue

                # Handle BRK.B -> BRK-B conversion
                symbol = ticker.replace(".", "-")

                # Validate: symbol should not contain spaces and should be short
                if " " in symbol or len(symbol) > 6:
                    logger.warning(f"Invalid symbol '{symbol}' (name: {name}), skipping")
                    continue

                stocks.append(
                    StockInfo(
                        symbol=symbol,
                        name=name,
                        sector=sector or "Unknown",
                        industry="",  # Not available in iShares CSV
                    )
                )

            return stocks

    def clear_cache(self) -> None:
        """Clear all cached universes."""
        self._cache.clear()
        logger.info("Cleared universe cache")

    def __repr__(self) -> str:
        """String representation."""
        return f"StockUniverseFetcher(cache_dir={self._cache_dir})"
