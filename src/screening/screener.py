"""Stock screener engine for finding investment opportunities."""

import hashlib
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import pandas_ta_classic  # noqa: F401 - Required to register .ta accessor
import yfinance as yf
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

from src.data.universe import StockInfo, StockUniverseFetcher
from src.strategies.signal import Signal

if TYPE_CHECKING:
    from src.daemon.config import LiquidityFilterConfig

MIN_MACD_DATA_POINTS = 35
RSI_OVERSOLD_THRESHOLD = 40
PE_VALUE_THRESHOLD = 25
PB_VALUE_THRESHOLD = 3
TRADING_DAYS_3M = 63
MIN_BREAKOUT_DATA_POINTS = 50
BREAKOUT_HIGH_PCT = 5
BREAKOUT_VOLUME_RATIO = 1.5

HTTP_RETRY = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception) & retry_if_not_exception_type(ValueError),
    reraise=True,
    before_sleep=lambda retry_state: logger.warning(
        f"Retry {retry_state.attempt_number} after {retry_state.outcome.exception()}"
    ),
)

SCREENING_CACHE_TTL = 3600  # 1 hour


class ScreeningCriteria(StrEnum):
    """Stock screening criteria."""

    MOMENTUM = "momentum"
    VALUE = "value"
    BREAKOUT = "breakout"


class ScreeningResult(BaseModel):
    """Single stock screening result."""

    symbol: str
    name: str
    sector: str
    score: float
    signal: Signal
    metrics: dict[str, float]
    reason: str


class ScreeningOutput(BaseModel):
    """Stock screening output."""

    criteria: ScreeningCriteria
    universe: str
    results: list[ScreeningResult]
    total_screened: int
    errors: list[str]
    screened_at: datetime


class StockScreener:
    """Screen stocks based on technical criteria."""

    def __init__(
        self,
        universe_fetcher: StockUniverseFetcher,
        liquidity_filters: LiquidityFilterConfig | None = None,
        cache_dir: str | None = None,
    ) -> None:
        """Initialize stock screener.

        Args:
            universe_fetcher: StockUniverseFetcher instance
            liquidity_filters: Liquidity filter configuration for US_LIQUID universe
            cache_dir: Cache directory path. Defaults to data/cache/screening/
        """
        self._universe_fetcher = universe_fetcher
        self._liquidity_filters = liquidity_filters
        self._cache_dir = Path(cache_dir or "data/cache/screening")
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache = Cache(str(self._cache_dir))
        logger.info(f"Initialized StockScreener (cache_dir={self._cache_dir})")

    def _cache_key(self, criteria: ScreeningCriteria, universe: str) -> str:
        """Generate cache key.

        Args:
            criteria: Screening criteria
            universe: Universe name

        Returns:
            Cache key string
        """
        raw = f"{criteria.value}:{universe}"
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    def screen(  # noqa: C901, PLR0912
        self,
        criteria: ScreeningCriteria,
        universe: str = "COMBINED",
        top_n: int = 10,
    ) -> ScreeningOutput:
        """Screen stocks based on criteria.

        Args:
            criteria: Screening criteria (momentum, value, breakout)
            universe: Universe to screen (SP500, NASDAQ100, COMBINED)
            top_n: Number of top results to return

        Returns:
            ScreeningOutput with top results
        """
        logger.info(f"Screening {universe} for {criteria.value} (top {top_n})")

        cache_key = self._cache_key(criteria, universe)
        cached = self._cache.get(cache_key)
        if cached:
            logger.debug("Cache hit for screening")
            output = ScreeningOutput.model_validate(cached)
            output.results = output.results[:top_n]
            return output

        # Fetch universe
        if universe == "SP500":
            stock_universe = self._universe_fetcher.fetch_sp500()
        elif universe == "NASDAQ100":
            stock_universe = self._universe_fetcher.fetch_nasdaq100()
        elif universe == "RUSSELL3000":
            stock_universe = self._universe_fetcher.fetch_russell3000()
        elif universe == "US_LIQUID":
            if self._liquidity_filters is None:
                msg = "US_LIQUID universe requires liquidity_filters config"
                raise ValueError(msg)
            stock_universe = self._universe_fetcher.fetch_us_liquid(self._liquidity_filters)
        else:
            stock_universe = self._universe_fetcher.fetch_combined()

        # Fetch market data in batches (1y for breakout to get 52-week high, 3mo for others)
        symbols = [s.symbol for s in stock_universe.stocks]
        stock_info = {s.symbol: s for s in stock_universe.stocks}
        period = "1y" if criteria == ScreeningCriteria.BREAKOUT else "3mo"
        market_data = self._fetch_batch_data(symbols, period=period)

        # Score each stock
        results = []
        errors = []
        for symbol in symbols:
            if symbol not in market_data:
                errors.append(symbol)
                continue

            try:
                df = market_data[symbol]
                info = stock_info[symbol]

                if criteria == ScreeningCriteria.MOMENTUM:
                    result = self._score_momentum(df, info)
                elif criteria == ScreeningCriteria.VALUE:
                    result = self._score_value(df, info, symbol)
                else:
                    result = self._score_breakout(df, info)

                if result:
                    results.append(result)
            except Exception as e:
                logger.debug(f"Error scoring {symbol}: {e}")
                errors.append(symbol)

        results.sort(key=lambda r: r.score, reverse=True)

        output = ScreeningOutput(
            criteria=criteria,
            universe=universe,
            results=results,
            total_screened=len(symbols),
            errors=errors,
            screened_at=datetime.now(UTC),
        )

        self._cache.set(cache_key, output.model_dump(), expire=SCREENING_CACHE_TTL)
        logger.info(f"Screened {len(symbols)} stocks, found {len(results)} matches, {len(errors)} errors")

        output.results = output.results[:top_n]
        return output

    @HTTP_RETRY
    def _fetch_batch_data(
        self,
        symbols: list[str],
        period: str = "3mo",
        batch_size: int = 50,
    ) -> dict[str, pd.DataFrame]:
        """Fetch market data for multiple symbols.

        Args:
            symbols: List of stock symbols
            period: Data period (e.g., "3mo", "6mo")
            batch_size: Symbols per batch

        Returns:
            Dict mapping symbol to OHLCV DataFrame
        """
        logger.info(f"Fetching data for {len(symbols)} symbols")

        all_data = {}
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i : i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(symbols) + batch_size - 1) // batch_size
            logger.debug(f"Fetching batch {batch_num}/{total_batches}")

            try:
                data = yf.download(batch, period=period, progress=False, group_by="ticker", threads=True)

                if len(batch) == 1:
                    if not data.empty:
                        df = data.dropna()
                        if not df.empty and len(df) >= MIN_MACD_DATA_POINTS:
                            all_data[batch[0]] = df
                else:
                    for sym in batch:
                        if sym in data.columns.get_level_values(0):
                            df = data[sym].dropna()
                            if not df.empty and len(df) >= MIN_MACD_DATA_POINTS:
                                all_data[sym] = df
            except Exception as e:
                logger.warning(f"Batch fetch failed: {e}")

        logger.info(f"Fetched data for {len(all_data)} symbols")
        return all_data

    def _score_momentum(self, df: pd.DataFrame, info: StockInfo) -> ScreeningResult | None:
        """Score stock for momentum criteria.

        Criteria: RSI < 40 (oversold) + MACD histogram positive + price above 50-day MA

        Args:
            df: OHLCV DataFrame
            info: Stock info

        Returns:
            ScreeningResult or None if doesn't match
        """
        df = df.copy()
        df.ta.rsi(length=14, append=True)
        df.ta.macd(fast=12, slow=26, signal=9, append=True)
        df.ta.sma(length=50, append=True)

        rsi_col = next(c for c in df.columns if c.startswith("RSI_14"))
        macd_hist_col = next(c for c in df.columns if c.startswith("MACDh_"))
        sma_col = next(c for c in df.columns if c.startswith("SMA_50"))

        latest = df.iloc[-1]
        rsi = float(latest[rsi_col])
        macd_hist = float(latest[macd_hist_col])
        close = float(latest["Close"])
        sma50 = float(latest[sma_col])

        # Momentum criteria: RSI < threshold, MACD hist positive, price > SMA50
        if rsi >= RSI_OVERSOLD_THRESHOLD or macd_hist <= 0 or close <= sma50:
            return None

        # Score: weight RSI distance from threshold, MACD strength, price above MA
        rsi_score = (RSI_OVERSOLD_THRESHOLD - rsi) / RSI_OVERSOLD_THRESHOLD  # 0-1, lower RSI = higher
        macd_score = min(macd_hist / abs(close) * 100, 1.0)  # Normalize
        ma_score = min((close - sma50) / sma50 * 10, 1.0)  # % above MA

        score = rsi_score * 0.4 + macd_score * 0.3 + ma_score * 0.3

        return ScreeningResult(
            symbol=info.symbol,
            name=info.name,
            sector=info.sector,
            score=round(score, 4),
            signal=Signal.BUY,
            metrics={
                "rsi": round(rsi, 2),
                "macd_hist": round(macd_hist, 4),
                "close": round(close, 2),
                "sma50": round(sma50, 2),
            },
            reason=f"RSI {rsi:.1f} (oversold), MACD bullish ({macd_hist:.4f}), above 50-day MA",
        )

    def _score_value(self, df: pd.DataFrame, info: StockInfo, symbol: str) -> ScreeningResult | None:
        """Score stock for value criteria.

        Criteria: Low P/E vs market + P/B < 3 + positive price momentum

        Args:
            df: OHLCV DataFrame
            info: Stock info
            symbol: Stock symbol

        Returns:
            ScreeningResult or None if doesn't match
        """
        try:
            ticker = yf.Ticker(symbol)
            ticker_info = ticker.info
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to fetch yfinance info for {symbol}: {e}")
            return None

        pe = ticker_info.get("trailingPE")
        pb = ticker_info.get("priceToBook")
        forward_pe = ticker_info.get("forwardPE")

        # Require positive P/E and P/B (not None, not 0, not negative)
        if not pe or not pb or pe <= 0 or pb <= 0:
            return None

        # Value criteria: P/E < threshold, P/B < threshold
        if pe >= PE_VALUE_THRESHOLD or pb >= PB_VALUE_THRESHOLD:
            return None

        # Calculate 3-month return
        if len(df) >= TRADING_DAYS_3M:
            start_price = float(df["Close"].iloc[-TRADING_DAYS_3M])
            end_price = float(df["Close"].iloc[-1])
            return_3m = (end_price - start_price) / start_price * 100
        else:
            return_3m = 0

        # Score: lower P/E and P/B = better, positive momentum helps
        pe_score = min(max(0.0, (PE_VALUE_THRESHOLD - pe) / PE_VALUE_THRESHOLD), 1.0)
        pb_score = min(max(0.0, (PB_VALUE_THRESHOLD - pb) / PB_VALUE_THRESHOLD), 1.0)
        momentum_score = min(max(return_3m / 20, 0), 1.0)  # Cap at 20% return

        score = pe_score * 0.4 + pb_score * 0.4 + momentum_score * 0.2

        return ScreeningResult(
            symbol=info.symbol,
            name=info.name,
            sector=info.sector,
            score=round(score, 4),
            signal=Signal.BUY,
            metrics={
                "pe_ratio": round(pe, 2),
                "pb_ratio": round(pb, 2),
                "forward_pe": round(forward_pe, 2) if forward_pe else 0,
                "return_3m": round(return_3m, 2),
            },
            reason=f"P/E {pe:.1f}, P/B {pb:.2f}, 3M return {return_3m:.1f}%",
        )

    def _score_breakout(self, df: pd.DataFrame, info: StockInfo) -> ScreeningResult | None:
        """Score stock for breakout criteria.

        Criteria: Price within 5% of 52-week high + volume > 1.5x average

        Args:
            df: OHLCV DataFrame
            info: Stock info

        Returns:
            ScreeningResult or None if doesn't match
        """
        if len(df) < MIN_BREAKOUT_DATA_POINTS:
            return None

        close = float(df["Close"].iloc[-1])
        high_52w = float(df["High"].max())
        volume = float(df["Volume"].iloc[-1])
        avg_volume = float(df["Volume"].rolling(20).mean().iloc[-1])

        # Breakout criteria: within 5% of 52-week high, volume > 1.5x average
        pct_from_high = (high_52w - close) / high_52w * 100
        volume_ratio = volume / avg_volume if avg_volume > 0 else 0

        if pct_from_high > BREAKOUT_HIGH_PCT or volume_ratio < BREAKOUT_VOLUME_RATIO:
            return None

        # Score: closer to high = better, higher volume = better
        high_score = (BREAKOUT_HIGH_PCT - pct_from_high) / BREAKOUT_HIGH_PCT
        volume_score = min((volume_ratio - 1) / 2, 1)

        score = high_score * 0.6 + volume_score * 0.4

        return ScreeningResult(
            symbol=info.symbol,
            name=info.name,
            sector=info.sector,
            score=round(score, 4),
            signal=Signal.BUY,
            metrics={
                "close": round(close, 2),
                "high_52w": round(high_52w, 2),
                "pct_from_high": round(pct_from_high, 2),
                "volume_ratio": round(volume_ratio, 2),
            },
            reason=f"{pct_from_high:.1f}% from 52-week high, volume {volume_ratio:.1f}x average",
        )

    def clear_cache(self) -> None:
        """Clear screening cache."""
        self._cache.clear()
        logger.info("Cleared screening cache")

    def __repr__(self) -> str:
        """String representation."""
        return f"StockScreener(cache_dir={self._cache_dir})"
