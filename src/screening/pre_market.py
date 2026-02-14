"""Pre-market screening for gap plays and catalysts."""

import asyncio
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf
from loguru import logger

from src.data.earnings import EarningsCalendarFetcher
from src.data.news import NewsFetcher
from src.data.universe import StockUniverseFetcher
from src.screening.models.pre_market import PreMarketCandidate, PreMarketResult

EASTERN = ZoneInfo("America/New_York")


class PreMarketScreener:
    """Pre-market screening for gap plays, volume spikes, and catalysts."""

    def __init__(
        self,
        universe_fetcher: StockUniverseFetcher,
        news_fetcher: NewsFetcher,
        earnings_fetcher: EarningsCalendarFetcher,
    ) -> None:
        """Initialize pre-market screener.

        Args:
            universe_fetcher: Stock universe fetcher for symbol lists
            news_fetcher: News fetcher for overnight catalyst detection
            earnings_fetcher: Earnings calendar fetcher for upcoming dates
        """
        self._universe_fetcher = universe_fetcher
        self._news_fetcher = news_fetcher
        self._earnings_fetcher = earnings_fetcher
        logger.info("Initialized PreMarketScreener")

    async def screen(
        self,
        universe: str = "NASDAQ100",
        top_n: int = 20,
        gap_threshold: float = 3.0,
        min_volume_ratio: float = 1.5,
        min_score: float = 0.60,
        timeout_seconds: int = 60,
        earnings_lookahead_days: int = 7,
        overnight_news_hours: int = 14,
        gap_weight: float = 0.50,
        volume_weight: float = 0.30,
        catalyst_weight: float = 0.20,
    ) -> PreMarketResult:
        """Screen universe for pre-market opportunities.

        Args:
            universe: Universe name (NASDAQ100, SP500, COMBINED)
            top_n: Top N candidates to return
            gap_threshold: Minimum gap percentage (absolute)
            min_volume_ratio: Minimum volume ratio vs 20-day average
            min_score: Minimum composite score
            timeout_seconds: Timeout for data fetching
            earnings_lookahead_days: Days ahead to check for earnings
            overnight_news_hours: Hours back to check for overnight news
            gap_weight: Gap score weight (0-1)
            volume_weight: Volume score weight (0-1)
            catalyst_weight: Catalyst score weight (0-1)

        Returns:
            PreMarketResult with filtered and ranked candidates
        """
        logger.info(f"Starting pre-market screening: universe={universe}, top_n={top_n}")

        symbols = await asyncio.to_thread(self._get_universe_symbols, universe)
        logger.info(f"Fetched {len(symbols)} symbols from {universe}")

        try:
            async with asyncio.timeout(timeout_seconds):
                market_data = await asyncio.to_thread(self._fetch_market_data, symbols)
                news_data = await self._fetch_overnight_news(symbols, overnight_news_hours)
                earnings_data = await asyncio.to_thread(
                    self._fetch_earnings_data, symbols, earnings_lookahead_days
                )
        except TimeoutError:
            logger.warning(f"Pre-market screening timeout after {timeout_seconds}s, using partial data")
            market_data = {}
            news_data = {}
            earnings_data = {}

        candidates = self._build_candidates(
            market_data=market_data,
            news_data=news_data,
            earnings_data=earnings_data,
            gap_weight=gap_weight,
            volume_weight=volume_weight,
            catalyst_weight=catalyst_weight,
        )

        filtered = self._filter_candidates(
            candidates=candidates,
            gap_threshold=gap_threshold,
            min_volume_ratio=min_volume_ratio,
            min_score=min_score,
        )

        ranked = self._rank_candidates(filtered, top_n=top_n)

        now = datetime.now(EASTERN)
        expires_at = now.replace(hour=9, minute=30, second=0, microsecond=0)
        if now >= expires_at:
            expires_at += timedelta(days=1)

        result = PreMarketResult(
            candidates=ranked,
            total_screened=len(symbols),
            filtered_count=len(filtered),
            screened_at=datetime.now(UTC),
            expires_at=expires_at,
            gap_plays_count=sum(1 for c in ranked if abs(c.gap_percent) >= gap_threshold),
            volume_spike_count=sum(1 for c in ranked if c.volume_ratio >= 2.0),
            catalyst_count=sum(1 for c in ranked if c.has_earnings or c.news_count > 0),
        )

        logger.info(
            f"Pre-market screening complete: {len(ranked)}/{len(filtered)} candidates "
            f"(gap={result.gap_plays_count}, vol={result.volume_spike_count}, "
            f"catalyst={result.catalyst_count})"
        )

        return result

    def _get_universe_symbols(self, universe: str) -> list[str]:
        """Get symbols from universe."""
        if universe == "NASDAQ100":
            stocks = self._universe_fetcher.fetch_nasdaq100().stocks
        elif universe == "SP500":
            stocks = self._universe_fetcher.fetch_sp500().stocks
        elif universe == "COMBINED":
            stocks = self._universe_fetcher.fetch_combined().stocks
        else:
            logger.warning(f"Unknown universe {universe}, falling back to NASDAQ100")
            stocks = self._universe_fetcher.fetch_nasdaq100().stocks

        return [s.symbol for s in stocks]

    def _fetch_market_data(self, symbols: list[str]) -> dict[str, dict]:
        """Fetch market data for gap detection.

        Returns dict: symbol -> {prev_close, open, yesterday_volume, avg_volume_20d, name, sector}
        """
        logger.info(f"Fetching market data for {len(symbols)} symbols")

        data = yf.download(
            tickers=symbols,
            period="1mo",
            interval="1d",
            progress=False,
            group_by="ticker",
            threads=True,
        )

        result = {}

        if len(symbols) == 1:
            symbol = symbols[0]
            if isinstance(data, pd.DataFrame) and not data.empty and len(data) >= 2:
                close_data = data.get("Close", pd.Series(dtype=float))
                open_data = data.get("Open", pd.Series(dtype=float))
                volume_data = data.get("Volume", pd.Series(dtype=float))

                if not close_data.empty and not open_data.empty and not volume_data.empty:
                    prev_close = float(close_data.iloc[-2])
                    current_open = float(open_data.iloc[-1])
                    yesterday_volume = int(volume_data.iloc[-2])
                    avg_volume_20d = float(volume_data.iloc[-20:].mean()) if len(volume_data) >= 20 else 0.0

                    try:
                        ticker = yf.Ticker(symbol)
                        info = ticker.info or {}
                        name = info.get("longName", symbol)
                        sector = info.get("sector", "Unknown")
                    except Exception:
                        name = symbol
                        sector = "Unknown"

                    result[symbol] = {
                        "prev_close": prev_close,
                        "open": current_open,
                        "yesterday_volume": yesterday_volume,
                        "avg_volume_20d": avg_volume_20d,
                        "name": name,
                        "sector": sector,
                    }
        else:
            for symbol in symbols:
                try:
                    if symbol not in data or data[symbol].empty or len(data[symbol]) < 2:
                        continue

                    symbol_data = data[symbol]
                    close_data = symbol_data.get("Close", pd.Series(dtype=float))
                    open_data = symbol_data.get("Open", pd.Series(dtype=float))
                    volume_data = symbol_data.get("Volume", pd.Series(dtype=float))

                    if close_data.empty or open_data.empty or volume_data.empty:
                        continue

                    prev_close = float(close_data.iloc[-2])
                    current_open = float(open_data.iloc[-1])
                    yesterday_volume = int(volume_data.iloc[-2])
                    avg_volume_20d = float(volume_data.iloc[-20:].mean()) if len(volume_data) >= 20 else 0.0

                    try:
                        ticker = yf.Ticker(symbol)
                        info = ticker.info or {}
                        name = info.get("longName", symbol)
                        sector = info.get("sector", "Unknown")
                    except Exception:
                        name = symbol
                        sector = "Unknown"

                    result[symbol] = {
                        "prev_close": prev_close,
                        "open": current_open,
                        "yesterday_volume": yesterday_volume,
                        "avg_volume_20d": avg_volume_20d,
                        "name": name,
                        "sector": sector,
                    }
                except Exception as e:
                    logger.opt(exception=True).debug(f"Failed to fetch market data for {symbol}: {e}")
                    continue

        logger.info(f"Successfully fetched market data for {len(result)}/{len(symbols)} symbols")
        return result

    async def _fetch_overnight_news(self, symbols: list[str], hours_back: int) -> dict[str, list[str]]:
        """Fetch overnight news for symbols.

        Returns dict: symbol -> [news_titles]
        """
        logger.info(f"Fetching overnight news for {len(symbols)} symbols")

        since = datetime.now(UTC) - timedelta(hours=hours_back)
        result = {}

        for symbol in symbols:
            try:
                articles = await self._news_fetcher.afetch_company_news(symbol=symbol, limit=5)
                overnight = [a.title for a in articles if a.published_at >= since]
                if overnight:
                    result[symbol] = overnight
            except Exception as e:
                logger.opt(exception=True).debug(f"Failed to fetch news for {symbol}: {e}")
                continue

        logger.info(f"Found overnight news for {len(result)}/{len(symbols)} symbols")
        return result

    def _fetch_earnings_data(self, symbols: list[str], lookahead_days: int) -> dict[str, datetime]:
        """Fetch upcoming earnings dates.

        Returns dict: symbol -> earnings_date
        """
        logger.info(f"Fetching earnings data for {len(symbols)} symbols")

        calendar = self._earnings_fetcher.fetch_earnings_dates(symbols)
        cutoff = datetime.now(UTC).date() + timedelta(days=lookahead_days)

        result = {}
        for event in calendar.events:
            if event.earnings_date <= cutoff:
                result[event.symbol] = datetime.combine(event.earnings_date, datetime.min.time())

        logger.info(f"Found upcoming earnings for {len(result)}/{len(symbols)} symbols")
        return result

    def _build_candidates(
        self,
        market_data: dict[str, dict],
        news_data: dict[str, list[str]],
        earnings_data: dict[str, datetime],
        gap_weight: float,
        volume_weight: float,
        catalyst_weight: float,
    ) -> list[PreMarketCandidate]:
        """Build candidates from fetched data."""
        candidates = []

        for symbol, data in market_data.items():
            prev_close = data["prev_close"]
            current_open = data["open"]
            yesterday_volume = data["yesterday_volume"]
            avg_volume_20d = data["avg_volume_20d"]

            if prev_close == 0 or avg_volume_20d == 0:
                continue

            gap_percent = ((current_open - prev_close) / prev_close) * 100
            volume_ratio = yesterday_volume / avg_volume_20d

            news_titles = news_data.get(symbol, [])
            earnings_date = earnings_data.get(symbol)
            has_earnings = earnings_date is not None

            gap_score = min(abs(gap_percent) / 10.0, 1.0)
            volume_score = min((volume_ratio - 1.0) / 3.0, 1.0) if volume_ratio >= 1.0 else 0.0
            catalyst_score = (0.5 if has_earnings else 0.0) + min(len(news_titles) / 5.0, 0.5)

            composite_score = (
                gap_weight * gap_score + volume_weight * volume_score + catalyst_weight * catalyst_score
            )

            candidate = PreMarketCandidate(
                symbol=symbol,
                name=data["name"],
                sector=data["sector"],
                prev_close=prev_close,
                current_open=current_open,
                gap_percent=gap_percent,
                yesterday_volume=yesterday_volume,
                avg_volume_20d=avg_volume_20d,
                volume_ratio=volume_ratio,
                has_earnings=has_earnings,
                earnings_date=earnings_date,
                news_count=len(news_titles),
                news_titles=news_titles,
                gap_score=gap_score,
                volume_score=volume_score,
                catalyst_score=catalyst_score,
                composite_score=composite_score,
                priority=1,
            )

            candidates.append(candidate)

        return candidates

    def _filter_candidates(
        self,
        candidates: list[PreMarketCandidate],
        gap_threshold: float,
        min_volume_ratio: float,
        min_score: float,
    ) -> list[PreMarketCandidate]:
        """Filter candidates by thresholds."""
        filtered = [
            c
            for c in candidates
            if abs(c.gap_percent) >= gap_threshold
            and c.volume_ratio >= min_volume_ratio
            and c.composite_score >= min_score
        ]

        logger.info(f"Filtered {len(filtered)}/{len(candidates)} candidates")
        return filtered

    def _rank_candidates(self, candidates: list[PreMarketCandidate], top_n: int) -> list[PreMarketCandidate]:
        """Rank candidates and assign priorities."""
        ranked = sorted(candidates, key=lambda c: c.composite_score, reverse=True)[:top_n]

        for i, candidate in enumerate(ranked):
            candidate.priority = min((i // 4) + 1, 5)

        return ranked
