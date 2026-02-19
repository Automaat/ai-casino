"""News trending watcher - continuous discovery via trending news.

Polls web search for trending stock news, tracks mention frequency,
detects spikes, and routes candidates through EventTriagePipeline.
"""

import asyncio
import re
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, cast

from loguru import logger
from pydantic import BaseModel, Field

from src.daemon.events import BaseEvent, NewsTrendingEvent
from src.watchers.base import PeriodicWatcher
from src.watchers.pipeline import EventTriagePipeline

if TYPE_CHECKING:
    from src.data.websearch import WebSearchFetcher


class NewsTrendingWatcherConfig(BaseModel):
    """Configuration for news trending watcher."""

    poll_interval: int = Field(default=600, ge=300, le=3600)
    trending_window_minutes: int = Field(default=60, ge=30, le=180)
    min_mention_threshold: int = Field(default=3, ge=2, le=10)
    relevance_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    max_candidates_per_cycle: int = Field(default=5, ge=1, le=20)
    search_queries: list[str] = Field(
        default_factory=lambda: [
            "trending stocks today",
            "hot stocks right now",
            "stock market movers",
        ]
    )
    max_results_per_query: int = Field(default=10, ge=5, le=20)


class NewsTrendingWatcher(PeriodicWatcher):
    """Monitor trending stocks in financial news (continuous discovery)."""

    def __init__(
        self,
        pipeline: EventTriagePipeline,
        websearch_fetcher: WebSearchFetcher,
        config: NewsTrendingWatcherConfig,
    ) -> None:
        """Initialize news trending watcher.

        Args:
            pipeline: Event triage pipeline for routing events
            websearch_fetcher: Web search fetcher for news queries
            config: Watcher configuration
        """
        super().__init__(poll_interval=config.poll_interval)
        self._pipeline = pipeline
        self.websearch_fetcher = websearch_fetcher
        self.trending_window_minutes = config.trending_window_minutes
        self.min_mention_threshold = config.min_mention_threshold
        self.max_candidates_per_cycle = config.max_candidates_per_cycle
        self.search_queries = config.search_queries
        self.max_results_per_query = config.max_results_per_query

        self._mention_history: dict[str, list[datetime]] = defaultdict(list)
        self._baselines: dict[str, float] = {}

        self._excluded_words = frozenset(
            {
                "I",
                "A",
                "THE",
                "CEO",
                "CFO",
                "IPO",
                "ETF",
                "GDP",
                "SEC",
                "FBI",
                "USA",
                "NYSE",
                "NASDAQ",
                "WSB",
                "DD",
                "YOLO",
                "FOMO",
                "FUD",
                "BUY",
                "SELL",
                "HOLD",
                "LONG",
                "SHORT",
                "CALL",
                "PUT",
                "ALL",
                "NEW",
                "OLD",
                "BIG",
                "LOW",
                "HIGH",
                "UP",
                "DOWN",
                "OUT",
                "FOR",
                "AND",
                "BUT",
                "NOT",
                "ARE",
                "WAS",
                "HAS",
                "HAD",
            }
        )

        logger.info(
            f"NewsTrendingWatcher initialized: "
            f"poll_interval={self.poll_interval}s, "
            f"trending_window={self.trending_window_minutes}min, "
            f"min_mentions={self.min_mention_threshold}"
        )

    @property
    def name(self) -> str:
        """Watcher display name."""
        return "NewsTrendingWatcher"

    async def _tick(self) -> None:
        """Fetch and process trending news events."""
        events = await self._fetch_events()
        if events:
            await self._pipeline.process(events)

    async def _fetch_events(self) -> list[BaseEvent]:
        """Fetch trending stock mentions from news search.

        Returns:
            List of NewsTrendingEvent for symbols exceeding threshold
        """
        logger.debug("Fetching trending news mentions")

        cycle_mentions: dict[str, list[str]] = defaultdict(list)

        for query in self.search_queries:
            try:
                response = await asyncio.to_thread(
                    self.websearch_fetcher.search_news, query, max_results=self.max_results_per_query
                )

                for result in response.results:
                    text = f"{result.title} {result.body}"
                    symbols = self._extract_tickers(text)

                    for symbol in symbols:
                        cycle_mentions[symbol].append(result.title)

            except Exception as e:
                logger.opt(exception=True).warning(f"News search failed for '{query}': {e}")
                continue

        now = datetime.now(UTC)
        cutoff = now - timedelta(minutes=self.trending_window_minutes)

        for symbol, articles in cycle_mentions.items():
            for _ in articles:
                self._mention_history[symbol].append(now)

            self._mention_history[symbol] = [ts for ts in self._mention_history[symbol] if ts >= cutoff]

        for symbol in list(self._mention_history.keys()):
            if symbol not in cycle_mentions:
                self._mention_history[symbol] = [ts for ts in self._mention_history[symbol] if ts >= cutoff]
                if not self._mention_history[symbol]:
                    del self._mention_history[symbol]

        trending_events: list[NewsTrendingEvent] = []

        for symbol, mention_times in self._mention_history.items():
            mention_count = len(mention_times)

            if mention_count < self.min_mention_threshold:
                continue

            baseline = self._baselines.get(symbol, 1.0)
            spike_ratio = mention_count / baseline

            self._baselines[symbol] = 0.9 * baseline + 0.1 * mention_count

            event = NewsTrendingEvent(
                event_id=f"newstrending_{symbol}_{now.isoformat()}",
                symbol=symbol,
                mention_count=mention_count,
                articles=cycle_mentions.get(symbol, [])[:10],
                baseline_count=baseline,
                spike_ratio=spike_ratio,
                timestamp=now,
                trending_window=self.trending_window_minutes,
            )
            trending_events.append(event)

        symbol_list = [e.symbol for e in trending_events[:5]]
        logger.info(f"Detected {len(trending_events)} trending symbols: {symbol_list}")

        trending_events.sort(key=lambda e: e.spike_ratio, reverse=True)
        return cast("list[BaseEvent]", trending_events[: self.max_candidates_per_cycle])

    def _extract_tickers(self, text: str) -> set[str]:
        """Extract stock tickers from text using regex.

        Args:
            text: Text to extract tickers from

        Returns:
            Set of extracted tickers (uppercase, 1-5 chars)
        """
        tickers = set()
        pattern = r"\$([A-Z]{1,5})\b|\b([A-Z]{2,5})\b"

        for match in re.finditer(pattern, text):
            ticker = match.group(1) or match.group(2)
            if ticker and ticker not in self._excluded_words:
                tickers.add(ticker)

        return tickers

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"NewsTrendingWatcher(poll_interval={self.poll_interval}s, "
            f"trending_window={self.trending_window_minutes}min)"
        )
