"""News watcher for breaking financial news events.

Polls configured sources, filters by breaking keywords and recency,
deduplicates via URL tracking, and routes events through EventTriagePipeline.
"""

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import ClassVar, cast

from loguru import logger

from src.cache.historical import HistoricalCache
from src.daemon.events import BaseEvent, NewsEvent
from src.data.base_news_fetcher import BaseNewsFetcher
from src.data.news import NewsFetcher
from src.v1.watchers.base import PeriodicWatcher
from src.v1.watchers.pipeline import EventTriagePipeline


@dataclass
class NewsWatcherConfig:
    """Configuration for NewsWatcher."""

    poll_interval: int = 300
    breaking_threshold_minutes: int = 15


class NewsWatcher(PeriodicWatcher):
    """Watcher for financial news events.

    Monitors news sources for breaking news using keyword detection and recency filtering.
    Maintains rolling window of seen URLs to prevent re-processing.
    """

    SOURCE_WEIGHTS: ClassVar[dict[str, float]] = {
        "marketaux": 1.0,
        "finnhub": 0.9,
        "newsdata": 0.8,
        "duckduckgo": 0.5,
    }

    BREAKING_KEYWORDS: ClassVar[frozenset[str]] = frozenset(
        {
            "breaking",
            "announces",
            "reports earnings",
            "earnings beat",
            "earnings miss",
            "guidance",
            "fda approval",
            "fda rejects",
            "merger",
            "acquisition",
            "lawsuit",
            "recall",
            "bankruptcy",
            "ceo",
            "executive",
            "halted",
            "investigation",
            "subpoena",
            "settles",
            "settlement",
            "partnership",
            "deal",
            "contract",
            "layoffs",
            "restructuring",
        }
    )

    def __init__(
        self,
        pipeline: EventTriagePipeline,
        historical_cache: HistoricalCache,
        fetchers: list[BaseNewsFetcher] | None = None,
        config: NewsWatcherConfig | None = None,
    ) -> None:
        """Initialize news watcher.

        Args:
            pipeline: Event triage pipeline for routing events
            historical_cache: Cache for fallback NewsFetcher initialization
            fetchers: News data sources (uses Marketaux fallback if not provided)
            config: Watcher configuration
        """
        cfg = config or NewsWatcherConfig()
        super().__init__(poll_interval=cfg.poll_interval)
        self._pipeline = pipeline
        self._historical_cache = historical_cache
        self.breaking_threshold_minutes = cfg.breaking_threshold_minutes
        self._fetchers = fetchers or []
        self._weights = self.SOURCE_WEIGHTS
        self._news_fetcher: NewsFetcher | None = None
        self._seen_urls: dict[str, str] = {}

        source_count = len(self._fetchers) if self._fetchers else "fallback"
        logger.info(
            f"NewsWatcher initialized (breaking_threshold={cfg.breaking_threshold_minutes}m, "
            f"sources={source_count})"
        )

    @property
    def name(self) -> str:
        """Watcher display name."""
        return "NewsWatcher"

    async def _tick(self) -> None:
        """Fetch and process news events."""
        events = await self._fetch_events()
        if events:
            await self._pipeline.process(events)

    async def _fetch_events(self) -> list[BaseEvent]:
        """Fetch breaking news from all configured sources.

        Returns:
            List of NewsEvent objects for breaking news
        """
        if self._fetchers:
            tasks = [fetcher.afetch_market_news(limit=50) for fetcher in self._fetchers]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            all_articles: list[tuple[object, str]] = []
            for fetcher, result in zip(self._fetchers, results, strict=True):
                if isinstance(result, (asyncio.CancelledError, KeyboardInterrupt)):
                    raise result
                if isinstance(result, BaseException):
                    logger.warning(f"{fetcher.get_source_name()} fetch failed: {result}")
                    continue
                all_articles.extend((art, fetcher.get_source_name()) for art in result)
        else:
            if self._news_fetcher is None:
                self._news_fetcher = NewsFetcher(historical_cache=self._historical_cache)
            articles = await self._news_fetcher.afetch_market_news(50)
            all_articles = [(art, "marketaux") for art in articles]

        deduplicated = self._deduplicate_by_url(all_articles)
        return self._filter_breaking(deduplicated)

    def _deduplicate_by_url(
        self,
        articles: list[tuple[object, str]],
    ) -> list[tuple[object, str]]:
        """Keep article from highest-weighted source per URL.

        Args:
            articles: List of (article, source) tuples

        Returns:
            Deduplicated list of (article, source) tuples
        """
        from src.data.news import NewsArticle

        url_map: dict[str, tuple[NewsArticle, str, float]] = {}

        for article, source in articles:
            if not isinstance(article, NewsArticle):
                continue

            weight = self._weights.get(source, 0.0)

            if article.url not in url_map:
                url_map[article.url] = (article, source, weight)
            else:
                _, existing_source, existing_weight = url_map[article.url]
                if weight > existing_weight:
                    url_map[article.url] = (article, source, weight)
                    logger.debug(f"Dedup: prefer {source} over {existing_source} for {article.url}")

        return [(art, src) for art, src, _ in url_map.values()]

    def _filter_breaking(
        self,
        articles: list[tuple[object, str]],
    ) -> list[BaseEvent]:
        """Filter breaking news from deduplicated articles.

        Args:
            articles: List of (article, source) tuples

        Returns:
            List of NewsEvent objects for breaking news
        """
        from src.data.news import NewsArticle

        breaking: list[BaseEvent] = []
        now = datetime.now(UTC)

        for article, source in articles:
            if not isinstance(article, NewsArticle):
                continue

            if article.url in self._seen_urls:
                continue

            age_minutes = (now - article.published_at).total_seconds() / 60
            if age_minutes > self.breaking_threshold_minutes:
                continue

            title_lower = article.title.lower()
            description_lower = article.description.lower()
            combined_text = f"{title_lower} {description_lower}"

            if any(kw in combined_text for kw in self.BREAKING_KEYWORDS):
                breaking.append(
                    cast(
                        "BaseEvent",
                        NewsEvent(
                            event_id=article.url,
                            event_type="news",
                            timestamp=article.published_at,
                            source=source,
                            article=article,
                        ),
                    )
                )
                self._seen_urls[article.url] = source
                logger.info(f"Breaking from {source}: {article.title[:60]}... ({age_minutes:.1f}m)")

        return breaking

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"NewsWatcher(poll_interval={self.poll_interval}s, "
            f"breaking_threshold={self.breaking_threshold_minutes}m)"
        )
