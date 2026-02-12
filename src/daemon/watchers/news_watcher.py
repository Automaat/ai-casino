"""News watcher for breaking financial news events.

Polls Marketaux API every 5 minutes, filters by breaking keywords and recency,
deduplicates via URL tracking, and triggers LLM triage + analysis for relevant events.
"""

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import ClassVar, cast

from loguru import logger

from src.cache.historical import HistoricalCache
from src.daemon.event_watcher import EventWatcher, EventWatcherConfig
from src.daemon.events import BaseEvent, NewsEvent
from src.data.base_news_fetcher import BaseNewsFetcher
from src.data.news import NewsFetcher


@dataclass
class NewsWatcherConfig:
    """Configuration for NewsWatcher."""

    poll_interval: int = 300
    relevance_threshold: float = 0.7
    cooldown_minutes: int = 15
    breaking_threshold_minutes: int = 15
    max_concurrent_analyses: int = 2


class NewsWatcher(EventWatcher):
    """Watcher for financial news events.

    Monitors Marketaux for breaking news using keyword detection and recency filtering.
    Maintains rolling window of seen URLs to prevent re-processing.
    """

    # Source weights for deduplication (higher = prefer this source)
    SOURCE_WEIGHTS: ClassVar[dict[str, float]] = {
        "marketaux": 1.0,
        "finnhub": 0.9,
        "newsdata": 0.8,
        "duckduckgo": 0.5,
    }

    # Breaking news keywords (case-insensitive)
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
        historical_cache: HistoricalCache,
        fetchers: list[BaseNewsFetcher] | None = None,
        source_weights: dict[str, float] | None = None,
        config: NewsWatcherConfig | None = None,
        **kwargs: int | float,
    ) -> None:
        """Initialize news watcher.

        Args:
            historical_cache: Shared cache for news data
            fetchers: List of news fetchers (uses Marketaux fallback if not provided)
            source_weights: Custom source weights for deduplication
            config: Configuration (uses defaults if not provided)
            **kwargs: Backward compat params (poll_interval, relevance_threshold, etc.)
        """
        # Backward compat: construct config from kwargs if provided
        if config is None and kwargs:
            defaults = NewsWatcherConfig()
            config = NewsWatcherConfig(
                poll_interval=int(kwargs.get("poll_interval", defaults.poll_interval)),
                relevance_threshold=float(kwargs.get("relevance_threshold", defaults.relevance_threshold)),
                cooldown_minutes=int(kwargs.get("cooldown_minutes", defaults.cooldown_minutes)),
                breaking_threshold_minutes=int(
                    kwargs.get("breaking_threshold_minutes", defaults.breaking_threshold_minutes)
                ),
                max_concurrent_analyses=int(
                    kwargs.get("max_concurrent_analyses", defaults.max_concurrent_analyses)
                ),
            )

        cfg = config or NewsWatcherConfig()
        base_config = EventWatcherConfig(
            poll_interval=cfg.poll_interval,
            relevance_threshold=cfg.relevance_threshold,
            cooldown_minutes=cfg.cooldown_minutes,
            max_concurrent_analyses=cfg.max_concurrent_analyses,
        )
        super().__init__(base_config, historical_cache)
        self.breaking_threshold_minutes = cfg.breaking_threshold_minutes
        self._fetchers = fetchers or []
        self._weights = source_weights or self.SOURCE_WEIGHTS
        self._news_fetcher: NewsFetcher | None = None
        self._seen_urls: dict[str, str] = {}  # url -> source

        source_count = len(self._fetchers) if self._fetchers else "fallback"
        logger.info(
            f"NewsWatcher initialized (breaking_threshold={cfg.breaking_threshold_minutes}m, "
            f"threshold={cfg.relevance_threshold}, sources={source_count})"
        )

    def _init_components(self) -> None:
        """Lazy initialization including news fetcher."""
        super()._init_components()
        if self._news_fetcher is None:
            self._news_fetcher = NewsFetcher(historical_cache=self._historical_cache)

    async def _fetch_events(self) -> list[BaseEvent]:
        """Fetch breaking news from all configured sources.

        Returns:
            List of NewsEvent objects for breaking news
        """
        self._init_components()

        # Parallel fetch from all sources
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
                # result is list[NewsArticle] at this point
                all_articles.extend((art, fetcher.get_source_name()) for art in result)
        else:
            # Fallback: lazy-init single Marketaux fetcher (backward compat)
            if self._news_fetcher is None:
                msg = "No fetchers configured and fallback failed"
                raise RuntimeError(msg)
            articles = await asyncio.to_thread(self._news_fetcher.fetch_market_news, 50)
            all_articles = [(art, "marketaux") for art in articles]

        # Deduplicate by URL (keep highest weight)
        deduplicated = self._deduplicate_by_url(all_articles)

        # Filter breaking news (existing logic)
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

            # Deduplication
            if article.url in self._seen_urls:
                continue

            # Recency check
            age_minutes = (now - article.published_at).total_seconds() / 60
            if age_minutes > self.breaking_threshold_minutes:
                continue

            # Keyword check
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
        """String representation."""
        return (
            f"NewsWatcher(poll_interval={self.poll_interval}s, "
            f"breaking_threshold={self.breaking_threshold_minutes}m)"
        )
