"""News watcher for breaking financial news events.

Polls Marketaux API every 5 minutes, filters by breaking keywords and recency,
deduplicates via URL tracking, and triggers LLM triage + analysis for relevant events.
"""

from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import ClassVar, cast

from loguru import logger

from src.cache.historical import HistoricalCache
from src.daemon.event_watcher import EventWatcher, EventWatcherConfig
from src.daemon.events import BaseEvent, NewsEvent
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

    def __init__(  # noqa: PLR0913,D417 - Backward compat, prefer NewsWatcherConfig
        self,
        historical_cache: HistoricalCache,
        config: NewsWatcherConfig | None = None,
        poll_interval: int | None = None,
        relevance_threshold: float | None = None,
        cooldown_minutes: int | None = None,
        breaking_threshold_minutes: int | None = None,
        max_concurrent_analyses: int | None = None,
    ) -> None:
        """Initialize news watcher.

        Args:
            historical_cache: Shared cache for news data
            config: Configuration (uses defaults if not provided)
            **Individual params for backward compatibility (prefer config object)
        """
        # Backward compat: construct config from individual params if provided
        if config is None and (
            poll_interval is not None
            or relevance_threshold is not None
            or cooldown_minutes is not None
            or breaking_threshold_minutes is not None
            or max_concurrent_analyses is not None
        ):
            defaults = NewsWatcherConfig()
            config = NewsWatcherConfig(
                poll_interval=poll_interval if poll_interval is not None else defaults.poll_interval,
                relevance_threshold=(
                    relevance_threshold if relevance_threshold is not None else defaults.relevance_threshold
                ),
                cooldown_minutes=(
                    cooldown_minutes if cooldown_minutes is not None else defaults.cooldown_minutes
                ),
                breaking_threshold_minutes=(
                    breaking_threshold_minutes
                    if breaking_threshold_minutes is not None
                    else defaults.breaking_threshold_minutes
                ),
                max_concurrent_analyses=(
                    max_concurrent_analyses
                    if max_concurrent_analyses is not None
                    else defaults.max_concurrent_analyses
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
        self._news_fetcher: NewsFetcher | None = None
        self._seen_urls: deque[str] = deque(maxlen=100)  # Rolling window auto-evicts oldest

        logger.info(
            f"NewsWatcher initialized (breaking_threshold={cfg.breaking_threshold_minutes}m, "
            f"threshold={cfg.relevance_threshold})"
        )

    def _init_components(self) -> None:
        """Lazy initialization including news fetcher."""
        super()._init_components()
        if self._news_fetcher is None:
            self._news_fetcher = NewsFetcher(historical_cache=self._historical_cache)

    async def _fetch_events(self) -> list[BaseEvent]:
        """Fetch breaking news from Marketaux.

        Returns:
            List of NewsEvent objects for breaking news
        """
        self._init_components()
        if self._news_fetcher is None:
            msg = "Failed to initialize NewsFetcher"
            raise RuntimeError(msg)

        # Fetch recent news (no symbol filter)
        articles = self._news_fetcher.fetch_market_news(limit=50)

        # Filter: breaking (published <N min ago + keywords)
        breaking: list[BaseEvent] = []
        now = datetime.now(UTC)

        for article in articles:
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
                            source="marketaux",
                            article=article,
                        ),
                    )
                )
                self._seen_urls.append(article.url)  # Auto-evicts oldest when maxlen reached
                logger.info(f"Breaking news detected: {article.title[:60]}... (age: {age_minutes:.1f}m)")

        return breaking

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"NewsWatcher(poll_interval={self.poll_interval}s, "
            f"breaking_threshold={self.breaking_threshold_minutes}m)"
        )
