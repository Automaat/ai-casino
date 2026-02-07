"""News watcher for breaking financial news events.

Polls Marketaux API every 5 minutes, filters by breaking keywords and recency,
deduplicates via URL tracking, and triggers LLM triage + analysis for relevant events.
"""

from datetime import UTC, datetime
from typing import ClassVar

from loguru import logger

from src.cache.historical import HistoricalCache
from src.daemon.event_watcher import EventWatcher
from src.daemon.events import NewsEvent
from src.data.news import NewsFetcher


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

    def __init__(
        self,
        historical_cache: HistoricalCache,
        poll_interval: int = 300,
        relevance_threshold: float = 0.7,
        cooldown_minutes: int = 15,
        breaking_threshold_minutes: int = 15,
    ) -> None:
        """Initialize news watcher.

        Args:
            historical_cache: Shared cache for news data
            poll_interval: Seconds between poll cycles
            relevance_threshold: Minimum relevance score to trigger analysis
            cooldown_minutes: Minutes to wait before re-analyzing same symbol
            breaking_threshold_minutes: Max age for breaking news (minutes)
        """
        super().__init__(
            poll_interval,
            relevance_threshold,
            cooldown_minutes,
            max_concurrent_analyses=2,  # Fixed value for news watcher
            historical_cache=historical_cache,
        )
        self.breaking_threshold_minutes = breaking_threshold_minutes
        self._news_fetcher: NewsFetcher | None = None
        self._seen_urls: set[str] = set()  # Rolling window

        logger.info(
            f"NewsWatcher initialized (breaking_threshold={breaking_threshold_minutes}m, "
            f"threshold={relevance_threshold})"
        )

    def _init_components(self) -> None:
        """Lazy initialization including news fetcher."""
        super()._init_components()
        if self._news_fetcher is None:
            self._news_fetcher = NewsFetcher(historical_cache=self._historical_cache)

    async def _fetch_events(self) -> list[NewsEvent]:
        """Fetch breaking news from Marketaux.

        Returns:
            List of NewsEvent objects for breaking news
        """
        self._init_components()

        # Fetch recent news (no symbol filter)
        articles = self._news_fetcher.fetch_market_news(limit=50)

        # Filter: breaking (published <N min ago + keywords)
        breaking = []
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
                    NewsEvent(
                        event_id=article.url,
                        event_type="news",
                        timestamp=article.published_at,
                        source="marketaux",
                        article=article,
                    )
                )
                self._seen_urls.add(article.url)
                logger.info(f"Breaking news detected: {article.title[:60]}... (age: {age_minutes:.1f}m)")

        # Keep rolling window of 100 URLs
        if len(self._seen_urls) > 100:
            self._seen_urls = set(list(self._seen_urls)[-100:])

        return breaking

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"NewsWatcher(poll_interval={self.poll_interval}s, "
            f"breaking_threshold={self.breaking_threshold_minutes}m)"
        )
