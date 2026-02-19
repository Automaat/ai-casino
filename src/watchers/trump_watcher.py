"""Trump Truth Social watcher for market-relevant posts.

Polls Trump's Truth Social feed, deduplicates via post ID tracking,
and routes events through EventTriagePipeline.
"""

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import cast

from loguru import logger

from src.cache.historical import HistoricalCache
from src.daemon.events import BaseEvent, TrumpEvent
from src.data.truth_social import TruthSocialFetcher
from src.watchers.base import PeriodicWatcher
from src.watchers.pipeline import EventTriagePipeline


@dataclass
class TrumpWatcherConfig:
    """Configuration for TrumpWatcher."""

    poll_interval: int = 300


class TrumpWatcher(PeriodicWatcher):
    """Watcher for Trump Truth Social posts.

    Monitors Trump's Truth Social feed for market-relevant posts.
    Maintains rolling window of seen post IDs to prevent re-processing.
    """

    def __init__(
        self,
        pipeline: EventTriagePipeline,
        historical_cache: HistoricalCache,
        config: TrumpWatcherConfig | None = None,
    ) -> None:
        """Initialize Trump watcher.

        Args:
            pipeline: Event triage pipeline for routing events
            historical_cache: Cache for TruthSocialFetcher initialization
            config: Watcher configuration
        """
        cfg = config or TrumpWatcherConfig()
        super().__init__(poll_interval=cfg.poll_interval)
        self._pipeline = pipeline
        self._historical_cache = historical_cache

        self._truth_fetcher: TruthSocialFetcher | None = None
        self._seen_post_ids: deque[str] = deque(maxlen=500)
        self._last_check: datetime | None = None
        self._last_post_id: str | None = None

        logger.info(f"TrumpWatcher initialized (poll_interval={cfg.poll_interval}s)")

    @property
    def name(self) -> str:
        """Watcher display name."""
        return "TrumpWatcher"

    async def _tick(self) -> None:
        """Fetch and process Trump post events."""
        events = await self._fetch_events()
        if events:
            await self._pipeline.process(events)

    async def _fetch_events(self) -> list[BaseEvent]:
        """Fetch new Trump posts since last check.

        Returns:
            List of TrumpEvent objects for new posts
        """
        if self._truth_fetcher is None:
            self._truth_fetcher = TruthSocialFetcher(historical_cache=self._historical_cache)

        if self._last_check is None:
            data = self._truth_fetcher.fetch_recent(hours=1)
        else:
            data = self._truth_fetcher.fetch_since(self._last_check)

        self._last_check = data.fetched_at

        if not data.posts:
            return []

        new_posts = []
        for post in data.posts:
            if post.id not in self._seen_post_ids:
                new_posts.append(post)
                self._seen_post_ids.append(post.id)

        if new_posts:
            self._last_post_id = new_posts[0].id

        logger.debug(f"Found {len(new_posts)} new Trump posts")

        events: list[BaseEvent] = [
            cast(
                "BaseEvent",
                TrumpEvent(
                    event_id=post.id,
                    timestamp=post.created_at,
                    post=post,
                ),
            )
            for post in new_posts
        ]
        return events

    def __repr__(self) -> str:
        """Return string representation."""
        return f"TrumpWatcher(poll_interval={self.poll_interval}s)"
