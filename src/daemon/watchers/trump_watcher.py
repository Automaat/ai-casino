"""Trump Truth Social watcher for market-relevant posts.

Polls Trump's Truth Social feed every 5 minutes, deduplicates via post ID tracking,
and triggers LLM triage + analysis for relevant posts.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from loguru import logger

from src.cache.historical import HistoricalCache
from src.daemon.event_watcher import EventWatcher, EventWatcherConfig
from src.daemon.events import BaseEvent, TrumpEvent
from src.data.truth_social import TruthSocialFetcher

if TYPE_CHECKING:
    from src.di.container import AppContainer


@dataclass
class TrumpWatcherConfig:
    """Configuration for TrumpWatcher."""

    poll_interval: int = 300
    relevance_threshold: float = 0.7
    cooldown_minutes: int = 15
    max_concurrent_analyses: int = 2


class TrumpWatcher(EventWatcher):
    """Watcher for Trump Truth Social posts.

    Monitors Trump's Truth Social feed for market-relevant posts.
    Maintains rolling window of seen post IDs to prevent re-processing.
    """

    def __init__(
        self,
        historical_cache: HistoricalCache,
        config: TrumpWatcherConfig | None = None,
        container: AppContainer | None = None,
        **kwargs: int | float,
    ) -> None:
        """Initialize Trump watcher.

        Args:
            historical_cache: Shared cache for Truth Social data
            config: Configuration (uses defaults if not provided)
            container: Optional DI container (auto-created if not provided)
            **kwargs: Backward compat params (poll_interval, relevance_threshold, etc.)
        """
        # Backward compat: construct config from kwargs if provided
        if config is None and kwargs:
            defaults = TrumpWatcherConfig()
            config = TrumpWatcherConfig(
                poll_interval=int(kwargs.get("poll_interval", defaults.poll_interval)),
                relevance_threshold=float(kwargs.get("relevance_threshold", defaults.relevance_threshold)),
                cooldown_minutes=int(kwargs.get("cooldown_minutes", defaults.cooldown_minutes)),
                max_concurrent_analyses=int(
                    kwargs.get("max_concurrent_analyses", defaults.max_concurrent_analyses)
                ),
            )

        cfg = config or TrumpWatcherConfig()
        base_config = EventWatcherConfig(
            poll_interval=cfg.poll_interval,
            relevance_threshold=cfg.relevance_threshold,
            cooldown_minutes=cfg.cooldown_minutes,
            max_concurrent_analyses=cfg.max_concurrent_analyses,
        )
        super().__init__(base_config, historical_cache, container=container)

        # Truth Social fetcher (lazy init)
        self._truth_fetcher: TruthSocialFetcher | None = None
        self._seen_post_ids: set[str] = set()
        self._last_post_id: str | None = None

        logger.info(
            f"TrumpWatcher initialized (poll_interval={cfg.poll_interval}s, "
            f"threshold={cfg.relevance_threshold})"
        )

    def _init_components(self) -> None:
        """Lazy initialization including Truth Social fetcher."""
        super()._init_components()
        if self._truth_fetcher is None:
            self._truth_fetcher = TruthSocialFetcher(historical_cache=self._historical_cache)

    async def _fetch_events(self) -> list[BaseEvent]:
        """Fetch new Trump posts since last check.

        Returns:
            List of TrumpEvent objects for new posts
        """
        self._init_components()
        if self._truth_fetcher is None:
            msg = "Failed to initialize TruthSocialFetcher"
            raise RuntimeError(msg)

        # Fetch recent posts (last hour on first run, since last check otherwise)
        if self._last_check is None:
            data = self._truth_fetcher.fetch_recent(hours=1)
        else:
            data = self._truth_fetcher.fetch_since(self._last_check)

        if not data.posts:
            return []

        # Filter to only new posts (deduplicate by ID)
        new_posts = []
        for post in data.posts:
            if post.id not in self._seen_post_ids:
                new_posts.append(post)
                self._seen_post_ids.add(post.id)

        if new_posts:
            self._last_post_id = new_posts[0].id

        logger.debug(f"Found {len(new_posts)} new Trump posts")

        # Convert to TrumpEvent objects (cast to satisfy Protocol variance)
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
