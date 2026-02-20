"""Social media watcher for Reddit-based market signal monitoring.

Detects two types of events:
- Volume spikes: 50%+ increase in symbol mentions between polls
- Viral posts: High-score posts (<1hr old, >1000 score, >80% upvote ratio)
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Final, cast

from loguru import logger

from src.cache.historical import HistoricalCache
from src.daemon.events import BaseEvent, SocialEvent
from src.data.reddit import RedditFetcher, RedditPost, TrendingTicker
from src.database.connection import get_session
from src.database.repositories.reddit import RedditPostRepository, RedditTickerMentionRepository
from src.v1.watchers.base import PeriodicWatcher
from src.v1.watchers.pipeline import EventTriagePipeline

_MAX_MENTION_TRACKING: Final[int] = 300
_VIRAL_POST_MAX_AGE_SECONDS: Final[int] = 3600


@dataclass
class SocialWatcherConfig:
    """Configuration for SocialWatcher."""

    poll_interval: int = 900
    volume_spike_threshold: float = 0.5
    viral_score_threshold: int = 1000
    viral_upvote_ratio: float = 0.8
    subreddits: list[str] = field(default_factory=lambda: ["wallstreetbets", "stocks"])


class SocialWatcher(PeriodicWatcher):
    """Watcher for Reddit-based social media events.

    Monitors Reddit communities for volume spikes and viral posts
    that may signal trading opportunities.
    """

    def __init__(
        self,
        pipeline: EventTriagePipeline,
        historical_cache: HistoricalCache,
        config: SocialWatcherConfig | None = None,
    ) -> None:
        """Initialize social watcher.

        Args:
            pipeline: Event triage pipeline for routing events
            historical_cache: Cache for RedditFetcher initialization
            config: Watcher configuration
        """
        cfg = config or SocialWatcherConfig()
        super().__init__(poll_interval=cfg.poll_interval)
        self._pipeline = pipeline
        self._historical_cache = historical_cache
        self.volume_spike_threshold = cfg.volume_spike_threshold
        self.viral_score_threshold = cfg.viral_score_threshold
        self.viral_upvote_ratio = cfg.viral_upvote_ratio
        self.subreddits = cfg.subreddits

        self._reddit_fetcher: RedditFetcher | None = None
        self._seen_post_ids: deque[str] = deque(maxlen=500)
        self._previous_mention_counts: dict[str, int] = {}
        self._mention_count_order: deque[str] = deque(maxlen=300)

        logger.info(
            f"SocialWatcher initialized (volume_spike={cfg.volume_spike_threshold:.0%}, "
            f"viral_score={cfg.viral_score_threshold}, subreddits={self.subreddits})"
        )

    @property
    def name(self) -> str:
        """Watcher display name."""
        return "SocialWatcher"

    async def _tick(self) -> None:
        """Fetch and process social events."""
        events = await self._fetch_events()
        if events:
            await self._pipeline.process(events)

    def _update_mention_baseline(self, symbol: str, count: int) -> None:
        """Update mention count baseline with LRU eviction.

        Args:
            symbol: Stock ticker symbol
            count: Current mention count
        """
        at_capacity = len(self._previous_mention_counts) >= _MAX_MENTION_TRACKING
        if symbol not in self._previous_mention_counts and at_capacity:
            oldest = self._mention_count_order[0]
            self._previous_mention_counts.pop(oldest, None)
            logger.debug(f"Evicted {oldest} from mention count tracking (LRU limit reached)")

        if symbol in self._mention_count_order:
            self._mention_count_order.remove(symbol)
        self._mention_count_order.append(symbol)
        self._previous_mention_counts[symbol] = count

    def _check_volume_spike(self, symbol: str, current_count: int, now: datetime) -> SocialEvent | None:
        """Check if symbol has volume spike and return event if detected."""
        if symbol not in self._previous_mention_counts:
            return None

        prev_count = self._previous_mention_counts[symbol]
        if prev_count == 0:
            return None

        delta_pct = ((current_count - prev_count) / prev_count) * 100
        if delta_pct < self.volume_spike_threshold * 100:
            return None

        logger.info(f"Volume spike detected: {symbol} ({prev_count} → {current_count}, +{delta_pct:.1f}%)")
        return SocialEvent(
            event_id=f"reddit_volume_{symbol}_{now.isoformat()}",
            event_type="social",
            timestamp=now,
            source="reddit",
            symbol=symbol,
            mention_count=current_count,
            mention_delta_pct=delta_pct,
            viral_post=None,
        )

    def _check_viral_posts(self, ticker: TrendingTicker, symbol: str, now: datetime) -> list[SocialEvent]:
        """Check ticker posts for viral content and return detected events."""
        events: list[SocialEvent] = []

        for post in ticker.sample_posts:
            event = self._check_viral_post(post, symbol, now)
            if event:
                events.append(event)

        return events

    def _check_viral_post(self, post: RedditPost, symbol: str, now: datetime) -> SocialEvent | None:
        """Check if single post is viral and return event if detected."""
        age_seconds = (now - post.created_utc).total_seconds()
        if age_seconds > _VIRAL_POST_MAX_AGE_SECONDS:
            return None
        if post.score < self.viral_score_threshold:
            return None
        if post.upvote_ratio < self.viral_upvote_ratio:
            return None
        if post.id in self._seen_post_ids:
            return None

        self._seen_post_ids.append(post.id)
        logger.info(
            f"Viral post detected: {symbol} - {post.title[:60]}... "
            f"(score: {post.score}, ratio: {post.upvote_ratio:.1%}, age: {age_seconds / 60:.1f}m)"
        )

        return SocialEvent(
            event_id=f"reddit_viral_{post.id}",
            event_type="social",
            timestamp=post.created_utc,
            source="reddit",
            symbol=symbol,
            mention_count=None,
            mention_delta_pct=None,
            viral_post=post,
        )

    async def _fetch_events_from_db(self, now: datetime) -> list[BaseEvent] | None:
        """Fetch social events from DB.

        Returns:
            List of events if DB has data, None if no data found
        """
        async with get_session() as session:
            post_repo = RedditPostRepository(session)
            mention_repo = RedditTickerMentionRepository(session)

            poll_window = self.poll_interval // 60
            viral_window = 60

            mention_counts = await mention_repo.get_mentions_in_window(window_minutes=poll_window)
            recent_posts = await post_repo.get_posts_in_window(
                window_minutes=viral_window, subreddits=self.subreddits
            )
            post_symbols_map = await mention_repo.get_post_symbols_map(window_minutes=viral_window)

            if not (mention_counts or recent_posts):
                return None

            events: list[BaseEvent] = []
            for symbol, current_count in mention_counts:
                volume_event = self._check_volume_spike(symbol, current_count, now)
                if volume_event:
                    events.append(cast("BaseEvent", volume_event))
                self._update_mention_baseline(symbol, current_count)

            for post in recent_posts:
                symbols = post_symbols_map.get(post.id, [])
                if not symbols:
                    continue
                for symbol in symbols:
                    viral_event = self._check_viral_post(post, symbol, now)
                    if viral_event:
                        events.append(cast("BaseEvent", viral_event))
                        break

            logger.debug(
                f"Fetched {len(events)} events from DB "
                f"(posts={len(recent_posts)}, viral_window={viral_window}min)"
            )
            return events

    async def _fetch_events_from_api(self, now: datetime) -> list[BaseEvent]:
        """Fetch social events from Reddit API (fallback).

        Returns:
            List of SocialEvent objects for detected signals
        """
        if self._reddit_fetcher is None:
            self._reddit_fetcher = RedditFetcher(historical_cache=self._historical_cache)

        trending = self._reddit_fetcher.fetch_trending_tickers(
            subreddits=self.subreddits, limit=100, min_mentions=1
        )

        events: list[BaseEvent] = []
        for ticker in trending:
            symbol = ticker.symbol
            current_count = ticker.mention_count

            volume_event = self._check_volume_spike(symbol, current_count, now)
            if volume_event:
                events.append(cast("BaseEvent", volume_event))

            viral_events = self._check_viral_posts(ticker, symbol, now)
            events.extend(cast("list[BaseEvent]", viral_events))

            self._update_mention_baseline(symbol, current_count)

        logger.debug(f"Fetched {len(events)} events from API (trending={len(trending)})")
        return events

    async def _fetch_events(self) -> list[BaseEvent]:
        """Fetch social events from Reddit (volume spikes + viral posts).

        Tries DB-based approach first, falls back to API-based.

        Returns:
            List of SocialEvent objects for detected signals
        """
        now = datetime.now(UTC)

        try:
            events = await self._fetch_events_from_db(now)
            if events is not None:
                return events
        except Exception as e:
            logger.debug(f"DB fetch failed ({e}), falling back to API-based approach")

        return await self._fetch_events_from_api(now)

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"SocialWatcher(poll_interval={self.poll_interval}s, "
            f"volume_spike={self.volume_spike_threshold:.0%}, "
            f"viral_score={self.viral_score_threshold})"
        )
