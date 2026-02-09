"""Social media watcher for Reddit-based market signal monitoring.

Detects two types of events:
- Volume spikes: 50%+ increase in symbol mentions between polls
- Viral posts: High-score posts (<1hr old, >1000 score, >80% upvote ratio)
"""

from collections import deque
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from loguru import logger

from src.cache.historical import HistoricalCache
from src.daemon.event_watcher import EventWatcher
from src.daemon.events import BaseEvent, SocialEvent
from src.data.reddit import RedditFetcher

if TYPE_CHECKING:
    from src.data.reddit import TrendingTicker


class SocialWatcher(EventWatcher):
    """Watcher for Reddit-based social media events.

    Monitors Reddit communities (r/wallstreetbets, r/stocks) for volume spikes
    and viral posts that may signal trading opportunities.
    """

    def __init__(  # noqa: PLR0913
        self,
        historical_cache: HistoricalCache,
        poll_interval: int = 900,
        relevance_threshold: float = 0.7,
        cooldown_minutes: int = 15,
        volume_spike_threshold: float = 0.5,
        viral_score_threshold: int = 1000,
        viral_upvote_ratio: float = 0.8,
        subreddits: list[str] | None = None,
        max_concurrent_analyses: int = 2,
    ) -> None:
        """Initialize social watcher.

        Args:
            historical_cache: Shared cache for social data
            poll_interval: Seconds between poll cycles
            relevance_threshold: Minimum relevance score to trigger analysis
            cooldown_minutes: Minutes to wait before re-analyzing same symbol
            volume_spike_threshold: Minimum % increase for volume spike (e.g., 0.5 = 50%)
            viral_score_threshold: Minimum post score for viral detection
            viral_upvote_ratio: Minimum upvote ratio for viral detection
            subreddits: Subreddits to monitor (defaults to wallstreetbets, stocks)
            max_concurrent_analyses: Maximum symbols to analyze per cycle
        """
        super().__init__(
            poll_interval,
            relevance_threshold,
            cooldown_minutes,
            max_concurrent_analyses=max_concurrent_analyses,
            historical_cache=historical_cache,
        )
        self.volume_spike_threshold = volume_spike_threshold
        self.viral_score_threshold = viral_score_threshold
        self.viral_upvote_ratio = viral_upvote_ratio
        self.subreddits = subreddits or ["wallstreetbets", "stocks"]

        # State tracking (in-memory)
        self._reddit_fetcher: RedditFetcher | None = None
        self._seen_post_ids: deque[str] = deque(maxlen=500)  # Auto-evict oldest
        self._previous_mention_counts: dict[str, int] = {}  # Symbol -> count baseline
        self._mention_count_order: deque[str] = deque(maxlen=300)  # LRU tracking

        logger.info(
            f"SocialWatcher initialized (volume_spike={volume_spike_threshold:.0%}, "
            f"viral_score={viral_score_threshold}, subreddits={self.subreddits})"
        )

    def _init_components(self) -> None:
        """Lazy initialization including Reddit fetcher."""
        super()._init_components()
        if self._reddit_fetcher is None:
            self._reddit_fetcher = RedditFetcher(historical_cache=self._historical_cache)

    def _update_mention_baseline(self, symbol: str, count: int) -> None:
        """Update mention count baseline with LRU eviction.

        Args:
            symbol: Stock ticker symbol
            count: Current mention count
        """
        # LRU eviction if at capacity
        if symbol not in self._previous_mention_counts and len(self._previous_mention_counts) >= 300:
            oldest = self._mention_count_order[0]
            self._previous_mention_counts.pop(oldest, None)
            logger.debug(f"Evicted {oldest} from mention count tracking (LRU limit reached)")

        # Update LRU order
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

        logger.info(
            f"Volume spike detected: {symbol} "
            f"({prev_count} → {current_count}, +{delta_pct:.1f}%)"
        )
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

    def _check_viral_posts(self, ticker: "TrendingTicker", symbol: str, now: datetime) -> list[SocialEvent]:
        """Check ticker posts for viral content and return detected events."""
        events: list[SocialEvent] = []

        for post in ticker.sample_posts:
            age_seconds = (now - post.created_utc).total_seconds()
            if age_seconds > 3600:  # >1hr old
                continue
            if post.score < self.viral_score_threshold:
                continue
            if post.upvote_ratio < self.viral_upvote_ratio:
                continue
            if post.id in self._seen_post_ids:
                continue

            events.append(
                SocialEvent(
                    event_id=f"reddit_viral_{post.id}",
                    event_type="social",
                    timestamp=post.created_utc,
                    source="reddit",
                    symbol=symbol,
                    mention_count=None,
                    mention_delta_pct=None,
                    viral_post=post,
                )
            )
            self._seen_post_ids.append(post.id)
            logger.info(
                f"Viral post detected: {symbol} - {post.title[:60]}... "
                f"(score: {post.score}, ratio: {post.upvote_ratio:.1%}, age: {age_seconds / 60:.1f}m)"
            )

        return events

    async def _fetch_events(self) -> list[BaseEvent]:
        """Fetch social events from Reddit (volume spikes + viral posts).

        Returns:
            List of SocialEvent objects for detected signals
        """
        self._init_components()
        if self._reddit_fetcher is None:
            msg = "Failed to initialize RedditFetcher"
            raise RuntimeError(msg)

        trending = self._reddit_fetcher.fetch_trending_tickers(
            subreddits=self.subreddits, limit=100, min_mentions=1
        )

        events: list[SocialEvent] = []
        now = datetime.now(UTC)

        for ticker in trending:
            symbol = ticker.symbol
            current_count = ticker.mention_count

            # Phase 1: Volume spike detection
            volume_event = self._check_volume_spike(symbol, current_count, now)
            if volume_event:
                events.append(volume_event)

            # Phase 2: Viral post detection
            viral_events = self._check_viral_posts(ticker, symbol, now)
            events.extend(viral_events)

            # Update baseline for next poll
            self._update_mention_baseline(symbol, current_count)

        return events

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"SocialWatcher(poll_interval={self.poll_interval}s, "
            f"volume_spike={self.volume_spike_threshold:.0%}, "
            f"viral_score={self.viral_score_threshold})"
        )
