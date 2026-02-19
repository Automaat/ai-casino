"""Social sentiment watcher — aggregates retail sentiment from Reddit DB + ApeWisdom."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from loguru import logger

from src.daemon.events import (
    PlatformSentiment,
    SocialSentimentDirection,
    SocialSentimentSignal,
)
from src.data.apewisdom import ApeWisdomFetcher, ApeWisdomTicker
from src.watchers.base import PeriodicWatcher


@dataclass
class SocialSentimentWatcherConfig:
    """Configuration for social sentiment watcher."""

    poll_interval_minutes: int = 30
    trending_rank_threshold: int = 20
    buzz_spike_threshold: float = 1.5
    symbols: list[str] | None = None


class SocialSentimentWatcher(PeriodicWatcher):
    """Background service that polls Reddit DB + ApeWisdom and computes social signals."""

    def __init__(
        self,
        apewisdom_fetcher: ApeWisdomFetcher,
        config: SocialSentimentWatcherConfig,
    ) -> None:
        """Initialize social sentiment watcher.

        Args:
            apewisdom_fetcher: ApeWisdom trending data fetcher
            config: Watcher configuration
        """
        super().__init__(poll_interval=config.poll_interval_minutes * 60)
        self._fetcher = apewisdom_fetcher
        self._config = config
        self._signals: dict[str, SocialSentimentSignal] = {}

    @property
    def name(self) -> str:
        """Watcher display name."""
        return "SocialSentimentWatcher"

    def get_signal(self, symbol: str) -> SocialSentimentSignal | None:
        """Return current social sentiment signal for a symbol (sync).

        Args:
            symbol: Stock ticker

        Returns:
            SocialSentimentSignal if available, None otherwise
        """
        return self._signals.get(symbol.upper())

    def _determine_direction(self, reddit_sentiment: float | None) -> SocialSentimentDirection:
        """Determine sentiment direction from Reddit DB score.

        Score is 0-1 where 0=bearish, 0.5=neutral, 1=bullish.

        Args:
            reddit_sentiment: Average sentiment score from DB (0-1), None if no data

        Returns:
            SocialSentimentDirection
        """
        if reddit_sentiment is None:
            return SocialSentimentDirection.NEUTRAL
        if reddit_sentiment > 0.6:
            return SocialSentimentDirection.BULLISH
        if reddit_sentiment < 0.4:
            return SocialSentimentDirection.BEARISH
        return SocialSentimentDirection.NEUTRAL

    def _compute_buzz_score(
        self,
        reddit_mentions: int,
        apewisdom_mentions: int,
    ) -> float:
        """Compute normalized buzz score (0-1) from combined mentions.

        Uses log scaling: 10 mentions ≈ 0.33, 100 ≈ 0.67, 1000+ ≈ 1.0

        Args:
            reddit_mentions: Mention count from Reddit DB
            apewisdom_mentions: Mention count from ApeWisdom

        Returns:
            Buzz score 0.0-1.0
        """
        import math

        total = reddit_mentions + apewisdom_mentions
        if total == 0:
            return 0.0
        # log10(1)=0, log10(10)=1, log10(100)=2, log10(1000)=3
        return min(1.0, math.log10(max(1, total)) / 3.0)

    def _compute_significance(
        self,
        buzz_score: float,
        mention_delta_pct: float,
        sentiment_strength: float,
        trending_rank: int | None,
    ) -> float:
        """Compute composite significance score (0.0-1.0).

        Weights: 40% buzz spike, 30% sentiment strength, 30% trending rank.

        Args:
            buzz_score: Normalized buzz (0-1)
            mention_delta_pct: % change in mentions vs prior period
            sentiment_strength: Distance from neutral (0-0.5)
            trending_rank: ApeWisdom rank (lower = better), None if not trending

        Returns:
            Significance score 0.0-1.0
        """
        # Buzz spike component: gate by buzz_spike_threshold, then normalize delta%
        if mention_delta_pct >= self._config.buzz_spike_threshold * 100:
            spike_score = min(1.0, max(0.0, mention_delta_pct / 150.0))
        elif mention_delta_pct > 0:
            spike_score = min(1.0, max(0.0, mention_delta_pct / 150.0)) * 0.5
        else:
            spike_score = buzz_score

        # Sentiment strength: 0.5 deviation = 1.0
        strength_score = min(1.0, sentiment_strength / 0.5)

        # Trending rank component: rank 1 = 1.0, rank at threshold ~= 0.0, not trending = 0.0
        if trending_rank is not None and trending_rank > 0:
            threshold = max(1, self._config.trending_rank_threshold)
            rank_score = max(0.0, 1.0 - (trending_rank - 1) / float(threshold))
        else:
            rank_score = 0.0

        return 0.4 * spike_score + 0.3 * strength_score + 0.3 * rank_score

    @staticmethod
    def _build_reason(
        direction: SocialSentimentDirection,
        platforms: list[PlatformSentiment],
        mention_delta_pct: float,
        trending_rank: int | None,
    ) -> str:
        """Build human-readable reason string.

        Args:
            direction: Sentiment direction
            platforms: Platform sentiment breakdown
            mention_delta_pct: Mention delta percentage
            trending_rank: ApeWisdom rank (None if not trending)

        Returns:
            Reason string
        """
        parts = []
        for p in platforms:
            label = "WSB" if p.platform == "apewisdom" else p.platform.capitalize()
            parts.append(f"{label} {p.mention_count} mentions")
        if mention_delta_pct > 0:
            parts.append(f"+{mention_delta_pct:.0f}% vs 24h ago")
        if trending_rank is not None:
            parts.append(f"trending #{trending_rank}")

        if not parts:
            return f"{direction} social sentiment, minimal activity"
        return f"{direction} social: {'; '.join(parts)}"

    async def _fetch_and_assess_symbol(
        self,
        symbol: str,
        ape_map: dict[str, ApeWisdomTicker],
    ) -> None:
        """Fetch and assess social sentiment for a single symbol.

        Args:
            symbol: Stock ticker
            ape_map: Pre-fetched ApeWisdom ticker map (symbol -> ApeWisdomTicker)
        """
        symbol_upper = symbol.upper()
        platforms: list[PlatformSentiment] = []
        reddit_mentions = 0
        reddit_sentiment: float | None = None
        apewisdom_mentions = 0
        mention_delta_pct = 0.0
        trending_rank: int | None = None

        # ApeWisdom data from pre-fetched map (avoids concurrent HTTP races)
        ape_ticker = ape_map.get(symbol_upper)
        if ape_ticker:
            apewisdom_mentions = ape_ticker.mentions
            trending_rank = ape_ticker.rank

            # Compute delta from 24h ago
            if ape_ticker.mentions_24h_ago > 0:
                mention_delta_pct = (
                    (ape_ticker.mentions - ape_ticker.mentions_24h_ago) / ape_ticker.mentions_24h_ago * 100
                )

            platforms.append(
                PlatformSentiment(
                    platform="apewisdom",
                    mention_count=apewisdom_mentions,
                    sentiment_score=0.0,  # ApeWisdom has no sentiment
                    mention_delta_pct=mention_delta_pct,
                )
            )

        # Reddit DB data (query latest hourly aggregate)
        try:
            reddit_data = await self._query_reddit_sentiment(symbol_upper)
            if reddit_data:
                reddit_mentions = reddit_data["mention_count"]
                reddit_sentiment = reddit_data["avg_sentiment"]
                platforms.append(
                    PlatformSentiment(
                        platform="reddit",
                        mention_count=reddit_mentions,
                        sentiment_score=reddit_data["raw_score"],
                        mention_delta_pct=0.0,  # No delta from DB aggregates
                    )
                )
        except Exception as e:
            logger.opt(exception=True).warning(f"Reddit DB query failed for {symbol_upper}: {e}")

        # Skip if no data from any platform
        if not platforms:
            return

        direction = self._determine_direction(reddit_sentiment)
        buzz_score = self._compute_buzz_score(reddit_mentions, apewisdom_mentions)
        is_trending = trending_rank is not None and trending_rank <= self._config.trending_rank_threshold
        sentiment_strength = abs((reddit_sentiment or 0.5) - 0.5)
        significance = self._compute_significance(
            buzz_score, mention_delta_pct, sentiment_strength, trending_rank
        )
        confidence = min(1.0, len(platforms) * 0.5)  # 0.5 per platform, max 1.0

        reason = self._build_reason(
            direction,
            platforms,
            mention_delta_pct,
            trending_rank if is_trending else None,
        )

        signal = SocialSentimentSignal(
            symbol=symbol_upper,
            direction=direction,
            confidence=confidence,
            buzz_score=buzz_score,
            platform_breakdown=platforms,
            is_trending=is_trending,
            significance_score=significance,
            reason=reason,
        )
        self._signals[symbol_upper] = signal

    async def _query_reddit_sentiment(self, symbol: str) -> dict | None:
        """Query latest Reddit sentiment aggregate from DB.

        Args:
            symbol: Stock ticker

        Returns:
            Dict with mention_count, avg_sentiment, raw_score or None
        """
        try:
            from src.database.connection import get_session
            from src.database.models.reddit import RedditTickerSentimentORM

            async with get_session() as session:
                from datetime import UTC, datetime, timedelta

                from sqlalchemy import func, select

                cutoff = datetime.now(UTC) - timedelta(hours=6)

                # Find the latest window_start for this symbol within cutoff
                latest_window_subq = (
                    select(func.max(RedditTickerSentimentORM.window_start))
                    .where(
                        RedditTickerSentimentORM.symbol == symbol,
                        RedditTickerSentimentORM.window_start >= cutoff,
                    )
                    .scalar_subquery()
                )

                # Aggregate across subreddits for that single latest window
                query = select(
                    func.sum(RedditTickerSentimentORM.mention_count).label("total_mentions"),
                    func.avg(RedditTickerSentimentORM.avg_sentiment).label("avg_sentiment"),
                    func.sum(RedditTickerSentimentORM.bullish_count).label("total_bullish"),
                    func.sum(RedditTickerSentimentORM.bearish_count).label("total_bearish"),
                    func.sum(RedditTickerSentimentORM.neutral_count).label("total_neutral"),
                ).where(
                    RedditTickerSentimentORM.symbol == symbol,
                    RedditTickerSentimentORM.window_start == latest_window_subq,
                )

                result = await session.execute(query)
                row = result.one_or_none()

                if not row or not row.total_mentions:
                    return None

                total_mentions = int(row.total_mentions)
                avg_sentiment = float(row.avg_sentiment)
                bullish = int(row.total_bullish or 0)
                bearish = int(row.total_bearish or 0)
                neutral = int(row.total_neutral or 0)
                total = bullish + bearish + neutral
                raw_score = ((bullish - bearish) / total) if total > 0 else 0.0

                return {
                    "mention_count": total_mentions,
                    "avg_sentiment": avg_sentiment,
                    "raw_score": raw_score,
                }

        except Exception as e:
            logger.opt(exception=True).warning(f"Reddit sentiment DB query failed: {e}")
            return None

    async def _tick(self) -> None:
        """Fetch and assess all configured symbols with concurrency limit."""
        symbols = self._config.symbols or []
        if not symbols:
            return

        # Fetch trending list once to avoid concurrent cache races in ApeWisdomFetcher
        trending = await asyncio.to_thread(self._fetcher.fetch_trending)
        ape_map = {t.ticker.upper(): t for t in trending}

        sem = asyncio.Semaphore(3)

        async def _limited(sym: str) -> None:
            async with sem:
                try:
                    await self._fetch_and_assess_symbol(sym, ape_map)
                except Exception as e:
                    logger.opt(exception=True).warning(f"Social sentiment assessment failed for {sym}: {e}")

        await asyncio.gather(*[_limited(s) for s in symbols])

        active = [s for s in self._signals.values() if s.significance_score >= 0.3]
        logger.info(f"Social sentiment assessed: {len(symbols)} symbols, {len(active)} with notable activity")

    def __repr__(self) -> str:
        """String representation."""
        return f"SocialSentimentWatcher(running={self.running}, signals={len(self._signals)})"
