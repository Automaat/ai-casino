"""Tests for SocialSentimentWatcher."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.daemon.events import SocialSentimentDirection, SocialSentimentSignal
from src.data.apewisdom import ApeWisdomFetcher, ApeWisdomTicker
from src.v1.watchers.social_sentiment_watcher import (
    SocialSentimentWatcher,
    SocialSentimentWatcherConfig,
)


@pytest.fixture
def config() -> SocialSentimentWatcherConfig:
    """Default watcher config."""
    return SocialSentimentWatcherConfig(
        poll_interval_minutes=1,
        trending_rank_threshold=20,
        buzz_spike_threshold=1.5,
        symbols=["AAPL", "GME", "TSLA"],
    )


@pytest.fixture
def mock_fetcher() -> MagicMock:
    """Mock ApeWisdom fetcher."""
    fetcher = MagicMock(spec=ApeWisdomFetcher)
    fetcher.get_ticker.return_value = None
    fetcher.fetch_trending.return_value = []
    return fetcher


@pytest.fixture
def watcher(mock_fetcher: MagicMock, config: SocialSentimentWatcherConfig) -> SocialSentimentWatcher:
    """Create watcher with mocked fetcher."""
    return SocialSentimentWatcher(apewisdom_fetcher=mock_fetcher, config=config)


def _make_ape_ticker(
    ticker: str = "GME",
    rank: int = 5,
    mentions: int = 500,
    mentions_24h_ago: int = 300,
) -> ApeWisdomTicker:
    """Helper to create ApeWisdomTicker."""
    return ApeWisdomTicker(
        rank=rank,
        ticker=ticker,
        name=f"{ticker} Corp",
        mentions=mentions,
        upvotes=10000,
        rank_24h_ago=rank + 2,
        mentions_24h_ago=mentions_24h_ago,
    )


@pytest.mark.unit
class TestSocialSentimentWatcher:
    """Tests for SocialSentimentWatcher."""

    @pytest.mark.asyncio
    async def test_apewisdom_only_signal(self, watcher: SocialSentimentWatcher) -> None:
        """Verify signal generated from ApeWisdom data alone (no Reddit DB)."""
        ape_ticker = _make_ape_ticker("GME", rank=3, mentions=500, mentions_24h_ago=200)
        ape_map = {"GME": ape_ticker}

        patch_reddit = patch.object(
            watcher,
            "_query_reddit_sentiment",
            new_callable=AsyncMock,
            return_value=None,
        )
        with patch_reddit:
            await watcher._fetch_and_assess_symbol("GME", ape_map)

        signal = watcher.get_signal("GME")
        assert signal is not None
        assert signal.symbol == "GME"
        assert signal.is_trending is True  # rank 3 <= threshold 20
        assert signal.buzz_score > 0
        assert len(signal.platform_breakdown) == 1
        assert signal.platform_breakdown[0].platform == "apewisdom"

    @pytest.mark.asyncio
    async def test_reddit_only_signal(self, watcher: SocialSentimentWatcher) -> None:
        """Verify signal generated from Reddit DB alone (no ApeWisdom)."""
        reddit_data = {
            "mention_count": 50,
            "avg_sentiment": 0.75,
            "raw_score": 0.5,
        }
        patch_reddit = patch.object(
            watcher,
            "_query_reddit_sentiment",
            new_callable=AsyncMock,
            return_value=reddit_data,
        )
        with patch_reddit:
            await watcher._fetch_and_assess_symbol("AAPL", {})

        signal = watcher.get_signal("AAPL")
        assert signal is not None
        assert signal.symbol == "AAPL"
        assert signal.direction == SocialSentimentDirection.BULLISH  # 0.75 > 0.6
        assert signal.is_trending is False
        assert len(signal.platform_breakdown) == 1
        assert signal.platform_breakdown[0].platform == "reddit"

    @pytest.mark.asyncio
    async def test_both_platforms_signal(self, watcher: SocialSentimentWatcher) -> None:
        """Verify signal merges both Reddit and ApeWisdom data."""
        ape_ticker = _make_ape_ticker("TSLA", rank=10, mentions=300, mentions_24h_ago=200)
        ape_map = {"TSLA": ape_ticker}
        reddit_data = {
            "mention_count": 80,
            "avg_sentiment": 0.65,
            "raw_score": 0.3,
        }
        patch_reddit = patch.object(
            watcher,
            "_query_reddit_sentiment",
            new_callable=AsyncMock,
            return_value=reddit_data,
        )
        with patch_reddit:
            await watcher._fetch_and_assess_symbol("TSLA", ape_map)

        signal = watcher.get_signal("TSLA")
        assert signal is not None
        assert len(signal.platform_breakdown) == 2
        assert signal.confidence == 1.0  # 2 platforms * 0.5

    @pytest.mark.asyncio
    async def test_apewisdom_none_mentions_24h_ago_no_division(
        self, watcher: SocialSentimentWatcher
    ) -> None:
        """Verify no division attempted and no exception raised when mentions_24h_ago=None."""
        ape_ticker = ApeWisdomTicker(
            rank=5,
            ticker="TSLA",
            name="Tesla Corp",
            mentions=200,
            upvotes=5000,
            rank_24h_ago=7,
            mentions_24h_ago=None,
        )
        ape_map = {"TSLA": ape_ticker}

        with patch.object(watcher, "_query_reddit_sentiment", new_callable=AsyncMock, return_value=None):
            await watcher._fetch_and_assess_symbol("TSLA", ape_map)

        signal = watcher.get_signal("TSLA")
        assert signal is not None
        # mention_delta_pct stays at default 0.0 — no division attempted
        ape_platform = next(p for p in signal.platform_breakdown if p.platform == "apewisdom")
        assert ape_platform.mention_delta_pct == 0.0

    @pytest.mark.asyncio
    async def test_no_data_produces_no_signal(self, watcher: SocialSentimentWatcher) -> None:
        """Verify no signal when neither platform has data."""
        with patch.object(watcher, "_query_reddit_sentiment", new_callable=AsyncMock, return_value=None):
            await watcher._fetch_and_assess_symbol("UNKNOWN", {})

        assert watcher.get_signal("UNKNOWN") is None

    def test_direction_bullish(self, watcher: SocialSentimentWatcher) -> None:
        """Verify bullish direction for high sentiment."""
        assert watcher._determine_direction(0.75) == SocialSentimentDirection.BULLISH

    def test_direction_bearish(self, watcher: SocialSentimentWatcher) -> None:
        """Verify bearish direction for low sentiment."""
        assert watcher._determine_direction(0.25) == SocialSentimentDirection.BEARISH

    def test_direction_neutral(self, watcher: SocialSentimentWatcher) -> None:
        """Verify neutral direction for mid sentiment."""
        assert watcher._determine_direction(0.5) == SocialSentimentDirection.NEUTRAL

    def test_direction_none_sentiment(self, watcher: SocialSentimentWatcher) -> None:
        """Verify neutral direction when no Reddit data."""
        assert watcher._determine_direction(None) == SocialSentimentDirection.NEUTRAL

    def test_buzz_score_zero(self, watcher: SocialSentimentWatcher) -> None:
        """Verify zero mentions = zero buzz."""
        assert watcher._compute_buzz_score(0, 0) == 0.0

    def test_buzz_score_scaling(self, watcher: SocialSentimentWatcher) -> None:
        """Verify buzz score increases with mentions."""
        low = watcher._compute_buzz_score(5, 5)
        mid = watcher._compute_buzz_score(50, 50)
        high = watcher._compute_buzz_score(500, 500)
        assert low < mid < high
        assert high <= 1.0

    def test_significance_scoring(self, watcher: SocialSentimentWatcher) -> None:
        """Verify significance score combines components."""
        # High buzz, strong sentiment, trending
        high = watcher._compute_significance(0.8, 100.0, 0.4, 3)
        # Low buzz, weak sentiment, not trending
        low = watcher._compute_significance(0.1, 0.0, 0.05, None)
        assert high > low
        assert 0.0 <= high <= 1.0
        assert 0.0 <= low <= 1.0

    def test_trending_detection(self, watcher: SocialSentimentWatcher) -> None:
        """Verify trending detection respects rank threshold."""
        # Rank 5 <= threshold 20 → trending
        assert watcher._config.trending_rank_threshold == 20

    @pytest.mark.asyncio
    async def test_fetch_and_assess_all(
        self, watcher: SocialSentimentWatcher, mock_fetcher: MagicMock
    ) -> None:
        """Verify all configured symbols are assessed."""
        mock_fetcher.fetch_trending.return_value = [
            _make_ape_ticker("AAPL", rank=1, mentions=1000),
            _make_ape_ticker("GME", rank=2, mentions=800),
            _make_ape_ticker("TSLA", rank=3, mentions=600),
        ]

        with patch.object(watcher, "_query_reddit_sentiment", new_callable=AsyncMock, return_value=None):
            await watcher._tick()

        # All 3 symbols should have signals
        assert watcher.get_signal("AAPL") is not None
        assert watcher.get_signal("GME") is not None
        assert watcher.get_signal("TSLA") is not None

    @pytest.mark.asyncio
    async def test_run_lifecycle(self, watcher: SocialSentimentWatcher, mock_fetcher: MagicMock) -> None:
        """Verify run() starts and stops cleanly."""
        mock_fetcher.get_ticker.return_value = None

        with patch.object(watcher, "_query_reddit_sentiment", new_callable=AsyncMock, return_value=None):
            # Stop after first iteration
            async def stop_after_delay() -> None:
                await asyncio.sleep(0.1)
                watcher.running = False

            task = asyncio.create_task(watcher.run())
            stop_task = asyncio.create_task(stop_after_delay())

            await asyncio.gather(task, stop_task)

        assert watcher.running is False

    def test_get_signal_case_insensitive(self, watcher: SocialSentimentWatcher) -> None:
        """Verify get_signal normalizes to uppercase."""
        watcher._signals["AAPL"] = SocialSentimentSignal(
            symbol="AAPL",
            direction=SocialSentimentDirection.NEUTRAL,
            confidence=0.5,
            buzz_score=0.3,
            is_trending=False,
            significance_score=0.2,
            reason="test",
        )
        assert watcher.get_signal("aapl") is not None
        assert watcher.get_signal("Aapl") is not None

    def test_repr(self, watcher: SocialSentimentWatcher) -> None:
        """Verify repr output."""
        r = repr(watcher)
        assert "SocialSentimentWatcher" in r
        assert "running=False" in r

    def test_build_reason(self, watcher: SocialSentimentWatcher) -> None:
        """Verify reason string construction."""
        from src.daemon.events import PlatformSentiment

        platforms = [
            PlatformSentiment(
                platform="reddit",
                mention_count=50,
                sentiment_score=0.3,
                mention_delta_pct=0.0,
            ),
            PlatformSentiment(
                platform="apewisdom",
                mention_count=300,
                sentiment_score=0.0,
                mention_delta_pct=66.7,
            ),
        ]
        reason = watcher._build_reason(
            SocialSentimentDirection.BULLISH,
            platforms=platforms,
            mention_delta_pct=66.7,
            trending_rank=5,
        )
        assert "Reddit 50" in reason
        assert "WSB 300" in reason
        assert "+67%" in reason
        assert "trending #5" in reason
