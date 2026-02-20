"""Tests for SocialWatcher."""

import os
from datetime import UTC, datetime, timedelta
from unittest.mock import Mock, patch

import pytest

from src.cache.historical import HistoricalCache
from src.data.reddit import RedditPost, TrendingTicker
from src.v1.watchers.pipeline import EventTriagePipeline
from src.v1.watchers.social_watcher import SocialWatcher, SocialWatcherConfig


@pytest.fixture(autouse=True)
def _mock_env_vars():
    """Set required env vars for tests."""
    with patch.dict(os.environ, {"ALPHA_VANTAGE_API_KEY": "test_key"}):
        yield


@pytest.fixture
def historical_cache(tmp_path):
    """Create test historical cache."""
    cache_path = tmp_path / "test_cache.db"
    return HistoricalCache(str(cache_path))


@pytest.fixture
def mock_reddit_fetcher():
    """Create mock Reddit fetcher."""
    return Mock()


@pytest.fixture
def mock_pipeline():
    """Mock EventTriagePipeline."""
    return Mock(spec=EventTriagePipeline)


@pytest.fixture
def social_watcher(historical_cache, mock_reddit_fetcher, mock_pipeline):
    """Create SocialWatcher with mock fetcher."""
    watcher = SocialWatcher(
        pipeline=mock_pipeline,
        historical_cache=historical_cache,
        config=SocialWatcherConfig(
            poll_interval=900,
            volume_spike_threshold=0.5,
            viral_score_threshold=1000,
            viral_upvote_ratio=0.8,
            subreddits=["wallstreetbets", "stocks"],
        ),
    )
    watcher._reddit_fetcher = mock_reddit_fetcher
    return watcher


def create_reddit_post(
    post_id: str,
    symbol: str,
    title: str,
    score: int = 500,
    upvote_ratio: float = 0.9,
    age_minutes: int = 30,
) -> RedditPost:
    """Helper to create test Reddit post."""
    return RedditPost(
        id=post_id,
        title=title,
        body=f"Discussion about {symbol}",
        subreddit="wallstreetbets",
        score=score,
        upvote_ratio=upvote_ratio,
        url=f"https://reddit.com/r/wallstreetbets/comments/{post_id}",
        created_utc=datetime.now(UTC) - timedelta(minutes=age_minutes),
        num_comments=100,
    )


def test_initialization(historical_cache):
    """Test SocialWatcher initialization."""
    watcher = SocialWatcher(
        pipeline=Mock(spec=EventTriagePipeline),
        historical_cache=historical_cache,
        config=SocialWatcherConfig(
            poll_interval=600,
            volume_spike_threshold=0.6,
            viral_score_threshold=1500,
            viral_upvote_ratio=0.85,
            subreddits=["wallstreetbets"],
        ),
    )

    assert watcher.poll_interval == 600
    assert watcher.volume_spike_threshold == 0.6
    assert watcher.viral_score_threshold == 1500
    assert watcher.viral_upvote_ratio == 0.85
    assert watcher.subreddits == ["wallstreetbets"]
    assert len(watcher._seen_post_ids) == 0
    assert len(watcher._previous_mention_counts) == 0


async def test_volume_spike_detection(social_watcher, mock_reddit_fetcher):
    """Test volume spike event detection."""
    # Setup: First poll establishes baseline
    ticker1 = TrendingTicker(
        symbol="AAPL",
        mention_count=10,
        total_score=5000,
        avg_upvote_ratio=0.85,
        sample_posts=[],
    )
    mock_reddit_fetcher.fetch_trending_tickers.return_value = [ticker1]

    events = await social_watcher._fetch_events()
    assert len(events) == 0  # No events on first poll (no baseline)
    assert social_watcher._previous_mention_counts["AAPL"] == 10

    # Second poll: 100% increase (10 -> 20)
    ticker2 = TrendingTicker(
        symbol="AAPL",
        mention_count=20,
        total_score=10000,
        avg_upvote_ratio=0.85,
        sample_posts=[],
    )
    mock_reddit_fetcher.fetch_trending_tickers.return_value = [ticker2]

    events = await social_watcher._fetch_events()
    assert len(events) == 1
    assert events[0].event_type == "social"
    assert events[0].symbol == "AAPL"
    assert events[0].mention_count == 20
    assert events[0].mention_delta_pct == 100.0
    assert events[0].viral_post is None


async def test_first_poll_no_baseline(social_watcher, mock_reddit_fetcher):
    """Test first poll does not generate volume spike events."""
    ticker = TrendingTicker(
        symbol="TSLA",
        mention_count=50,
        total_score=25000,
        avg_upvote_ratio=0.9,
        sample_posts=[],
    )
    mock_reddit_fetcher.fetch_trending_tickers.return_value = [ticker]

    events = await social_watcher._fetch_events()
    assert len(events) == 0
    assert social_watcher._previous_mention_counts["TSLA"] == 50


async def test_viral_post_detection(social_watcher, mock_reddit_fetcher):
    """Test viral post event detection."""
    viral_post = create_reddit_post(
        post_id="abc123",
        symbol="GME",
        title="GME to the moon!",
        score=2000,
        upvote_ratio=0.95,
        age_minutes=30,
    )

    ticker = TrendingTicker(
        symbol="GME",
        mention_count=15,
        total_score=8000,
        avg_upvote_ratio=0.9,
        sample_posts=[viral_post],
    )
    mock_reddit_fetcher.fetch_trending_tickers.return_value = [ticker]

    events = await social_watcher._fetch_events()
    assert len(events) == 1
    assert events[0].event_type == "social"
    assert events[0].symbol == "GME"
    assert events[0].viral_post is not None
    assert events[0].viral_post.id == "abc123"
    assert events[0].viral_post.score == 2000
    assert events[0].mention_count is None
    assert events[0].mention_delta_pct is None


async def test_viral_post_age_filtering(social_watcher, mock_reddit_fetcher):
    """Test posts older than 1hr are excluded."""
    old_post = create_reddit_post(
        post_id="old123",
        symbol="NVDA",
        title="NVDA analysis",
        score=2000,
        upvote_ratio=0.95,
        age_minutes=90,  # >1hr
    )

    ticker = TrendingTicker(
        symbol="NVDA",
        mention_count=20,
        total_score=10000,
        avg_upvote_ratio=0.9,
        sample_posts=[old_post],
    )
    mock_reddit_fetcher.fetch_trending_tickers.return_value = [ticker]

    events = await social_watcher._fetch_events()
    assert len(events) == 0  # Old post filtered out


async def test_viral_post_score_filtering(social_watcher, mock_reddit_fetcher):
    """Test posts below score threshold are excluded."""
    low_score_post = create_reddit_post(
        post_id="low123",
        symbol="AMD",
        title="AMD discussion",
        score=500,  # Below 1000 threshold
        upvote_ratio=0.95,
        age_minutes=30,
    )

    ticker = TrendingTicker(
        symbol="AMD",
        mention_count=25,
        total_score=12000,
        avg_upvote_ratio=0.9,
        sample_posts=[low_score_post],
    )
    mock_reddit_fetcher.fetch_trending_tickers.return_value = [ticker]

    events = await social_watcher._fetch_events()
    assert len(events) == 0  # Low score post filtered out


async def test_viral_post_ratio_filtering(social_watcher, mock_reddit_fetcher):
    """Test posts below upvote ratio threshold are excluded."""
    low_ratio_post = create_reddit_post(
        post_id="ratio123",
        symbol="MSFT",
        title="MSFT news",
        score=2000,
        upvote_ratio=0.7,  # Below 0.8 threshold
        age_minutes=30,
    )

    ticker = TrendingTicker(
        symbol="MSFT",
        mention_count=30,
        total_score=15000,
        avg_upvote_ratio=0.85,
        sample_posts=[low_ratio_post],
    )
    mock_reddit_fetcher.fetch_trending_tickers.return_value = [ticker]

    events = await social_watcher._fetch_events()
    assert len(events) == 0  # Low ratio post filtered out


async def test_viral_post_deduplication(social_watcher, mock_reddit_fetcher):
    """Test duplicate post IDs are filtered out."""
    viral_post = create_reddit_post(
        post_id="dup123",
        symbol="GOOGL",
        title="GOOGL breakout",
        score=2000,
        upvote_ratio=0.95,
        age_minutes=30,
    )

    ticker = TrendingTicker(
        symbol="GOOGL",
        mention_count=18,
        total_score=9000,
        avg_upvote_ratio=0.9,
        sample_posts=[viral_post],
    )
    mock_reddit_fetcher.fetch_trending_tickers.return_value = [ticker]

    # First fetch: post detected
    events = await social_watcher._fetch_events()
    assert len(events) == 1
    assert "dup123" in social_watcher._seen_post_ids

    # Second fetch: same post ID filtered
    events = await social_watcher._fetch_events()
    assert len(events) == 0


async def test_multiple_events_same_poll(social_watcher, mock_reddit_fetcher):
    """Test multiple events (volume + viral) in same poll."""
    # Establish baseline first
    ticker1 = TrendingTicker(
        symbol="AAPL",
        mention_count=10,
        total_score=5000,
        avg_upvote_ratio=0.85,
        sample_posts=[],
    )
    mock_reddit_fetcher.fetch_trending_tickers.return_value = [ticker1]
    await social_watcher._fetch_events()

    # Second poll: volume spike + viral post
    viral_post = create_reddit_post(
        post_id="multi123",
        symbol="AAPL",
        title="AAPL earnings beat!",
        score=3000,
        upvote_ratio=0.98,
        age_minutes=15,
    )

    ticker2 = TrendingTicker(
        symbol="AAPL",
        mention_count=20,  # 100% increase
        total_score=30000,
        avg_upvote_ratio=0.9,
        sample_posts=[viral_post],
    )
    mock_reddit_fetcher.fetch_trending_tickers.return_value = [ticker2]

    events = await social_watcher._fetch_events()
    assert len(events) == 2

    # Check volume spike event
    volume_event = next(e for e in events if e.mention_delta_pct is not None)
    assert volume_event.symbol == "AAPL"
    assert volume_event.mention_count == 20
    assert volume_event.mention_delta_pct == 100.0

    # Check viral event
    viral_event = next(e for e in events if e.viral_post is not None)
    assert viral_event.symbol == "AAPL"
    assert viral_event.viral_post.id == "multi123"


async def test_zero_previous_count_handling(social_watcher, mock_reddit_fetcher):
    """Test handling of zero previous count (no div/0 error)."""
    # Manually set previous count to 0
    social_watcher._previous_mention_counts["FAKE"] = 0

    ticker = TrendingTicker(
        symbol="FAKE",
        mention_count=10,
        total_score=5000,
        avg_upvote_ratio=0.85,
        sample_posts=[],
    )
    mock_reddit_fetcher.fetch_trending_tickers.return_value = [ticker]

    events = await social_watcher._fetch_events()
    assert len(events) == 0  # No spike event (skip div/0)
    assert social_watcher._previous_mention_counts["FAKE"] == 10


async def test_negative_delta_handling(social_watcher, mock_reddit_fetcher):
    """Test negative delta (mentions decreased) does not trigger event."""
    # Establish baseline
    ticker1 = TrendingTicker(
        symbol="META",
        mention_count=50,
        total_score=25000,
        avg_upvote_ratio=0.85,
        sample_posts=[],
    )
    mock_reddit_fetcher.fetch_trending_tickers.return_value = [ticker1]
    await social_watcher._fetch_events()

    # Second poll: mentions decreased
    ticker2 = TrendingTicker(
        symbol="META",
        mention_count=30,  # -40%
        total_score=15000,
        avg_upvote_ratio=0.85,
        sample_posts=[],
    )
    mock_reddit_fetcher.fetch_trending_tickers.return_value = [ticker2]

    events = await social_watcher._fetch_events()
    assert len(events) == 0  # No event for negative delta


async def test_repr(social_watcher):
    """Test string representation."""
    repr_str = repr(social_watcher)
    assert "SocialWatcher" in repr_str
    assert "poll_interval=900s" in repr_str
    assert "volume_spike=50%" in repr_str
    assert "viral_score=1000" in repr_str
