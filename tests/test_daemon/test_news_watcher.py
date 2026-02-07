"""Tests for NewsWatcher."""

from datetime import UTC, datetime, timedelta
from unittest.mock import Mock, patch

import pytest

from src.cache.historical import HistoricalCache
from src.daemon.watchers.news_watcher import NewsWatcher
from src.data.news import NewsArticle


@pytest.fixture
def mock_historical_cache():
    """Mock historical cache."""
    return Mock(spec=HistoricalCache)


@pytest.fixture
def news_watcher(mock_historical_cache):
    """Create NewsWatcher instance."""
    return NewsWatcher(
        historical_cache=mock_historical_cache,
        poll_interval=300,
        relevance_threshold=0.7,
        cooldown_minutes=15,
        breaking_threshold_minutes=15,
    )


def test_news_watcher_init(news_watcher, mock_historical_cache):
    """Test NewsWatcher initialization."""
    assert news_watcher.poll_interval == 300
    assert news_watcher.relevance_threshold == 0.7
    assert news_watcher.cooldown_minutes == 15
    assert news_watcher.breaking_threshold_minutes == 15
    assert news_watcher._historical_cache == mock_historical_cache
    assert news_watcher._news_fetcher is None
    assert len(news_watcher._seen_urls) == 0


def test_breaking_keywords_defined(news_watcher):
    """Test that breaking keywords are defined."""
    assert "breaking" in NewsWatcher.BREAKING_KEYWORDS
    assert "announces" in NewsWatcher.BREAKING_KEYWORDS
    assert "reports earnings" in NewsWatcher.BREAKING_KEYWORDS
    assert len(NewsWatcher.BREAKING_KEYWORDS) > 10


@pytest.mark.asyncio
async def test_fetch_events_breaking_news(news_watcher):
    """Test fetching breaking news events."""
    now = datetime.now(UTC)

    # Create breaking news (recent + keyword)
    breaking_article = NewsArticle(
        title="Breaking: Apple announces major acquisition",
        description="Apple Inc announces acquisition deal",
        source="Bloomberg",
        published_at=now - timedelta(minutes=5),
        url="https://example.com/breaking-1",
    )

    # Create old news (should be filtered)
    old_article = NewsArticle(
        title="Breaking: Old news",
        description="Old news content",
        source="Reuters",
        published_at=now - timedelta(minutes=30),
        url="https://example.com/old-1",
    )

    # Create recent but not breaking
    normal_article = NewsArticle(
        title="Market update for today",
        description="Normal market update",
        source="CNBC",
        published_at=now - timedelta(minutes=3),
        url="https://example.com/normal-1",
    )

    with patch.object(news_watcher, "_init_components"):
        news_watcher._news_fetcher = Mock()
        news_watcher._news_fetcher.fetch_market_news.return_value = [
            breaking_article,
            old_article,
            normal_article,
        ]

        events = await news_watcher._fetch_events()

    # Should only return breaking article
    assert len(events) == 1
    assert events[0].article.title == "Breaking: Apple announces major acquisition"
    assert events[0].event_id == "https://example.com/breaking-1"
    assert events[0].source == "marketaux"


@pytest.mark.asyncio
async def test_fetch_events_deduplication(news_watcher):
    """Test URL deduplication in fetch_events."""
    now = datetime.now(UTC)

    article = NewsArticle(
        title="Breaking news about merger",
        description="Merger announcement",
        source="WSJ",
        published_at=now - timedelta(minutes=2),
        url="https://example.com/duplicate",
    )

    with patch.object(news_watcher, "_init_components"):
        news_watcher._news_fetcher = Mock()
        news_watcher._news_fetcher.fetch_market_news.return_value = [article]

        # First fetch
        events1 = await news_watcher._fetch_events()
        assert len(events1) == 1

        # Second fetch with same article
        events2 = await news_watcher._fetch_events()
        assert len(events2) == 0  # Deduplicated


@pytest.mark.asyncio
async def test_fetch_events_rolling_window(news_watcher):
    """Test that seen_urls maintains rolling window of 100."""
    now = datetime.now(UTC)

    # Add 110 URLs to seen_urls
    for i in range(110):
        news_watcher._seen_urls.add(f"https://example.com/{i}")

    article = NewsArticle(
        title="Breaking: New announcement",
        description="New content",
        source="Reuters",
        published_at=now - timedelta(minutes=1),
        url="https://example.com/new",
    )

    with patch.object(news_watcher, "_init_components"):
        news_watcher._news_fetcher = Mock()
        news_watcher._news_fetcher.fetch_market_news.return_value = [article]

        await news_watcher._fetch_events()

    # Should keep rolling window of 100
    assert len(news_watcher._seen_urls) == 100


@pytest.mark.asyncio
async def test_fetch_events_no_breaking_news(news_watcher):
    """Test when no breaking news is found."""
    with patch.object(news_watcher, "_init_components"):
        news_watcher._news_fetcher = Mock()
        news_watcher._news_fetcher.fetch_market_news.return_value = []

        events = await news_watcher._fetch_events()

    assert len(events) == 0


def test_repr(news_watcher):
    """Test string representation."""
    repr_str = repr(news_watcher)

    assert "NewsWatcher" in repr_str
    assert "300s" in repr_str
    assert "15m" in repr_str
