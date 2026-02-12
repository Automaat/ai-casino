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


async def test_fetch_events_seen_urls_dict(news_watcher):
    """Test that seen_urls tracks URL-to-source mapping."""
    now = datetime.now(UTC)

    # Add URLs to seen_urls dict
    for i in range(10):
        news_watcher._seen_urls[f"https://example.com/{i}"] = "marketaux"

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

        # New URL should be added
        assert "https://example.com/new" in news_watcher._seen_urls
        assert news_watcher._seen_urls["https://example.com/new"] == "marketaux"


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


@pytest.mark.asyncio
async def test_multi_source_deduplication(mock_historical_cache):
    """Test weighted deduplication across multiple sources."""
    from unittest.mock import AsyncMock

    from src.data.base_news_fetcher import BaseNewsFetcher

    # Mock marketaux fetcher (highest weight)
    mock_marketaux = AsyncMock(spec=BaseNewsFetcher)
    mock_marketaux.get_source_name.return_value = "marketaux"
    mock_marketaux.afetch_market_news.return_value = [
        NewsArticle(
            title="Breaking: Article 1",
            description="Important news",
            url="https://example.com/1",
            published_at=datetime.now(UTC),
            source="Bloomberg",
        )
    ]

    # Mock duckduckgo fetcher (lowest weight)
    mock_duckduckgo = AsyncMock(spec=BaseNewsFetcher)
    mock_duckduckgo.get_source_name.return_value = "duckduckgo"
    mock_duckduckgo.afetch_market_news.return_value = [
        NewsArticle(
            title="Breaking: Article 1 DDG",
            description="Different description",
            url="https://example.com/1",  # Same URL
            published_at=datetime.now(UTC),
            source="DDG Source",
        ),
        NewsArticle(
            title="Breaking: Article 2",
            description="Unique DDG article",
            url="https://example.com/2",
            published_at=datetime.now(UTC),
            source="DDG Source",
        ),
    ]

    watcher = NewsWatcher(
        historical_cache=mock_historical_cache,
        fetchers=[mock_marketaux, mock_duckduckgo],
    )

    events = await watcher._fetch_events()

    # Should keep marketaux article (higher weight) + unique DDG article
    urls = [e.article.url for e in events]
    assert "https://example.com/1" in urls
    assert "https://example.com/2" in urls

    # Verify marketaux source kept for duplicate URL
    event1 = next(e for e in events if e.article.url == "https://example.com/1")
    assert event1.source == "marketaux"


@pytest.mark.asyncio
async def test_multi_source_handles_fetcher_failure(mock_historical_cache):
    """Test graceful degradation when one fetcher fails."""
    from unittest.mock import AsyncMock

    import httpx

    from src.data.base_news_fetcher import BaseNewsFetcher

    mock_marketaux = AsyncMock(spec=BaseNewsFetcher)
    mock_marketaux.get_source_name.return_value = "marketaux"
    mock_marketaux.afetch_market_news.side_effect = httpx.HTTPError("API down")

    mock_finnhub = AsyncMock(spec=BaseNewsFetcher)
    mock_finnhub.get_source_name.return_value = "finnhub"
    mock_finnhub.afetch_market_news.return_value = [
        NewsArticle(
            title="Breaking: Finnhub article",
            description="Announcement",
            url="https://example.com/finnhub",
            published_at=datetime.now(UTC),
            source="Finnhub",
        )
    ]

    watcher = NewsWatcher(
        historical_cache=mock_historical_cache,
        fetchers=[mock_marketaux, mock_finnhub],
    )

    events = await watcher._fetch_events()

    # Should get finnhub article despite marketaux failure
    assert len(events) > 0
    assert events[0].source == "finnhub"


@pytest.mark.asyncio
async def test_multi_source_no_fetchers_uses_fallback(mock_historical_cache):
    """Test fallback to single Marketaux fetcher when no fetchers provided."""
    watcher = NewsWatcher(
        historical_cache=mock_historical_cache,
        breaking_threshold_minutes=15,
    )

    # Mock the fallback fetcher
    with patch.object(watcher, "_init_components"):
        watcher._news_fetcher = Mock()
        watcher._news_fetcher.fetch_market_news.return_value = [
            NewsArticle(
                title="Breaking: Fallback article",
                description="Test",
                url="https://example.com/fallback",
                published_at=datetime.now(UTC),
                source="Marketaux",
            )
        ]

        events = await watcher._fetch_events()

    assert len(events) > 0
    assert events[0].source == "marketaux"
