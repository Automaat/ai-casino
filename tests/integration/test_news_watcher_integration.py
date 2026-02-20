"""Integration tests for NewsWatcher lifecycle."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, Mock

import pytest

from src.cache.historical import HistoricalCache
from src.data.base_news_fetcher import BaseNewsFetcher
from src.data.news import NewsArticle
from src.v1.watchers.news_watcher import NewsWatcher, NewsWatcherConfig
from src.v1.watchers.pipeline import EventTriagePipeline


@pytest.fixture
def mock_historical_cache():
    """Mock historical cache."""
    return Mock(spec=HistoricalCache)


@pytest.fixture
def mock_multi_source_fetchers():
    """Mock all 4 news fetchers with get_source_name()."""
    marketaux = AsyncMock(spec=BaseNewsFetcher)
    marketaux.get_source_name.return_value = "marketaux"
    marketaux.afetch_market_news.return_value = []

    finnhub = AsyncMock(spec=BaseNewsFetcher)
    finnhub.get_source_name.return_value = "finnhub"
    finnhub.afetch_market_news.return_value = []

    newsdata = AsyncMock(spec=BaseNewsFetcher)
    newsdata.get_source_name.return_value = "newsdata"
    newsdata.afetch_market_news.return_value = []

    duckduckgo = AsyncMock(spec=BaseNewsFetcher)
    duckduckgo.get_source_name.return_value = "duckduckgo"
    duckduckgo.afetch_market_news.return_value = []

    return [marketaux, finnhub, newsdata, duckduckgo]


@pytest.fixture
def mock_pipeline():
    """Mock EventTriagePipeline."""
    pipeline = Mock(spec=EventTriagePipeline)
    pipeline.process = AsyncMock()
    return pipeline


@pytest.fixture
def integration_watcher(mock_historical_cache, mock_multi_source_fetchers, mock_pipeline):
    """NewsWatcher configured for integration tests."""
    config = NewsWatcherConfig(
        poll_interval=60,
        breaking_threshold_minutes=15,
    )
    return NewsWatcher(
        pipeline=mock_pipeline,
        historical_cache=mock_historical_cache,
        fetchers=mock_multi_source_fetchers,
        config=config,
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_watcher_tick_calls_pipeline(integration_watcher, mock_multi_source_fetchers, mock_pipeline):
    """Test _tick() fetches events and routes through pipeline."""
    now = datetime.now(UTC)
    breaking_article = NewsArticle(
        title="Breaking: Apple announces major product",
        description="Apple Inc announces new product line",
        url="https://example.com/breaking",
        published_at=now - timedelta(minutes=5),
        source="Bloomberg",
    )

    # Mock fetcher to return breaking news
    mock_multi_source_fetchers[0].afetch_market_news.return_value = [breaking_article]

    await integration_watcher._tick()

    # Verify: fetch happened and pipeline.process was called with events
    mock_multi_source_fetchers[0].afetch_market_news.assert_called_once()
    mock_pipeline.process.assert_called_once()
    events_arg = mock_pipeline.process.call_args[0][0]
    assert len(events_arg) == 1
    assert events_arg[0].article.title == "Breaking: Apple announces major product"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_multi_source_deduplication_integration(mock_historical_cache, mock_pipeline):
    """Test weighted deduplication keeps highest-weight source."""
    now = datetime.now(UTC)
    duplicate_url = "https://example.com/duplicate"

    # Marketaux (weight 1.0)
    mock_marketaux = AsyncMock(spec=BaseNewsFetcher)
    mock_marketaux.get_source_name.return_value = "marketaux"
    mock_marketaux.afetch_market_news.return_value = [
        NewsArticle(
            title="Breaking: Marketaux version",
            description="Marketaux description",
            url=duplicate_url,
            published_at=now - timedelta(minutes=5),
            source="Marketaux",
        )
    ]

    # Finnhub (weight 0.9)
    mock_finnhub = AsyncMock(spec=BaseNewsFetcher)
    mock_finnhub.get_source_name.return_value = "finnhub"
    mock_finnhub.afetch_market_news.return_value = [
        NewsArticle(
            title="Breaking: Finnhub version",
            description="Finnhub description",
            url=duplicate_url,
            published_at=now - timedelta(minutes=5),
            source="Finnhub",
        )
    ]

    watcher = NewsWatcher(
        pipeline=mock_pipeline,
        historical_cache=mock_historical_cache,
        fetchers=[mock_marketaux, mock_finnhub],
        config=NewsWatcherConfig(),
    )

    events = await watcher._fetch_events()

    # Should only have one event (Marketaux preferred)
    assert len(events) == 1
    assert events[0].source == "marketaux"
    assert events[0].event_id == duplicate_url

    # Verify URL tracked in _seen_urls
    assert duplicate_url in watcher._seen_urls
    assert watcher._seen_urls[duplicate_url] == "marketaux"
