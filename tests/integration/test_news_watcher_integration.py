"""Integration tests for NewsWatcher lifecycle."""

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, Mock

import pytest

from src.cache.historical import HistoricalCache
from src.daemon.events import EventSignal, Sentiment, TriageResult, Urgency
from src.daemon.watchers.news_watcher import NewsWatcher, NewsWatcherConfig
from src.data.base_news_fetcher import BaseNewsFetcher
from src.data.news import NewsArticle
from src.strategies.signal import Signal
from src.workflows.types import TradingDecision, TradingWorkflowResult


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
def mock_triage_agent():
    """Mock EventTriageAgent with high-relevance triage result."""
    agent = AsyncMock()
    agent.analyze.return_value = TriageResult(
        event_id="test-event",
        event_type="news",
        relevance=0.9,
        urgency=Urgency.IMMEDIATE,
        sentiment=Sentiment.BULLISH,
        symbols=["AAPL"],
        confidence=0.85,
        reasoning="Test reasoning",
    )
    return agent


@pytest.fixture
def mock_workflow():
    """Mock TradingWorkflow with analysis results."""
    workflow = AsyncMock()

    # Use model_construct to bypass validation
    result = TradingWorkflowResult.model_construct(
        symbol="AAPL",
        decision=TradingDecision.model_construct(
            action=Signal.BUY,
            confidence=0.85,
            reasoning=["Test analysis"],
            risk_level="LOW",
        ),
        technical=Mock(),
        sentiment=Mock(),
        news=Mock(),
        bullish=Mock(),
        bearish=Mock(),
        risk=Mock(),
    )

    workflow.analyze.return_value = result
    return workflow


@pytest.fixture
def integration_watcher(mock_historical_cache, mock_multi_source_fetchers):
    """NewsWatcher configured for integration tests."""
    config = NewsWatcherConfig(
        poll_interval=60,
        relevance_threshold=0.7,
        cooldown_minutes=15,
        breaking_threshold_minutes=15,
        max_concurrent_analyses=2,
        period_days=30,
    )
    return NewsWatcher(
        historical_cache=mock_historical_cache,
        fetchers=mock_multi_source_fetchers,
        config=config,
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_watcher_lifecycle_single_cycle(
    integration_watcher, mock_multi_source_fetchers, mock_triage_agent, mock_workflow
):
    """Test complete lifecycle: fetch → triage → analyze (skip signal emission)."""
    from unittest.mock import patch

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

    # Inject mocks
    integration_watcher._triage_agent = mock_triage_agent
    integration_watcher._workflow = mock_workflow

    # Patch _emit_signal to suppress actual signal emission/side effects
    with patch.object(integration_watcher, "_emit_signal") as mock_emit:
        # Run single cycle
        await integration_watcher._run_cycle()

    # Verify: fetch → triage → analyze
    mock_multi_source_fetchers[0].afetch_market_news.assert_called_once()
    mock_triage_agent.analyze.assert_called_once()
    mock_workflow.analyze.assert_called_once_with("AAPL", period_days=30)

    # Verify signal would have been emitted
    mock_emit.assert_called_once()
    signal_arg = mock_emit.call_args[0][0]
    assert isinstance(signal_arg, EventSignal)
    assert signal_arg.triage.relevance == 0.9
    assert "AAPL" in signal_arg.analyses


@pytest.mark.integration
@pytest.mark.asyncio
async def test_multi_source_deduplication_integration(mock_historical_cache):
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


@pytest.mark.integration
@pytest.mark.asyncio
async def test_cooldown_prevents_duplicate_analysis(
    integration_watcher, mock_multi_source_fetchers, mock_triage_agent, mock_workflow
):
    """Test cooldown prevents re-analyzing same symbol."""
    now = datetime.now(UTC)
    article = NewsArticle(
        title="Breaking: Important news",
        description="Critical announcement",
        url="https://example.com/1",
        published_at=now - timedelta(minutes=5),
        source="Bloomberg",
    )

    mock_multi_source_fetchers[0].afetch_market_news.return_value = [article]
    integration_watcher._triage_agent = mock_triage_agent
    integration_watcher._workflow = mock_workflow

    # First cycle - should analyze
    await integration_watcher._run_cycle()
    assert mock_workflow.analyze.call_count == 1

    # Second cycle - same symbol in cooldown
    article2 = NewsArticle(
        title="Breaking: Another news",
        description="Another announcement",
        url="https://example.com/2",  # Different URL
        published_at=now - timedelta(minutes=3),
        source="Bloomberg",
    )
    mock_multi_source_fetchers[0].afetch_market_news.return_value = [article2]

    await integration_watcher._run_cycle()

    # Should not analyze again (AAPL in cooldown)
    assert mock_workflow.analyze.call_count == 1  # Still 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_concurrent_event_processing(
    integration_watcher, mock_multi_source_fetchers, mock_triage_agent, mock_workflow
):
    """Test concurrent analysis with semaphore enforcement."""
    now = datetime.now(UTC)

    # Create 5 breaking articles for different symbols
    articles = [
        NewsArticle(
            title=f"Breaking: {symbol} announcement",
            description=f"{symbol} news",
            url=f"https://example.com/{i}",
            published_at=now - timedelta(minutes=5),
            source="Bloomberg",
        )
        for i, symbol in enumerate(["AAPL", "TSLA", "GOOGL", "MSFT", "AMZN"])
    ]

    mock_multi_source_fetchers[0].afetch_market_news.return_value = articles

    # Mock triage to return different symbols
    async def triage_side_effect(event):
        # Extract symbol from title
        title = event.article.title
        symbol = title.split()[1]
        return TriageResult(
            event_id=event.event_id,
            event_type=event.event_type,
            relevance=0.9,
            urgency=Urgency.IMMEDIATE,
            sentiment=Sentiment.BULLISH,
            symbols=[symbol],
            confidence=0.85,
            reasoning="Test reasoning",
        )

    mock_triage_agent.analyze.side_effect = triage_side_effect
    integration_watcher._triage_agent = mock_triage_agent
    integration_watcher._workflow = mock_workflow

    # Track concurrent calls
    max_concurrent = 0
    current_concurrent = 0
    lock = asyncio.Lock()

    # Create result to return
    result = TradingWorkflowResult.model_construct(
        symbol="AAPL",  # Will be overridden
        decision=TradingDecision.model_construct(
            action=Signal.BUY,
            confidence=0.85,
            reasoning=["Test"],
            risk_level="LOW",
        ),
        technical=Mock(),
        sentiment=Mock(),
        news=Mock(),
        bullish=Mock(),
        bearish=Mock(),
        risk=Mock(),
    )

    async def track_concurrent(symbol, *args, **kwargs):
        nonlocal max_concurrent, current_concurrent
        async with lock:
            current_concurrent += 1
            max_concurrent = max(max_concurrent, current_concurrent)

        await asyncio.sleep(0.01)  # Simulate work

        async with lock:
            current_concurrent -= 1

        # Return result with correct symbol
        return TradingWorkflowResult.model_construct(
            symbol=symbol,
            decision=result.decision,
            technical=result.technical,
            sentiment=result.sentiment,
            news=result.news,
            bullish=result.bullish,
            bearish=result.bearish,
            risk=result.risk,
        )

    mock_workflow.analyze.side_effect = track_concurrent

    await integration_watcher._run_cycle()

    # Should respect max_concurrent_analyses=2
    assert max_concurrent <= 2
    assert mock_workflow.analyze.call_count == 2  # Limited by max_concurrent_analyses


@pytest.mark.integration
@pytest.mark.asyncio
async def test_fetcher_failure_graceful_degradation(mock_historical_cache):
    """Test graceful degradation when one fetcher fails."""
    import httpx

    now = datetime.now(UTC)

    # Marketaux fails
    mock_marketaux = AsyncMock(spec=BaseNewsFetcher)
    mock_marketaux.get_source_name.return_value = "marketaux"
    mock_marketaux.afetch_market_news.side_effect = httpx.HTTPError("API down")

    # Finnhub succeeds
    mock_finnhub = AsyncMock(spec=BaseNewsFetcher)
    mock_finnhub.get_source_name.return_value = "finnhub"
    mock_finnhub.afetch_market_news.return_value = [
        NewsArticle(
            title="Breaking: Finnhub article",
            description="Announcement",
            url="https://example.com/finnhub",
            published_at=now - timedelta(minutes=5),
            source="Finnhub",
        )
    ]

    watcher = NewsWatcher(
        historical_cache=mock_historical_cache,
        fetchers=[mock_marketaux, mock_finnhub],
        config=NewsWatcherConfig(),
    )

    events = await watcher._fetch_events()

    # Should get finnhub article despite marketaux failure
    assert len(events) == 1
    assert events[0].source == "finnhub"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_triage_partial_failure_continues(
    integration_watcher, mock_multi_source_fetchers, mock_triage_agent, mock_workflow
):
    """Test cycle continues when some triage calls fail."""
    now = datetime.now(UTC)

    # Create 3 articles
    articles = [
        NewsArticle(
            title=f"Breaking: Article {i}",
            description=f"Description {i}",
            url=f"https://example.com/{i}",
            published_at=now - timedelta(minutes=5),
            source="Bloomberg",
        )
        for i in range(3)
    ]

    mock_multi_source_fetchers[0].afetch_market_news.return_value = articles

    # Mock triage: first 2 succeed, third fails
    call_count = 0

    async def triage_with_failure(event):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            return TriageResult(
                event_id=event.event_id,
                event_type=event.event_type,
                relevance=0.9,
                urgency=Urgency.IMMEDIATE,
                sentiment=Sentiment.BULLISH,
                symbols=["AAPL"],
                confidence=0.85,
                reasoning="Test reasoning",
            )
        raise ValueError("Triage failed")

    mock_triage_agent.analyze.side_effect = triage_with_failure
    integration_watcher._triage_agent = mock_triage_agent
    integration_watcher._workflow = mock_workflow

    # Should not raise exception
    await integration_watcher._run_cycle()

    # Should have processed 2 successful triages
    assert mock_triage_agent.analyze.call_count == 3
    assert mock_workflow.analyze.call_count == 1  # One symbol analyzed


@pytest.mark.integration
@pytest.mark.asyncio
async def test_watcher_shutdown_stops_polling():
    """Test watcher stops cleanly when running flag set to False."""
    from unittest.mock import patch

    mock_cache = Mock(spec=HistoricalCache)
    watcher = NewsWatcher(
        historical_cache=mock_cache,
        config=NewsWatcherConfig(poll_interval=1),
    )

    # Mock _run_cycle to avoid real work
    cycle_calls = []

    async def mock_cycle():
        cycle_calls.append(1)
        await asyncio.sleep(0.01)

    with patch.object(watcher, "_run_cycle", side_effect=mock_cycle):
        # Start watcher as background task
        watcher_task = asyncio.create_task(watcher.run())

        # Let it run 2 cycles
        await asyncio.sleep(0.1)

        # Stop watcher
        watcher.running = False

        # Wait for shutdown
        await asyncio.wait_for(watcher_task, timeout=2.0)

    # Should have completed some cycles but stopped cleanly
    assert len(cycle_calls) >= 1
    assert watcher_task.done()
    assert not watcher_task.cancelled()
