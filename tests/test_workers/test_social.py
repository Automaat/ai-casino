"""Tests for social sentiment worker - ApeWisdom integration."""

from unittest.mock import AsyncMock, Mock

import pytest

from src.agents.social import SocialSentimentAnalysis
from src.data.apewisdom import ApeWisdomFetcher, ApeWisdomTicker
from src.tools.models import ToolDefinition
from src.workers.social import SocialSentimentWorker


@pytest.fixture
def mock_apewisdom_ticker():
    """ApeWisdom ticker with 24h history."""
    return ApeWisdomTicker(
        rank=5,
        ticker="AAPL",
        name="Apple Inc",
        mentions=150,
        upvotes=80,
        rank_24h_ago=8,
        mentions_24h_ago=100,
    )


@pytest.fixture
def mock_apewisdom_ticker_new():
    """ApeWisdom ticker with no 24h history (newly trending)."""
    return ApeWisdomTicker(
        rank=15,
        ticker="AAPL",
        name="Apple Inc",
        mentions=30,
        upvotes=15,
        rank_24h_ago=0,
        mentions_24h_ago=0,
    )


@pytest.fixture
def mock_apewisdom_ticker_null_history():
    """ApeWisdom ticker with None 24h history (API returned null)."""
    return ApeWisdomTicker(
        rank=5,
        ticker="AAPL",
        name="Apple Inc",
        mentions=50,
        upvotes=20,
        rank_24h_ago=8,
        mentions_24h_ago=None,
    )


@pytest.fixture
def mock_social_worker(test_container):
    """Create SocialSentimentWorker with mocked dependencies."""
    mock_apewisdom = Mock(spec=ApeWisdomFetcher)
    mock_apewisdom.get_ticker.return_value = None
    return SocialSentimentWorker(
        llm_client=test_container.llm_client(),
        finnhub_fetcher=Mock(),
        reddit_fetcher=Mock(),
        finbert=test_container.finbert_sentiment(),
        apewisdom=mock_apewisdom,
    )


def test_social_worker_init(mock_social_worker):
    """Test worker initialization."""
    assert mock_social_worker.llm is not None
    assert mock_social_worker.finnhub is not None
    assert mock_social_worker.reddit is not None
    assert mock_social_worker.finbert is not None
    assert mock_social_worker.apewisdom is not None


def test_social_worker_tool_definition(mock_social_worker):
    """Test tool definition structure."""
    tool_def = mock_social_worker.get_tool_definition()

    assert isinstance(tool_def, ToolDefinition)
    assert tool_def.type == "function"
    assert tool_def.function.name == "analyze_social_sentiment"
    assert "symbol" in tool_def.function.parameters.properties
    assert "symbol" in tool_def.function.parameters.required


async def test_fetch_apewisdom_success(mock_social_worker, mock_apewisdom_ticker):
    """Test _fetch_apewisdom returns ticker when found."""
    mock_social_worker.apewisdom.get_ticker.return_value = mock_apewisdom_ticker

    result = await mock_social_worker._fetch_apewisdom("AAPL")

    assert result is mock_apewisdom_ticker
    mock_social_worker.apewisdom.get_ticker.assert_called_once_with("AAPL")


async def test_fetch_apewisdom_not_found(mock_social_worker):
    """Test _fetch_apewisdom returns None when ticker not in trending."""
    mock_social_worker.apewisdom.get_ticker.return_value = None

    result = await mock_social_worker._fetch_apewisdom("XYZ")

    assert result is None


async def test_fetch_apewisdom_exception_returns_none(mock_social_worker):
    """Test _fetch_apewisdom returns None on exception."""
    mock_social_worker.apewisdom.get_ticker.side_effect = RuntimeError("API error")

    result = await mock_social_worker._fetch_apewisdom("AAPL")

    assert result is None


def test_format_apewisdom_summary_not_trending(mock_social_worker):
    """Test summary when ticker not in ApeWisdom trending."""
    result = mock_social_worker._format_apewisdom_summary(None, None)

    assert result == "Not in ApeWisdom trending"


def test_format_apewisdom_summary_with_delta(mock_social_worker, mock_apewisdom_ticker):
    """Test summary shows delta percentage when 24h history available."""
    result = mock_social_worker._format_apewisdom_summary(mock_apewisdom_ticker, 50.0)

    assert "Rank #5" in result
    assert "150 mentions" in result
    assert "+50%" in result


def test_format_apewisdom_summary_negative_delta(mock_social_worker, mock_apewisdom_ticker):
    """Test summary shows negative delta when mentions declined."""
    result = mock_social_worker._format_apewisdom_summary(mock_apewisdom_ticker, -33.0)

    assert "-33%" in result


def test_format_apewisdom_summary_newly_trending(mock_social_worker, mock_apewisdom_ticker_new):
    """Test summary shows NEW when mentions_24h_ago is 0."""
    result = mock_social_worker._format_apewisdom_summary(mock_apewisdom_ticker_new, None)

    assert "Rank #15" in result
    assert "30 mentions" in result
    assert "NEW" in result


async def test_analyze_computes_delta_with_history(mock_social_worker, mock_apewisdom_ticker):
    """Test analyze() computes mention delta when 24h history exists."""
    mock_social_worker._fetch_all_sources = AsyncMock(return_value=(None, None, None))
    mock_social_worker._fetch_apewisdom = AsyncMock(return_value=mock_apewisdom_ticker)
    mock_social_worker._compute_reddit_sentiment = AsyncMock(return_value=None)
    mock_social_worker.llm.astructured = AsyncMock(
        return_value=Mock(
            interpretation="Bullish social sentiment.",
            sentiment_label="BULLISH",
            confidence_keywords=["strong"],
        )
    )

    result = await mock_social_worker.analyze("AAPL")

    assert isinstance(result, SocialSentimentAnalysis)
    assert result.apewisdom_rank == 5
    assert result.apewisdom_mentions == 150
    assert result.apewisdom_mention_delta_pct == pytest.approx(50.0)  # (150-100)/100*100


async def test_analyze_delta_none_when_no_24h_history(mock_social_worker, mock_apewisdom_ticker_new):
    """Test analyze() sets delta to None when mentions_24h_ago is 0 (newly trending)."""
    mock_social_worker._fetch_all_sources = AsyncMock(return_value=(None, None, None))
    mock_social_worker._fetch_apewisdom = AsyncMock(return_value=mock_apewisdom_ticker_new)
    mock_social_worker._compute_reddit_sentiment = AsyncMock(return_value=None)
    mock_social_worker.llm.astructured = AsyncMock(
        return_value=Mock(
            interpretation="New trending stock.",
            sentiment_label="NEUTRAL",
            confidence_keywords=[],
        )
    )

    result = await mock_social_worker.analyze("AAPL")

    assert isinstance(result, SocialSentimentAnalysis)
    assert result.apewisdom_rank == 15
    assert result.apewisdom_mentions == 30
    assert result.apewisdom_mention_delta_pct is None


async def test_analyze_apewisdom_fields_none_when_not_trending(mock_social_worker):
    """Test analyze() sets all apewisdom fields to None when not trending."""
    mock_social_worker._fetch_all_sources = AsyncMock(return_value=(None, None, None))
    mock_social_worker._fetch_apewisdom = AsyncMock(return_value=None)
    mock_social_worker._compute_reddit_sentiment = AsyncMock(return_value=None)
    mock_social_worker.llm.astructured = AsyncMock(
        return_value=Mock(
            interpretation="No social data.",
            sentiment_label="NEUTRAL",
            confidence_keywords=[],
        )
    )

    result = await mock_social_worker.analyze("UNKNOWN")

    assert isinstance(result, SocialSentimentAnalysis)
    assert result.apewisdom_rank is None
    assert result.apewisdom_mentions is None
    assert result.apewisdom_mention_delta_pct is None


async def test_analyze_delta_none_when_mentions_24h_ago_is_none(
    mock_social_worker, mock_apewisdom_ticker_null_history
):
    """Test analyze() sets delta to None when mentions_24h_ago is None."""
    mock_social_worker._fetch_all_sources = AsyncMock(return_value=(None, None, None))
    mock_social_worker._fetch_apewisdom = AsyncMock(return_value=mock_apewisdom_ticker_null_history)
    mock_social_worker._compute_reddit_sentiment = AsyncMock(return_value=None)
    mock_social_worker.llm.astructured = AsyncMock(
        return_value=Mock(
            interpretation="No history available.",
            sentiment_label="NEUTRAL",
            confidence_keywords=[],
        )
    )

    result = await mock_social_worker.analyze("AAPL")

    assert isinstance(result, SocialSentimentAnalysis)
    assert result.apewisdom_rank == 5
    assert result.apewisdom_mentions == 50
    assert result.apewisdom_mention_delta_pct is None


def test_format_apewisdom_summary_null_mentions_24h_ago(
    mock_social_worker, mock_apewisdom_ticker_null_history
):
    """Test summary shows NEW when mentions_24h_ago is None."""
    result = mock_social_worker._format_apewisdom_summary(mock_apewisdom_ticker_null_history, None)

    assert "Rank #5" in result
    assert "50 mentions" in result
    assert "NEW" in result


def test_social_worker_repr(mock_social_worker):
    """Test string representation."""
    repr_str = repr(mock_social_worker)

    assert "SocialSentimentWorker" in repr_str
