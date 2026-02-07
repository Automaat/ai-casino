"""Tests for EventTriageAgent."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock

import pytest

from src.agents.event_triage import EventTriageAgent, TriageLLMResponse
from src.daemon.events import NewsEvent, Sentiment, Urgency
from src.data.news import NewsArticle
from src.models.providers.base import StructuredOutputError


@pytest.fixture
def mock_llm_client():
    """Mock LLM client."""
    client = Mock()
    client.astructured = AsyncMock()
    return client


@pytest.fixture
def sample_news_event():
    """Sample news event for testing."""
    article = NewsArticle(
        title="Apple announces record earnings beat",
        description="Apple Inc reports quarterly earnings above expectations",
        source="MarketWatch",
        published_at=datetime.now(UTC),
        url="https://example.com/apple-earnings",
    )
    return NewsEvent(
        event_id="test-event-1",
        event_type="news",
        timestamp=datetime.now(UTC),
        source="marketaux",
        article=article,
    )


@pytest.mark.asyncio
async def test_event_triage_agent_init(mock_llm_client):
    """Test EventTriageAgent initialization."""
    agent = EventTriageAgent(mock_llm_client)

    assert agent.llm == mock_llm_client
    assert agent._prompts is not None


@pytest.mark.asyncio
async def test_analyze_successful_triage(mock_llm_client, sample_news_event):
    """Test successful event triage."""
    llm_response = TriageLLMResponse(
        relevance=0.85,
        symbols=["AAPL"],
        urgency="IMMEDIATE",
        sentiment="BULLISH",
        confidence=0.9,
        reasoning="Strong earnings beat for Apple",
    )
    mock_llm_client.astructured.return_value = llm_response

    agent = EventTriageAgent(mock_llm_client)
    result = await agent.analyze(sample_news_event)

    assert result.event_id == "test-event-1"
    assert result.event_type == "news"
    assert result.relevance == 0.85
    assert result.symbols == ["AAPL"]
    assert result.urgency == Urgency.IMMEDIATE
    assert result.sentiment == Sentiment.BULLISH
    assert result.confidence == 0.9
    assert "earnings" in result.reasoning.lower()

    mock_llm_client.astructured.assert_called_once()


@pytest.mark.asyncio
async def test_analyze_normalizes_symbols(mock_llm_client, sample_news_event):
    """Test that symbols are normalized to uppercase."""
    llm_response = TriageLLMResponse(
        relevance=0.7,
        symbols=["aapl", "msft"],  # lowercase
        urgency="WATCHLIST",
        sentiment="NEUTRAL",
        confidence=0.6,
        reasoning="Minor news",
    )
    mock_llm_client.astructured.return_value = llm_response

    agent = EventTriageAgent(mock_llm_client)
    result = await agent.analyze(sample_news_event)

    assert result.symbols == ["AAPL", "MSFT"]  # Normalized to uppercase


@pytest.mark.asyncio
async def test_analyze_fallback_on_structured_output_error(mock_llm_client, sample_news_event):
    """Test fallback behavior when structured output fails."""
    mock_llm_client.astructured.side_effect = StructuredOutputError("Parse failed")

    agent = EventTriageAgent(mock_llm_client)
    result = await agent.analyze(sample_news_event)

    # Should return low relevance with IGNORE urgency
    assert result.relevance == 0.3
    assert result.urgency == Urgency.IGNORE
    assert result.confidence == 0.0
    assert result.symbols == []
    assert "Triage failed" in result.reasoning


@pytest.mark.asyncio
async def test_analyze_ignore_urgency(mock_llm_client, sample_news_event):
    """Test event with IGNORE urgency."""
    llm_response = TriageLLMResponse(
        relevance=0.2,
        symbols=[],
        urgency="IGNORE",
        sentiment="NEUTRAL",
        confidence=0.5,
        reasoning="Not trading-relevant",
    )
    mock_llm_client.astructured.return_value = llm_response

    agent = EventTriageAgent(mock_llm_client)
    result = await agent.analyze(sample_news_event)

    assert result.urgency == Urgency.IGNORE
    assert result.relevance == 0.2


def test_repr(mock_llm_client):
    """Test string representation."""
    agent = EventTriageAgent(mock_llm_client)
    repr_str = repr(agent)

    assert "EventTriageAgent" in repr_str
    assert "llm=" in repr_str
