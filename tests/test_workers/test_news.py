"""Tests for news worker."""

from src.tools.models import ToolDefinition
from src.workers.news import NewsAnalysis


def test_news_worker_init(test_container):
    """Test worker initialization."""
    worker = test_container.news_worker()

    assert worker.llm is not None
    assert worker._prompts is not None
    tool_def = worker.get_tool_definition()
    assert tool_def is not None


async def test_news_worker_analyze_with_articles(test_container, sample_news_articles):
    """Test analysis with news articles."""
    worker = test_container.news_worker()

    result = await worker.analyze("AAPL", sample_news_articles)

    assert isinstance(result, NewsAnalysis)
    assert len(result.key_themes) > 0
    assert result.impact_assessment
    assert result.recommendation


async def test_news_worker_analyze_empty_articles(test_container):
    """Test analysis with no articles returns appropriate defaults."""
    worker = test_container.news_worker()

    result = await worker.analyze("AAPL", [])

    assert isinstance(result, NewsAnalysis)
    assert result.key_themes == ["No recent news"]
    assert "Insufficient data" in result.impact_assessment
    assert "Wait for more information" in result.recommendation


async def test_news_worker_structured_output_fallback(test_container, sample_news_articles):
    """Test fallback to text parsing when structured output fails."""
    from unittest.mock import AsyncMock

    from src.models.providers.base import StructuredOutputError

    worker = test_container.news_worker()

    # Mock astructured to raise error, acomplete to return text
    worker.llm.astructured = AsyncMock(side_effect=StructuredOutputError("Test error"))
    worker.llm.acomplete = AsyncMock(
        return_value="""
        Key themes:
        1. Market expansion
        2. Revenue growth
        3. Competitive positioning

        Impact assessment: Positive market impact expected

        Recommendation: Consider buying on strength
        """
    )

    result = await worker.analyze("AAPL", sample_news_articles)

    assert isinstance(result, NewsAnalysis)
    assert len(result.key_themes) > 0
    assert result.impact_assessment
    assert result.recommendation
    worker.llm.astructured.assert_called_once()
    worker.llm.acomplete.assert_called_once()


async def test_news_worker_theme_extraction(test_container):
    """Test theme extraction from text response."""
    worker = test_container.news_worker()

    response = """
    Key themes identified:
    1. Strong earnings growth
    2. Market expansion initiatives
    3. Product innovation
    - Increased competition
    • Regulatory challenges
    """

    themes = worker._extract_themes(response)

    assert isinstance(themes, list)
    assert len(themes) <= 5
    for theme in themes:
        assert len(theme) >= 5
        assert len(theme) <= 100


async def test_news_worker_section_extraction(test_container):
    """Test section extraction from text response."""
    worker = test_container.news_worker()

    response = """
    Key themes: Revenue growth, Market expansion

    Impact assessment: Strong positive impact on stock price expected due to
    excellent fundamentals and market positioning.

    Recommendation: Buy on current strength with confidence
    """

    impact = worker._extract_section(response, "impact")
    recommendation = worker._extract_section(response, "recommendation")

    assert isinstance(impact, str)
    assert len(impact) > 0
    assert isinstance(recommendation, str)
    assert len(recommendation) > 0


def test_news_worker_tool_definition(test_container):
    """Test tool definition for supervisor integration."""
    worker = test_container.news_worker()

    tool_def = worker.get_tool_definition()

    assert isinstance(tool_def, ToolDefinition)
    assert tool_def.type == "function"
    assert tool_def.function.name == "analyze_news"
    assert "symbol" in tool_def.function.parameters.properties
    assert "symbol" in tool_def.function.parameters.required


def test_news_worker_repr(test_container):
    """Test string representation."""
    worker = test_container.news_worker()

    repr_str = repr(worker)

    assert "NewsWorker" in repr_str
    assert "llm=" in repr_str
