"""Tests for WebResearchAgent."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.web_researcher import (
    ResearchCategory,
    WebResearchAgent,
    WebResearchAnalysis,
    WebResearchResult,
)


@pytest.fixture
def mock_search_tool():
    """Mock WebSearchTool."""
    mock = MagicMock()
    mock.TOOL_NAME = "web_search"
    mock.get_tool_definition.return_value = {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "search_type": {"type": "string", "enum": ["general", "news"]},
                },
                "required": ["query", "search_type"],
            },
        },
    }
    mock.execute.return_value = """Search results for 'AAPL stock latest news today' (news):

1. Apple Announces Record Earnings
   URL: https://example.com/earnings
   Apple reports quarterly revenue of $120B, beating expectations
   Source: Reuters

2. AAPL Stock Surges on Product News
   URL: https://example.com/stock
   Apple shares rise 5% following product announcement
   Source: Bloomberg"""
    return mock


MOCK_OLLAMA_RESPONSE = """SUMMARY: Apple stock shows strong momentum with bullish sentiment.

FINDINGS:
- Record quarterly revenue of $120B beat analyst expectations
- Stock price surged 5% on product news
- Analysts upgrading price targets
- Strong institutional buying interest

SENTIMENT: Bullish"""

MOCK_TOOLS_RESPONSE = """SUMMARY: Apple stock shows strong momentum with bullish sentiment.

FINDINGS:
- Record quarterly revenue of $120B beat analyst expectations
- Stock price surged 5% on product news
- Analysts upgrading price targets

SENTIMENT: Bullish"""


@pytest.fixture
def mock_llm_client_no_tools():
    """Mock LLM client without tool support (Ollama)."""
    mock = MagicMock()
    mock.provider = "ollama"
    mock.supports_tools = False
    mock.acomplete = AsyncMock(return_value=MOCK_OLLAMA_RESPONSE)
    return mock


@pytest.fixture
def mock_llm_client_with_tools():
    """Mock LLM client with tool support (Claude/OpenAI)."""
    mock = MagicMock()
    mock.provider = "anthropic"
    mock.supports_tools = True
    mock.acomplete_with_tools = AsyncMock(return_value=MOCK_TOOLS_RESPONSE)
    return mock


@pytest.fixture
def agent_no_tools(mock_llm_client_no_tools, mock_search_tool):
    """Create agent without tool calling support."""
    return WebResearchAgent(mock_llm_client_no_tools, mock_search_tool)


@pytest.fixture
def agent_with_tools(mock_llm_client_with_tools, mock_search_tool):
    """Create agent with tool calling support."""
    return WebResearchAgent(mock_llm_client_with_tools, mock_search_tool)


class TestWebResearchAgent:
    """Tests for WebResearchAgent."""

    async def test_research_no_tools(self, agent_no_tools, mock_llm_client_no_tools):
        """Test research with template-based queries (Ollama)."""
        result = await agent_no_tools.research("AAPL", categories=[ResearchCategory.LATEST_NEWS])

        assert isinstance(result, WebResearchAnalysis)
        assert result.symbol == "AAPL"
        assert len(result.results) == 1
        assert result.results[0].category == ResearchCategory.LATEST_NEWS
        mock_llm_client_no_tools.acomplete.assert_called_once()

    async def test_research_with_tools(self, agent_with_tools, mock_llm_client_with_tools):
        """Test research with tool calling (Claude/OpenAI)."""
        result = await agent_with_tools.research("AAPL", categories=[ResearchCategory.LATEST_NEWS])

        assert isinstance(result, WebResearchAnalysis)
        assert result.symbol == "AAPL"
        assert len(result.results) == 1
        mock_llm_client_with_tools.acomplete_with_tools.assert_called_once()

    async def test_research_with_tools_respects_max_calls(self, agent_with_tools, mock_llm_client_with_tools):
        """Test research with tool calling respects max_tool_calls=3."""
        await agent_with_tools.research("AAPL", categories=[ResearchCategory.LATEST_NEWS])

        # Verify acomplete_with_tools called with max_tool_calls=3
        call_args = mock_llm_client_with_tools.acomplete_with_tools.call_args
        assert call_args.kwargs["max_tool_calls"] == 3

    async def test_research_all_categories(self, agent_no_tools):
        """Test research with all categories."""
        result = await agent_no_tools.research("AAPL")

        assert len(result.results) == 4
        categories = {r.category for r in result.results}
        assert categories == set(ResearchCategory)

    async def test_research_result_parsing(self, agent_no_tools):
        """Test parsing of research results."""
        result = await agent_no_tools.research("AAPL", categories=[ResearchCategory.LATEST_NEWS])

        research_result = result.results[0]
        assert "Apple" in research_result.summary or "earnings" in research_result.summary.lower()
        assert len(research_result.key_findings) >= 1
        assert research_result.sentiment_indication in ["Bullish", "Bearish", "Neutral"]
        assert 0.0 <= research_result.confidence <= 1.0

    async def test_overall_sentiment_aggregation(self, agent_no_tools, mock_llm_client_no_tools):
        """Test sentiment aggregation across categories."""
        mock_llm_client_no_tools.acomplete = AsyncMock(
            return_value="""SUMMARY: Positive outlook.

FINDINGS:
- Strong performance
- Good metrics

SENTIMENT: Bullish"""
        )

        result = await agent_no_tools.research(
            "AAPL",
            categories=[
                ResearchCategory.LATEST_NEWS,
                ResearchCategory.MARKET_SENTIMENT,
            ],
        )

        assert result.overall_sentiment == "Bullish"

    async def test_neutral_sentiment_on_mixed(self, agent_no_tools, mock_llm_client_no_tools):
        """Test neutral sentiment when mixed."""
        responses = [
            """SUMMARY: Good news.\nFINDINGS:\n- Positive\nSENTIMENT: Bullish""",
            """SUMMARY: Bad news.\nFINDINGS:\n- Negative\nSENTIMENT: Bearish""",
        ]
        mock_llm_client_no_tools.acomplete = AsyncMock(side_effect=responses)

        result = await agent_no_tools.research(
            "AAPL",
            categories=[
                ResearchCategory.LATEST_NEWS,
                ResearchCategory.MARKET_SENTIMENT,
            ],
        )

        assert result.overall_sentiment == "Neutral"

    def test_extract_sentiment_bullish(self, agent_no_tools):
        """Test bullish sentiment extraction."""
        response = "SENTIMENT: Bullish"
        sentiment = agent_no_tools._extract_sentiment(response)
        assert sentiment == "Bullish"

    def test_extract_sentiment_bearish(self, agent_no_tools):
        """Test bearish sentiment extraction."""
        response = "SENTIMENT: bearish outlook"
        sentiment = agent_no_tools._extract_sentiment(response)
        assert sentiment == "Bearish"

    def test_extract_sentiment_neutral_fallback(self, agent_no_tools):
        """Test neutral sentiment as fallback."""
        response = "No clear sentiment"
        sentiment = agent_no_tools._extract_sentiment(response)
        assert sentiment == "Neutral"

    def test_extract_findings(self, agent_no_tools):
        """Test findings extraction."""
        response = """FINDINGS:
- First finding
- Second finding
- Third finding
SENTIMENT: Neutral"""

        findings = agent_no_tools._extract_findings(response)

        assert len(findings) == 3
        assert "First finding" in findings[0]

    def test_calculate_confidence(self, agent_no_tools):
        """Test confidence calculation."""
        findings = ["Finding 1 with enough detail", "Finding 2", "Finding 3"]
        confidence = agent_no_tools._calculate_confidence(findings, "Bullish")

        assert 0.5 <= confidence <= 1.0

    def test_repr(self, agent_no_tools):
        """Test string representation."""
        repr_str = repr(agent_no_tools)
        assert "WebResearchAgent" in repr_str
        assert "ollama" in repr_str


class TestWebResearchResult:
    """Tests for WebResearchResult model."""

    def test_create_result(self):
        """Test creating research result."""
        result = WebResearchResult(
            category=ResearchCategory.LATEST_NEWS,
            summary="Test summary",
            key_findings=["Finding 1", "Finding 2"],
            sentiment_indication="Bullish",
            confidence=0.75,
            sources_count=5,
        )

        assert result.category == ResearchCategory.LATEST_NEWS
        assert result.confidence == 0.75


class TestWebResearchAnalysis:
    """Tests for WebResearchAnalysis model."""

    def test_create_analysis(self):
        """Test creating analysis."""
        analysis = WebResearchAnalysis(
            symbol="AAPL",
            results=[
                WebResearchResult(
                    category=ResearchCategory.LATEST_NEWS,
                    summary="Test",
                    key_findings=["Finding"],
                    sentiment_indication="Bullish",
                    confidence=0.8,
                )
            ],
            overall_sentiment="Bullish",
            confidence=0.8,
        )

        assert analysis.symbol == "AAPL"
        assert len(analysis.results) == 1

    def test_repr(self):
        """Test string representation."""
        analysis = WebResearchAnalysis(
            symbol="AAPL",
            results=[],
            overall_sentiment="Neutral",
            confidence=0.5,
        )

        repr_str = repr(analysis)
        assert "AAPL" in repr_str
        assert "Neutral" in repr_str
