"""Tests for AnalyzeStockTool."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.strategies.signal import Signal
from src.tools.analyze_stock import AnalyzeStockTool


@pytest.fixture
def tool():
    """Create AnalyzeStockTool."""
    return AnalyzeStockTool()


@pytest.fixture
def mock_workflow_result():
    """Create mock TradingWorkflowResult."""
    result = MagicMock()
    result.symbol = "AAPL"
    result.decision.action = Signal.BUY
    result.decision.confidence = 0.85
    result.decision.rationale = "Strong technical and sentiment signals."
    result.risk.validation.risk_level = "LOW"
    result.technical.signal = Signal.BUY
    result.technical.rsi = 45.5
    result.technical.macd_hist = 0.25
    result.technical.interpretation = "Bullish momentum"
    result.sentiment.sentiment = "positive"
    result.sentiment.score = 0.72
    result.sentiment.confidence = 0.88
    result.news.overall_sentiment = "positive"
    result.news.key_themes = ["earnings beat", "product launch"]
    result.warnings = []
    return result


class TestAnalyzeStockTool:
    """Tests for AnalyzeStockTool."""

    def test_name(self, tool):
        """Test tool name."""
        assert tool.name == "analyze_stock"

    def test_requires_confirmation(self, tool):
        """Test that tool requires confirmation."""
        assert tool.requires_confirmation is True

    def test_get_tool_definition(self, tool):
        """Test tool definition format."""
        definition = tool.get_tool_definition()

        assert definition["type"] == "function"
        assert definition["function"]["name"] == "analyze_stock"
        assert "description" in definition["function"]
        assert "expensive" in definition["function"]["description"].lower()

        params = definition["function"]["parameters"]
        assert "symbol" in params["properties"]
        assert "period_days" in params["properties"]
        assert "symbol" in params["required"]

    def test_execute_success(self, tool, mock_workflow_result):
        """Test successful execution."""
        with (
            patch("src.models.llm.LLMClient"),
            patch("src.data.market.MarketDataFetcher"),
            patch("src.data.news.NewsFetcher"),
            patch("src.models.sentiment.FinBERTSentiment"),
            patch("src.data.fundamental.FundamentalDataFetcher"),
            patch("src.workflows.trading.TradingWorkflow") as mock_workflow_cls,
        ):
            mock_workflow = MagicMock()
            mock_workflow.analyze = AsyncMock(return_value=mock_workflow_result)
            mock_workflow_cls.return_value = mock_workflow

            result = tool.execute("AAPL", period_days=90)

            assert "AAPL" in result
            assert "BUY" in result
            assert "Technical Analysis" in result
            mock_workflow.analyze.assert_called_once_with("AAPL", 90)

    def test_execute_default_period(self, tool, mock_workflow_result):
        """Test execution with default period."""
        with (
            patch("src.models.llm.LLMClient"),
            patch("src.data.market.MarketDataFetcher"),
            patch("src.data.news.NewsFetcher"),
            patch("src.models.sentiment.FinBERTSentiment"),
            patch("src.data.fundamental.FundamentalDataFetcher"),
            patch("src.workflows.trading.TradingWorkflow") as mock_workflow_cls,
        ):
            mock_workflow = MagicMock()
            mock_workflow.analyze = AsyncMock(return_value=mock_workflow_result)
            mock_workflow_cls.return_value = mock_workflow

            tool.execute("AAPL")

            mock_workflow.analyze.assert_called_once_with("AAPL", 90)

    def test_execute_uppercase_symbol(self, tool, mock_workflow_result):
        """Test that symbol is uppercased."""
        with (
            patch("src.models.llm.LLMClient"),
            patch("src.data.market.MarketDataFetcher"),
            patch("src.data.news.NewsFetcher"),
            patch("src.models.sentiment.FinBERTSentiment"),
            patch("src.data.fundamental.FundamentalDataFetcher"),
            patch("src.workflows.trading.TradingWorkflow") as mock_workflow_cls,
        ):
            mock_workflow = MagicMock()
            mock_workflow.analyze = AsyncMock(return_value=mock_workflow_result)
            mock_workflow_cls.return_value = mock_workflow

            tool.execute("aapl", period_days=90)

            mock_workflow.analyze.assert_called_once_with("AAPL", 90)

    def test_execute_error_handling(self, tool):
        """Test error handling on workflow failure."""
        with (
            patch("src.models.llm.LLMClient"),
            patch("src.data.market.MarketDataFetcher"),
            patch("src.data.news.NewsFetcher"),
            patch("src.models.sentiment.FinBERTSentiment"),
            patch("src.data.fundamental.FundamentalDataFetcher"),
            patch("src.workflows.trading.TradingWorkflow") as mock_workflow_cls,
        ):
            mock_workflow = MagicMock()
            mock_workflow.analyze = AsyncMock(side_effect=Exception("Workflow error"))
            mock_workflow_cls.return_value = mock_workflow

            result = tool.execute("INVALID")

            assert "Analysis failed" in result
            assert "Workflow error" in result

    def test_format_result_content(self, tool, mock_workflow_result):
        """Test formatted result content."""
        result = tool._format_result(mock_workflow_result)

        assert "# AAPL Trading Analysis" in result
        assert "**Recommendation:** BUY" in result
        assert "**Confidence:** 85%" in result
        assert "**Risk Level:** LOW" in result
        assert "## Technical Analysis" in result
        assert "RSI:" in result
        assert "## Sentiment Analysis" in result
        assert "## News Analysis" in result
        assert "## Decision Rationale" in result

    def test_format_result_with_warnings(self, tool, mock_workflow_result):
        """Test formatted result includes warnings."""
        mock_workflow_result.warnings = ["Fundamental data unavailable", "Rate limit hit"]

        result = tool._format_result(mock_workflow_result)

        assert "## Warnings" in result
        assert "Fundamental data unavailable" in result
        assert "Rate limit hit" in result

    def test_repr(self, tool):
        """Test string representation."""
        repr_str = repr(tool)
        assert "AnalyzeStockTool" in repr_str
