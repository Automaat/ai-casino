"""Tests for AnalyzeStockTool."""

from unittest.mock import MagicMock

import pytest

from src.strategies.signal import Signal
from src.tools.analyze_stock import AnalyzeStockTool


@pytest.fixture
def mock_workflow_result():
    """Create mock TradingWorkflowResult."""
    result = MagicMock()
    result.symbol = "AAPL"
    result.decision.action = Signal.BUY
    result.decision.confidence = 0.85
    result.decision.reasoning = ["Strong technical signals", "Positive sentiment"]
    result.risk.validation.risk_level = "LOW"
    result.technical.signal = Signal.BUY
    result.technical.rsi = 45.5
    result.technical.macd_hist = 0.25
    result.technical.interpretation = "Bullish momentum"
    result.sentiment.overall_sentiment = "positive"
    result.sentiment.sentiment_score = 0.72
    result.sentiment.positive_ratio = 0.65
    result.sentiment.negative_ratio = 0.15
    result.sentiment.neutral_ratio = 0.20
    result.news.key_themes = ["earnings beat", "product launch"]
    result.news.impact_assessment = "Strong positive impact expected"
    result.news.recommendation = "Consider buying on positive momentum"
    result.warnings = []
    return result


class TestAnalyzeStockTool:
    """Tests for AnalyzeStockTool."""

    def test_name(self, test_container_full):
        """Test tool name."""
        tool = AnalyzeStockTool(container=test_container_full)
        assert tool.name == "analyze_stock"

    def test_requires_confirmation(self, test_container_full):
        """Test that tool requires confirmation."""
        tool = AnalyzeStockTool(container=test_container_full)
        assert tool.requires_confirmation is True

    def test_get_tool_definition(self, test_container_full):
        """Test tool definition format."""
        tool = AnalyzeStockTool(container=test_container_full)
        definition = tool.get_tool_definition()

        assert definition["type"] == "function"
        assert definition["function"]["name"] == "analyze_stock"
        assert "description" in definition["function"]
        assert "expensive" in definition["function"]["description"].lower()

        params = definition["function"]["parameters"]
        assert "symbol" in params["properties"]
        assert "period_days" in params["properties"]
        assert "symbol" in params["required"]

    def test_execute_success(self, test_container_full, mock_workflow_result):
        """Test successful execution."""
        from dependency_injector import providers

        tool = AnalyzeStockTool(container=test_container_full)

        mock_workflow = MagicMock()

        async def mock_analyze(symbol: str, period_days: int):
            return mock_workflow_result

        mock_workflow.analyze = mock_analyze

        # Override workflow_momentum provider (used by analyze_stock)
        test_container_full.workflow_momentum.override(providers.Factory(lambda **_kwargs: mock_workflow))

        result = tool.execute(symbol="AAPL", period_days=90)

        assert "AAPL" in result
        assert "BUY" in result
        assert "Technical Analysis" in result

    def test_execute_default_period(self, test_container_full, mock_workflow_result):
        """Test execution with default period."""
        from dependency_injector import providers

        tool = AnalyzeStockTool(container=test_container_full)

        mock_workflow = MagicMock()

        async def mock_analyze(symbol: str, period_days: int):
            return mock_workflow_result

        mock_workflow.analyze = mock_analyze

        test_container_full.workflow_momentum.override(providers.Factory(lambda **_kwargs: mock_workflow))

        result = tool.execute(symbol="AAPL")

        assert "AAPL" in result

    def test_execute_uppercase_symbol(self, test_container_full, mock_workflow_result):
        """Test that symbol is uppercased."""
        from dependency_injector import providers

        tool = AnalyzeStockTool(container=test_container_full)

        mock_workflow = MagicMock()

        async def mock_analyze(symbol: str, period_days: int):
            # Verify symbol is uppercased
            assert symbol == "AAPL"
            return mock_workflow_result

        mock_workflow.analyze = mock_analyze

        test_container_full.workflow_momentum.override(providers.Factory(lambda **_kwargs: mock_workflow))

        result = tool.execute(symbol="aapl", period_days=90)

        assert "AAPL" in result

    def test_execute_error_handling(self, test_container_full):
        """Test error handling on workflow failure."""
        from dependency_injector import providers

        tool = AnalyzeStockTool(container=test_container_full)

        mock_workflow = MagicMock()

        async def mock_analyze(symbol: str, period_days: int):
            msg = "Workflow error"
            raise RuntimeError(msg)

        mock_workflow.analyze = mock_analyze

        test_container_full.workflow_momentum.override(providers.Factory(lambda **_kwargs: mock_workflow))

        result = tool.execute(symbol="INVALID")

        assert "Analysis failed" in result
        assert "Workflow error" in result

    def test_format_result_content(self, test_container_full, mock_workflow_result):
        """Test formatted result content."""
        tool = AnalyzeStockTool(container=test_container_full)
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

        # Sentiment ratios
        assert "65%" in result  # positive_ratio
        assert "15%" in result  # negative_ratio
        assert "20%" in result  # neutral_ratio

        # News fields
        assert "Strong positive impact expected" in result  # impact_assessment
        assert "Consider buying on positive momentum" in result  # recommendation

        # Decision reasoning bullets
        assert "Strong technical signals" in result  # reasoning[0]
        assert "Positive sentiment" in result  # reasoning[1]

    def test_format_result_with_warnings(self, test_container_full, mock_workflow_result):
        """Test formatted result includes warnings."""
        tool = AnalyzeStockTool(container=test_container_full)
        mock_workflow_result.warnings = ["Fundamental data unavailable", "Rate limit hit"]

        result = tool._format_result(mock_workflow_result)

        assert "## Warnings" in result
        assert "Fundamental data unavailable" in result
        assert "Rate limit hit" in result

    def test_repr(self, test_container_full):
        """Test string representation."""
        tool = AnalyzeStockTool(container=test_container_full)
        repr_str = repr(tool)
        assert "AnalyzeStockTool" in repr_str
