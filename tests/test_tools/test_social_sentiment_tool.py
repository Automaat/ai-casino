"""Tests for GetSocialSentimentTool."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.tools.social_sentiment import GetSocialSentimentTool


@pytest.fixture
def mock_analysis():
    """Create mock SocialSentimentAnalysis."""
    analysis = MagicMock()
    analysis.sentiment_label = "BULLISH"
    analysis.overall_social_score = 0.65
    analysis.social_momentum = "rising"
    analysis.confidence = 0.82
    analysis.finnhub_sentiment = 0.45
    analysis.reddit_sentiment = 0.72
    analysis.wsb_mentions_24h = 42
    analysis.interpretation = "Strong bullish social sentiment across platforms."
    return analysis


class TestGetSocialSentimentTool:
    """Tests for GetSocialSentimentTool."""

    def test_name(self, test_container_full):
        """Test tool name."""
        tool = GetSocialSentimentTool(container=test_container_full)
        assert tool.name == "get_social_sentiment"

    def test_requires_confirmation(self, test_container_full):
        """Test that tool doesn't require confirmation."""
        tool = GetSocialSentimentTool(container=test_container_full)
        assert tool.requires_confirmation is False

    def test_get_tool_definition(self, test_container_full):
        """Test tool definition format."""
        tool = GetSocialSentimentTool(container=test_container_full)
        definition = tool.get_tool_definition().model_dump(mode="json", by_alias=True, exclude_none=True)

        assert definition["type"] == "function"
        assert definition["function"]["name"] == "get_social_sentiment"
        assert "description" in definition["function"]

        params = definition["function"]["parameters"]
        assert "symbol" in params["properties"]
        assert "symbol" in params["required"]

    def test_execute_success(self, test_container_full, mock_analysis):
        """Test successful execution."""
        from dependency_injector import providers

        from src.workers.social import SocialSentimentWorker

        tool = GetSocialSentimentTool(container=test_container_full)

        mock_analyst = MagicMock(spec=SocialSentimentWorker)
        mock_analyst.analyze = AsyncMock(return_value=mock_analysis)
        test_container_full.social_sentiment_worker.override(providers.Factory(lambda: mock_analyst))

        result = tool.execute(symbol="AAPL")

        assert "AAPL" in result
        assert "BULLISH" in result
        assert "0.65" in result
        assert "rising" in result
        assert "42" in result
        mock_analyst.analyze.assert_called_once_with("AAPL")

    def test_execute_uppercase_symbol(self, test_container_full, mock_analysis):
        """Test that symbol is uppercased."""
        from dependency_injector import providers

        from src.workers.social import SocialSentimentWorker

        tool = GetSocialSentimentTool(container=test_container_full)

        mock_analyst = MagicMock(spec=SocialSentimentWorker)
        mock_analyst.analyze = AsyncMock(return_value=mock_analysis)
        test_container_full.social_sentiment_worker.override(providers.Factory(lambda: mock_analyst))

        tool.execute(symbol="aapl")

        mock_analyst.analyze.assert_called_once_with("AAPL")

    def test_execute_error_handling(self, test_container_full):
        """Test error handling on failure."""
        from dependency_injector import providers

        from src.workers.social import SocialSentimentWorker

        tool = GetSocialSentimentTool(container=test_container_full)

        mock_analyst = MagicMock(spec=SocialSentimentWorker)
        mock_analyst.analyze = AsyncMock(side_effect=Exception("API error"))
        test_container_full.social_sentiment_worker.override(providers.Factory(lambda: mock_analyst))

        result = tool.execute(symbol="INVALID")

        assert "Social sentiment analysis failed" in result
        assert "API error" in result

    def test_format_result_none_sentiments(self, test_container_full):
        """Test formatting with None sentiment values."""
        tool = GetSocialSentimentTool(container=test_container_full)
        analysis = MagicMock()
        analysis.sentiment_label = "NEUTRAL"
        analysis.overall_social_score = 0.0
        analysis.social_momentum = "stable"
        analysis.confidence = 0.5
        analysis.finnhub_sentiment = None
        analysis.reddit_sentiment = None
        analysis.wsb_mentions_24h = 0
        analysis.interpretation = "No data available."

        result = tool._format_result("AAPL", analysis)

        assert "N/A" in result
        assert "NEUTRAL" in result

    def test_repr(self, test_container_full):
        """Test string representation."""
        tool = GetSocialSentimentTool(container=test_container_full)
        repr_str = repr(tool)
        assert "GetSocialSentimentTool" in repr_str
