"""Tests for GetSocialSentimentTool."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.tools.social_sentiment import GetSocialSentimentTool


@pytest.fixture
def tool():
    """Create GetSocialSentimentTool."""
    return GetSocialSentimentTool()


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

    def test_name(self, tool):
        """Test tool name."""
        assert tool.name == "get_social_sentiment"

    def test_requires_confirmation(self, tool):
        """Test that tool doesn't require confirmation."""
        assert tool.requires_confirmation is False

    def test_get_tool_definition(self, tool):
        """Test tool definition format."""
        definition = tool.get_tool_definition()

        assert definition["type"] == "function"
        assert definition["function"]["name"] == "get_social_sentiment"
        assert "description" in definition["function"]

        params = definition["function"]["parameters"]
        assert "symbol" in params["properties"]
        assert "symbol" in params["required"]

    def test_execute_success(self, tool, mock_analysis):
        """Test successful execution."""
        with (
            patch("src.models.llm.LLMClient"),
            patch("src.data.finnhub.FinnhubFetcher"),
            patch("src.data.reddit.RedditFetcher"),
            patch("src.models.sentiment.get_finbert_sentiment"),
            patch("src.agents.social.SocialSentimentAnalyst") as mock_analyst_cls,
        ):
            mock_instance = MagicMock()
            mock_instance.analyze = AsyncMock(return_value=mock_analysis)
            mock_analyst_cls.return_value = mock_instance

            result = tool.execute("AAPL")

            assert "AAPL" in result
            assert "BULLISH" in result
            assert "0.65" in result
            assert "rising" in result
            assert "42" in result
            mock_instance.analyze.assert_called_once_with("AAPL")

    def test_execute_uppercase_symbol(self, tool, mock_analysis):
        """Test that symbol is uppercased."""
        with (
            patch("src.models.llm.LLMClient"),
            patch("src.data.finnhub.FinnhubFetcher"),
            patch("src.data.reddit.RedditFetcher"),
            patch("src.models.sentiment.get_finbert_sentiment"),
            patch("src.agents.social.SocialSentimentAnalyst") as mock_analyst_cls,
        ):
            mock_instance = MagicMock()
            mock_instance.analyze = AsyncMock(return_value=mock_analysis)
            mock_analyst_cls.return_value = mock_instance

            tool.execute("aapl")

            mock_instance.analyze.assert_called_once_with("AAPL")

    def test_execute_error_handling(self, tool):
        """Test error handling on failure."""
        with (
            patch("src.models.llm.LLMClient", side_effect=Exception("API error")),
        ):
            result = tool.execute("INVALID")

            assert "Social sentiment analysis failed" in result
            assert "API error" in result

    def test_format_result_none_sentiments(self, tool):
        """Test formatting with None sentiment values."""
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

    def test_repr(self, tool):
        """Test string representation."""
        repr_str = repr(tool)
        assert "GetSocialSentimentTool" in repr_str
