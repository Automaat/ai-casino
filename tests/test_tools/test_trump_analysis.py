"""Tests for Trump analysis tool."""

from unittest.mock import MagicMock, patch

import pytest

from src.tools.trump_analysis import TrumpAnalysisTool


class TestTrumpAnalysisTool:
    """Tests for TrumpAnalysisTool."""

    def test_name(self):
        """Test tool name."""
        tool = TrumpAnalysisTool()
        assert tool.name == "analyze_trump_posts"

    def test_tool_definition(self):
        """Test tool definition structure."""
        tool = TrumpAnalysisTool()
        definition = tool.get_tool_definition()

        assert definition["type"] == "function"
        assert definition["function"]["name"] == "analyze_trump_posts"
        assert "Truth Social" in definition["function"]["description"]
        assert "days" in definition["function"]["parameters"]["properties"]

    def test_does_not_require_confirmation(self):
        """Test that tool does not require confirmation."""
        tool = TrumpAnalysisTool()
        assert tool.requires_confirmation is False

    def test_execute_no_posts(self):
        """Test execute with no posts found."""
        tool = TrumpAnalysisTool()

        with patch("src.data.truth_social.TruthSocialFetcher") as mock_fetcher_cls:
            mock_fetcher = MagicMock()
            mock_fetcher.fetch_recent.return_value = MagicMock(posts=[])
            mock_fetcher_cls.return_value = mock_fetcher

            result = tool.execute(days=3)

            assert "No Trump posts found" in result

    def test_execute_clamps_days(self):
        """Test that days parameter is clamped to 1-7."""
        tool = TrumpAnalysisTool()

        with patch("src.data.truth_social.TruthSocialFetcher") as mock_fetcher_cls:
            mock_fetcher = MagicMock()
            mock_fetcher.fetch_recent.return_value = MagicMock(posts=[])
            mock_fetcher_cls.return_value = mock_fetcher

            tool.execute(days=10)
            mock_fetcher.fetch_recent.assert_called_with(hours=168)  # 7 * 24

            tool.execute(days=0)
            mock_fetcher.fetch_recent.assert_called_with(hours=24)  # 1 * 24

    def test_execute_handles_error(self):
        """Test execute handles errors gracefully."""
        tool = TrumpAnalysisTool()

        with patch("src.data.truth_social.TruthSocialFetcher") as mock_fetcher_cls:
            mock_fetcher_cls.side_effect = Exception("API error")

            result = tool.execute(days=3)

            assert "Failed to analyze Trump posts" in result

    def test_repr(self):
        """Test string representation."""
        tool = TrumpAnalysisTool()
        assert repr(tool) == "TrumpAnalysisTool()"


class TestFormatAnalysis:
    """Tests for analysis formatting."""

    @pytest.fixture
    def mock_analysis(self):
        """Create mock analysis result."""
        from src.agents.trump import TrumpAnalysis
        from src.strategies.momentum import Signal

        return TrumpAnalysis(
            market_relevant=True,
            signal=Signal.BUY,
            mentioned_tickers=["TSLA", "AAPL"],
            sentiment="positive",
            confidence=0.75,
            key_phrases=["great for the economy", "buy stocks"],
            interpretation="Posts suggest bullish sentiment.",
            post_count=5,
        )

    def test_format_analysis_structure(self, mock_analysis):
        """Test formatted analysis contains all sections."""
        tool = TrumpAnalysisTool()
        result = tool._format_analysis(mock_analysis, days=3)

        assert "# Trump Analysis (Last 3 Days)" in result
        assert "**Posts Analyzed:** 5" in result
        assert "**Market Relevant:** Yes" in result
        assert "**Signal:** BUY" in result
        assert "**Confidence:** 75%" in result
        assert "**Sentiment:** positive" in result
        assert "$TSLA" in result
        assert "$AAPL" in result
        assert "great for the economy" in result
        assert "Posts suggest bullish sentiment." in result
