"""Tests for FetchMarketContextTool."""

from unittest.mock import MagicMock

import pytest

from src.tools.game_plan.market_context import FetchMarketContextTool


@pytest.fixture
def mock_fetcher() -> MagicMock:
    """Create mock market fetcher."""
    return MagicMock()


class TestFetchMarketContextTool:
    """Tests for FetchMarketContextTool."""

    @pytest.mark.unit
    def test_formats_futures_data(self, mock_fetcher: MagicMock) -> None:
        """Formats futures changes as markdown."""
        mock_fetcher.fetch_overnight_futures.return_value = {"ES=F": 0.45, "NQ=F": -0.32}
        tool = FetchMarketContextTool(mock_fetcher)

        result = tool.execute()

        assert "ES=F" in result
        assert "+0.45%" in result
        assert "NQ=F" in result
        assert "-0.32%" in result
        assert "up" in result
        assert "down" in result

    @pytest.mark.unit
    def test_handles_empty_data(self, mock_fetcher: MagicMock) -> None:
        """Returns unavailable message when no data."""
        mock_fetcher.fetch_overnight_futures.return_value = {}
        tool = FetchMarketContextTool(mock_fetcher)

        result = tool.execute()

        assert "unavailable" in result.lower()

    @pytest.mark.unit
    def test_custom_symbols(self, mock_fetcher: MagicMock) -> None:
        """Passes custom symbols to fetcher."""
        mock_fetcher.fetch_overnight_futures.return_value = {"YM=F": 0.1}
        tool = FetchMarketContextTool(mock_fetcher)

        tool.execute(symbols="YM=F,RTY=F")

        mock_fetcher.fetch_overnight_futures.assert_called_once_with(["YM=F", "RTY=F"])

    @pytest.mark.unit
    def test_tool_definition(self, mock_fetcher: MagicMock) -> None:
        """Tool definition has correct name."""
        tool = FetchMarketContextTool(mock_fetcher)
        defn = tool.get_tool_definition()

        assert defn.function.name == "fetch_market_context"
