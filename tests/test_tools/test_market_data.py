"""Tests for GetMarketDataTool."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.tools.market_data import GetMarketDataTool


@pytest.fixture
def tool():
    """Create GetMarketDataTool."""
    return GetMarketDataTool()


@pytest.fixture
def sample_market_data():
    """Create sample MarketData object."""
    from src.data.market import MarketData

    dates = pd.date_range(end=datetime.now(), periods=30, freq="D")
    data = pd.DataFrame(
        {
            "Open": [150.0 + i for i in range(30)],
            "High": [152.0 + i for i in range(30)],
            "Low": [148.0 + i for i in range(30)],
            "Close": [151.0 + i for i in range(30)],
            "Volume": [1000000 + i * 10000 for i in range(30)],
        },
        index=dates,
    )

    return MarketData(
        symbol="AAPL",
        data=data,
        last_updated=datetime.now(),
    )


class TestGetMarketDataTool:
    """Tests for GetMarketDataTool."""

    def test_name(self, tool):
        """Test tool name."""
        assert tool.name == "get_market_data"

    def test_requires_confirmation(self, tool):
        """Test that tool doesn't require confirmation."""
        assert tool.requires_confirmation is False

    def test_get_tool_definition(self, tool):
        """Test tool definition format."""
        definition = tool.get_tool_definition()

        assert definition["type"] == "function"
        assert definition["function"]["name"] == "get_market_data"
        assert "description" in definition["function"]

        params = definition["function"]["parameters"]
        assert "symbol" in params["properties"]
        assert "days" in params["properties"]
        assert "symbol" in params["required"]

    def test_execute_success(self, tool, sample_market_data):
        """Test successful execution."""
        with patch("src.data.market.MarketDataFetcher") as mock_fetcher_cls:
            mock_instance = MagicMock()
            mock_instance.fetch_daily.return_value = sample_market_data
            mock_fetcher_cls.return_value = mock_instance

            result = tool.execute("AAPL", days=30)

            assert "AAPL" in result
            assert "Current Price:" in result
            assert "Change:" in result
            mock_instance.fetch_daily.assert_called_once_with("AAPL", 30)

    def test_execute_default_days(self, tool, sample_market_data):
        """Test execution with default days."""
        with patch("src.data.market.MarketDataFetcher") as mock_fetcher_cls:
            mock_instance = MagicMock()
            mock_instance.fetch_daily.return_value = sample_market_data
            mock_fetcher_cls.return_value = mock_instance

            tool.execute("AAPL")

            mock_instance.fetch_daily.assert_called_once_with("AAPL", 30)

    def test_execute_uppercase_symbol(self, tool, sample_market_data):
        """Test that symbol is uppercased."""
        with patch("src.data.market.MarketDataFetcher") as mock_fetcher_cls:
            mock_instance = MagicMock()
            mock_instance.fetch_daily.return_value = sample_market_data
            mock_fetcher_cls.return_value = mock_instance

            tool.execute("aapl", days=30)

            mock_instance.fetch_daily.assert_called_once_with("AAPL", 30)

    def test_execute_error_handling(self, tool):
        """Test error handling on fetch failure."""
        with patch("src.data.market.MarketDataFetcher") as mock_fetcher_cls:
            mock_instance = MagicMock()
            mock_instance.fetch_daily.side_effect = Exception("API error")
            mock_fetcher_cls.return_value = mock_instance

            result = tool.execute("INVALID")

            assert "Failed to fetch market data" in result
            assert "API error" in result

    def test_format_data_content(self, tool, sample_market_data):
        """Test formatted data content."""
        result = tool._format_data(sample_market_data)

        assert "# AAPL Market Data" in result
        assert "Current Price:" in result
        assert "Change:" in result
        assert "Today's Range" in result
        assert "Open:" in result
        assert "High:" in result
        assert "Low:" in result
        assert "Volume:" in result
        assert "-Day Summary" in result
        assert "Last updated:" in result

    def test_repr(self, tool):
        """Test string representation."""
        repr_str = repr(tool)
        assert "GetMarketDataTool" in repr_str
