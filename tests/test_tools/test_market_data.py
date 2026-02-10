"""Tests for GetMarketDataTool."""

from datetime import datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.tools.market_data import GetMarketDataTool


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

    def test_name(self, test_container_full):
        """Test tool name."""
        tool = GetMarketDataTool(container=test_container_full)
        assert tool.name == "get_market_data"

    def test_requires_confirmation(self, test_container_full):
        """Test that tool doesn't require confirmation."""
        tool = GetMarketDataTool(container=test_container_full)
        assert tool.requires_confirmation is False

    def test_get_tool_definition(self, test_container_full):
        """Test tool definition format."""
        tool = GetMarketDataTool(container=test_container_full)
        definition = tool.get_tool_definition()

        assert definition["type"] == "function"
        assert definition["function"]["name"] == "get_market_data"
        assert "description" in definition["function"]

        params = definition["function"]["parameters"]
        assert "symbol" in params["properties"]
        assert "days" in params["properties"]
        assert "symbol" in params["required"]

    def test_execute_success(self, test_container_full, sample_market_data):
        """Test successful execution."""
        tool = GetMarketDataTool(container=test_container_full)

        mock_fetcher = MagicMock()
        mock_fetcher.fetch_daily.return_value = sample_market_data
        test_container_full.market_fetcher.override(mock_fetcher)

        result = tool.execute(symbol="AAPL", days=30)

        assert "AAPL" in result
        assert "Current Price:" in result
        assert "Change:" in result
        mock_fetcher.fetch_daily.assert_called_once_with("AAPL", 30)

    def test_execute_default_days(self, test_container_full, sample_market_data):
        """Test execution with default days."""
        tool = GetMarketDataTool(container=test_container_full)

        mock_fetcher = MagicMock()
        mock_fetcher.fetch_daily.return_value = sample_market_data
        test_container_full.market_fetcher.override(mock_fetcher)

        tool.execute(symbol="AAPL")

        mock_fetcher.fetch_daily.assert_called_once_with("AAPL", 30)

    def test_execute_uppercase_symbol(self, test_container_full, sample_market_data):
        """Test that symbol is uppercased."""
        tool = GetMarketDataTool(container=test_container_full)

        mock_fetcher = MagicMock()
        mock_fetcher.fetch_daily.return_value = sample_market_data
        test_container_full.market_fetcher.override(mock_fetcher)

        tool.execute(symbol="aapl", days=30)

        mock_fetcher.fetch_daily.assert_called_once_with("AAPL", 30)

    def test_execute_error_handling(self, test_container_full):
        """Test error handling on fetch failure."""
        tool = GetMarketDataTool(container=test_container_full)

        mock_fetcher = MagicMock()
        mock_fetcher.fetch_daily.side_effect = Exception("API error")
        test_container_full.market_fetcher.override(mock_fetcher)

        result = tool.execute(symbol="INVALID")

        assert "Failed to fetch market data" in result
        assert "API error" in result

    def test_format_data_content(self, test_container_full, sample_market_data):
        """Test formatted data content."""
        tool = GetMarketDataTool(container=test_container_full)
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

    def test_repr(self, test_container_full):
        """Test string representation."""
        tool = GetMarketDataTool(container=test_container_full)
        repr_str = repr(tool)
        assert "GetMarketDataTool" in repr_str
