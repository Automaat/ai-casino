"""Tests for GenerateTearsheetTool."""

from datetime import UTC, datetime
from unittest.mock import MagicMock, PropertyMock

import pytest

from src.tools.tearsheet import GenerateTearsheetTool


@pytest.fixture
def mock_tearsheet():
    """Create mock TearSheet."""
    ts = MagicMock()
    ts.symbol = "AAPL"
    ts.start_date = datetime(2023, 1, 1, tzinfo=UTC)
    ts.end_date = datetime(2024, 1, 1, tzinfo=UTC)
    ts.cagr = 0.1523
    ts.sharpe_ratio = 1.34
    ts.sortino_ratio = 1.89
    ts.calmar_ratio = 2.15
    ts.max_drawdown = -0.0712
    ts.volatility_annual = 0.1845
    ts.win_rate = 0.58
    ts.profit_factor = 1.67
    ts.benchmark_symbol = "SPY"
    ts.benchmark_cagr = 0.1234
    ts.benchmark_sharpe = 1.12
    ts.alpha = 0.0289
    ts.beta = 0.85
    ts.html_report_path = "/home/user/.ai-casino/tearsheets/AAPL_20240101.html"
    return ts


@pytest.fixture
def mock_trade():
    """Create mock closed trade."""
    trade = MagicMock()
    trade.symbol = "AAPL"
    trade.is_closed.return_value = True
    trade.timestamp = datetime.now(UTC)
    return trade


class TestGenerateTearsheetTool:
    """Tests for GenerateTearsheetTool."""

    def test_name(self, test_container_full):
        """Test tool name."""
        tool = GenerateTearsheetTool(container=test_container_full)
        assert tool.name == "generate_tearsheet"

    def test_requires_confirmation(self, test_container_full):
        """Test that tool doesn't require confirmation."""
        tool = GenerateTearsheetTool(container=test_container_full)
        assert tool.requires_confirmation is False

    def test_get_tool_definition(self, test_container_full):
        """Test tool definition format."""
        tool = GenerateTearsheetTool(container=test_container_full)
        definition = tool.get_tool_definition()

        assert definition["type"] == "function"
        assert definition["function"]["name"] == "generate_tearsheet"
        assert "description" in definition["function"]

        params = definition["function"]["parameters"]
        assert "symbol" in params["properties"]
        assert "period" in params["properties"]
        assert "benchmark" in params["properties"]
        assert "symbol" in params["required"]

    def test_execute_success(self, test_container_full, mock_tearsheet, mock_trade):
        """Test successful execution."""
        import pandas as pd

        from src.metrics.quantstats_reporter import QuantStatsReporter

        tool = GenerateTearsheetTool(container=test_container_full)

        mock_tracker = MagicMock()
        mock_tracker.trades = [mock_trade]
        test_container_full.metrics_tracker.override(mock_tracker)

        mock_fetcher = MagicMock()
        mock_market_data = MagicMock()
        mock_market_data.data = pd.DataFrame({"Close": pd.Series([100.0, 101.0, 99.5])})
        mock_fetcher.fetch_daily.return_value = mock_market_data
        test_container_full.market_fetcher.override(mock_fetcher)

        mock_reporter = MagicMock(spec=QuantStatsReporter)
        mock_reporter.generate_tearsheet.return_value = mock_tearsheet
        test_container_full.quantstats_reporter.override(mock_reporter)

        result = tool.execute(symbol="AAPL")

        assert "AAPL" in result
        assert "1.34" in result  # sharpe
        assert "SPY" in result  # benchmark

    def test_execute_no_trades(self, test_container_full):
        """Test handling no trades found."""
        tool = GenerateTearsheetTool(container=test_container_full)

        mock_tracker = MagicMock()
        mock_tracker.trades = []
        test_container_full.metrics_tracker.override(mock_tracker)

        result = tool.execute(symbol="AAPL")

        assert "No closed trades found" in result

    def test_execute_error_handling(self, test_container_full):
        """Test error handling on failure."""
        tool = GenerateTearsheetTool(container=test_container_full)

        mock_tracker = MagicMock()
        type(mock_tracker).trades = PropertyMock(side_effect=Exception("DB error"))
        test_container_full.metrics_tracker.override(mock_tracker)

        result = tool.execute(symbol="AAPL")

        assert "Tearsheet generation failed" in result
        assert "DB error" in result

    def test_parse_period(self, test_container_full):
        """Test period parsing."""
        tool = GenerateTearsheetTool(container=test_container_full)
        assert tool._parse_period("1m") == 30
        assert tool._parse_period("3m") == 90
        assert tool._parse_period("6m") == 180
        assert tool._parse_period("1y") == 365
        assert tool._parse_period("all") == -1
        assert tool._parse_period("60") == 60
        assert tool._parse_period("invalid") == 365

    def test_format_result_no_benchmark(self, test_container_full, mock_tearsheet):
        """Test formatting without benchmark."""
        tool = GenerateTearsheetTool(container=test_container_full)
        mock_tearsheet.benchmark_symbol = None

        result = tool._format_result(mock_tearsheet)

        assert "AAPL" in result
        assert "SPY" not in result

    def test_repr(self, test_container_full):
        """Test string representation."""
        tool = GenerateTearsheetTool(container=test_container_full)
        repr_str = repr(tool)
        assert "GenerateTearsheetTool" in repr_str
