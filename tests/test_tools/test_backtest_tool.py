"""Tests for RunBacktestTool."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.tools.backtest import RunBacktestTool


@pytest.fixture
def tool():
    """Create RunBacktestTool."""
    return RunBacktestTool()


@pytest.fixture
def mock_backtest_result():
    """Create mock BacktestResult."""
    result = MagicMock()
    result.symbol = "AAPL"
    result.start_date = datetime(2023, 1, 1)
    result.end_date = datetime(2024, 1, 1)
    result.total_return = 0.2534
    result.sharpe_ratio = 1.45
    result.max_drawdown = -0.0823
    result.win_rate = 0.62
    result.total_trades = 48
    result.avg_return_per_trade = 0.0053
    return result


class TestRunBacktestTool:
    """Tests for RunBacktestTool."""

    def test_name(self, tool):
        """Test tool name."""
        assert tool.name == "run_backtest"

    def test_requires_confirmation(self, tool):
        """Test that tool requires confirmation."""
        assert tool.requires_confirmation is True

    def test_get_tool_definition(self, tool):
        """Test tool definition format."""
        definition = tool.get_tool_definition()

        assert definition["type"] == "function"
        assert definition["function"]["name"] == "run_backtest"
        assert "description" in definition["function"]

        params = definition["function"]["parameters"]
        assert "symbol" in params["properties"]
        assert "start_date" in params["properties"]
        assert "end_date" in params["properties"]
        assert "cash" in params["properties"]
        assert set(params["required"]) == {"symbol", "start_date", "end_date"}

    def test_execute_success(self, tool, mock_backtest_result):
        """Test successful execution."""
        with patch("src.backtesting.runner.BacktestRunner") as mock_runner_cls:
            mock_runner = MagicMock()
            mock_runner.run_backtest.return_value = mock_backtest_result
            mock_runner_cls.return_value = mock_runner

            result = tool.execute("AAPL", "2023-01-01", "2024-01-01")

            assert "AAPL" in result
            assert "25.34%" in result  # total return
            assert "1.45" in result  # sharpe
            assert "48" in result  # total trades
            mock_runner.run_backtest.assert_called_once_with("AAPL", "2023-01-01", "2024-01-01")

    def test_execute_uppercase_symbol(self, tool, mock_backtest_result):
        """Test that symbol is uppercased."""
        with patch("src.backtesting.runner.BacktestRunner") as mock_runner_cls:
            mock_runner = MagicMock()
            mock_runner.run_backtest.return_value = mock_backtest_result
            mock_runner_cls.return_value = mock_runner

            tool.execute("aapl", "2023-01-01", "2024-01-01")

            mock_runner.run_backtest.assert_called_once_with("AAPL", "2023-01-01", "2024-01-01")

    def test_execute_custom_cash(self, tool, mock_backtest_result):
        """Test execution with custom cash."""
        with patch("src.backtesting.runner.BacktestRunner") as mock_runner_cls:
            mock_runner = MagicMock()
            mock_runner.run_backtest.return_value = mock_backtest_result
            mock_runner_cls.return_value = mock_runner

            tool.execute("AAPL", "2023-01-01", "2024-01-01", cash=50000)

            mock_runner_cls.assert_called_once_with(cash=50000.0)

    def test_execute_error_handling(self, tool):
        """Test error handling on failure."""
        with patch("src.backtesting.runner.BacktestRunner") as mock_runner_cls:
            mock_runner = MagicMock()
            mock_runner.run_backtest.side_effect = Exception("No data available")
            mock_runner_cls.return_value = mock_runner

            result = tool.execute("INVALID", "2023-01-01", "2024-01-01")

            assert "Backtest failed" in result
            assert "No data available" in result

    def test_repr(self, tool):
        """Test string representation."""
        repr_str = repr(tool)
        assert "RunBacktestTool" in repr_str
