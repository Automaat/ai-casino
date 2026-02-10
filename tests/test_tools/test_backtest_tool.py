"""Tests for RunBacktestTool."""

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from src.tools.backtest import RunBacktestTool


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

    def test_name(self, test_container_full):
        """Test tool name."""
        tool = RunBacktestTool(container=test_container_full)
        assert tool.name == "run_backtest"

    def test_requires_confirmation(self, test_container_full):
        """Test that tool requires confirmation."""
        tool = RunBacktestTool(container=test_container_full)
        assert tool.requires_confirmation is True

    def test_get_tool_definition(self, test_container_full):
        """Test tool definition format."""
        tool = RunBacktestTool(container=test_container_full)
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

    def test_execute_success(self, test_container_full, mock_backtest_result):
        """Test successful execution."""
        from src.backtesting.runner import BacktestRunner

        tool = RunBacktestTool(container=test_container_full)

        mock_runner = MagicMock(spec=BacktestRunner)
        mock_runner.run_backtest.return_value = mock_backtest_result
        test_container_full.backtest_runner.override(mock_runner)

        result = tool.execute(symbol="AAPL", start_date="2023-01-01", end_date="2024-01-01")

        assert "AAPL" in result
        assert "25.34%" in result  # total return
        assert "1.45" in result  # sharpe
        assert "48" in result  # total trades
        mock_runner.run_backtest.assert_called_once_with("AAPL", "2023-01-01", "2024-01-01")

    def test_execute_uppercase_symbol(self, test_container_full, mock_backtest_result):
        """Test that symbol is uppercased."""
        from src.backtesting.runner import BacktestRunner

        tool = RunBacktestTool(container=test_container_full)

        mock_runner = MagicMock(spec=BacktestRunner)
        mock_runner.run_backtest.return_value = mock_backtest_result
        test_container_full.backtest_runner.override(mock_runner)

        tool.execute(symbol="aapl", start_date="2023-01-01", end_date="2024-01-01")

        mock_runner.run_backtest.assert_called_once_with("AAPL", "2023-01-01", "2024-01-01")

    def test_execute_custom_cash(self, test_container_full, mock_backtest_result):
        """Test execution with custom cash."""
        from dependency_injector import providers
        from src.backtesting.runner import BacktestRunner

        tool = RunBacktestTool(container=test_container_full)

        mock_runner = MagicMock(spec=BacktestRunner)
        mock_runner.run_backtest.return_value = mock_backtest_result

        factory_called = False
        original_cash = None

        def mock_factory(cash=10000.0):
            nonlocal factory_called, original_cash
            factory_called = True
            original_cash = cash
            return mock_runner

        test_container_full.backtest_runner.override(providers.Factory(mock_factory))

        tool.execute(symbol="AAPL", start_date="2023-01-01", end_date="2024-01-01", cash=50000)

        assert factory_called
        assert original_cash == 50000.0

    def test_execute_error_handling(self, test_container_full):
        """Test error handling on failure."""
        from src.backtesting.runner import BacktestRunner

        tool = RunBacktestTool(container=test_container_full)

        mock_runner = MagicMock(spec=BacktestRunner)
        mock_runner.run_backtest.side_effect = Exception("No data available")
        test_container_full.backtest_runner.override(mock_runner)

        result = tool.execute(symbol="INVALID", start_date="2023-01-01", end_date="2024-01-01")

        assert "Backtest failed" in result
        assert "No data available" in result

    def test_repr(self, test_container_full):
        """Test string representation."""
        tool = RunBacktestTool(container=test_container_full)
        repr_str = repr(tool)
        assert "RunBacktestTool" in repr_str
