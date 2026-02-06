"""Tests for OptimizePortfolioTool."""

from unittest.mock import MagicMock, patch

import pytest

from src.tools.portfolio import OptimizePortfolioTool


@pytest.fixture
def tool():
    """Create OptimizePortfolioTool."""
    return OptimizePortfolioTool()


@pytest.fixture
def mock_optimization_result():
    """Create mock OptimizationResult."""
    result = MagicMock()
    result.strategy_name = "momentum"
    result.symbol = "AAPL"
    result.best_params = {"rsi_period": 14, "macd_fast": 12, "macd_slow": 26}
    result.best_metrics = {
        "sharpe_ratio": 1.87,
        "total_return": 0.3245,
        "max_drawdown": 0.0912,
    }
    result.total_trials = 50
    result.optimization_time_seconds = 42.3
    return result


class TestOptimizePortfolioTool:
    """Tests for OptimizePortfolioTool."""

    def test_name(self, tool):
        """Test tool name."""
        assert tool.name == "optimize_portfolio"

    def test_requires_confirmation(self, tool):
        """Test that tool requires confirmation."""
        assert tool.requires_confirmation is True

    def test_get_tool_definition(self, tool):
        """Test tool definition format."""
        definition = tool.get_tool_definition()

        assert definition["type"] == "function"
        assert definition["function"]["name"] == "optimize_portfolio"
        assert "description" in definition["function"]

        params = definition["function"]["parameters"]
        assert "symbol" in params["properties"]
        assert "start_date" in params["properties"]
        assert "end_date" in params["properties"]
        assert "strategy" in params["properties"]
        assert "n_trials" in params["properties"]
        assert set(params["required"]) == {"symbol", "start_date", "end_date"}

        strategy_prop = params["properties"]["strategy"]
        assert "enum" in strategy_prop
        assert "momentum" in strategy_prop["enum"]

    def test_execute_success(self, tool, mock_optimization_result):
        """Test successful execution."""
        with patch("src.optimization.optimizer.OptunaOptimizer") as mock_optimizer_cls:
            mock_optimizer = MagicMock()
            mock_optimizer.optimize.return_value = mock_optimization_result
            mock_optimizer_cls.return_value = mock_optimizer

            result = tool.execute("AAPL", "2023-01-01", "2024-01-01")

            assert "AAPL" in result
            assert "momentum" in result
            assert "1.87" in result  # sharpe
            assert "50" in result  # trials
            assert "42.3" in result  # time
            mock_optimizer.optimize.assert_called_once_with("AAPL", "2023-01-01", "2024-01-01", "momentum")

    def test_execute_uppercase_symbol(self, tool, mock_optimization_result):
        """Test that symbol is uppercased."""
        with patch("src.optimization.optimizer.OptunaOptimizer") as mock_optimizer_cls:
            mock_optimizer = MagicMock()
            mock_optimizer.optimize.return_value = mock_optimization_result
            mock_optimizer_cls.return_value = mock_optimizer

            tool.execute("aapl", "2023-01-01", "2024-01-01")

            mock_optimizer.optimize.assert_called_once_with("AAPL", "2023-01-01", "2024-01-01", "momentum")

    def test_execute_custom_strategy(self, tool, mock_optimization_result):
        """Test execution with custom strategy."""
        with patch("src.optimization.optimizer.OptunaOptimizer") as mock_optimizer_cls:
            mock_optimizer = MagicMock()
            mock_optimizer.optimize.return_value = mock_optimization_result
            mock_optimizer_cls.return_value = mock_optimizer

            tool.execute("AAPL", "2023-01-01", "2024-01-01", strategy="trend_following")

            mock_optimizer.optimize.assert_called_once_with(
                "AAPL", "2023-01-01", "2024-01-01", "trend_following"
            )

    def test_execute_custom_trials(self, tool, mock_optimization_result):
        """Test execution with custom trial count."""
        with patch("src.optimization.optimizer.OptunaOptimizer") as mock_optimizer_cls:
            mock_optimizer = MagicMock()
            mock_optimizer.optimize.return_value = mock_optimization_result
            mock_optimizer_cls.return_value = mock_optimizer

            tool.execute("AAPL", "2023-01-01", "2024-01-01", n_trials=100)

            mock_optimizer_cls.assert_called_once_with(n_trials=100)

    def test_execute_error_handling(self, tool):
        """Test error handling on failure."""
        with patch("src.optimization.optimizer.OptunaOptimizer") as mock_optimizer_cls:
            mock_optimizer = MagicMock()
            mock_optimizer.optimize.side_effect = Exception("Optimization failed")
            mock_optimizer_cls.return_value = mock_optimizer

            result = tool.execute("INVALID", "2023-01-01", "2024-01-01")

            assert "Optimization failed" in result

    def test_format_result_float_params(self, tool):
        """Test formatting with float parameters."""
        result = MagicMock()
        result.symbol = "AAPL"
        result.strategy_name = "momentum"
        result.best_params = {"rsi_period": 14, "threshold": 0.025}
        result.best_metrics = {"sharpe_ratio": 1.5, "total_return": 0.2, "max_drawdown": 0.1}
        result.total_trials = 50
        result.optimization_time_seconds = 30.0

        formatted = tool._format_result(result)

        assert "0.0250" in formatted  # float param
        assert "14" in formatted  # int param

    def test_repr(self, tool):
        """Test string representation."""
        repr_str = repr(tool)
        assert "OptimizePortfolioTool" in repr_str
