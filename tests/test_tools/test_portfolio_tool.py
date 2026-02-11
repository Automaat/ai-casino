"""Tests for OptimizePortfolioTool."""

from unittest.mock import MagicMock

import pytest

from src.tools.portfolio import OptimizePortfolioTool


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

    def test_name(self, test_container_full):
        """Test tool name."""
        tool = OptimizePortfolioTool(container=test_container_full)
        assert tool.name == "optimize_portfolio"

    def test_requires_confirmation(self, test_container_full):
        """Test that tool requires confirmation."""
        tool = OptimizePortfolioTool(container=test_container_full)
        assert tool.requires_confirmation is True

    def test_get_tool_definition(self, test_container_full):
        """Test tool definition format."""
        tool = OptimizePortfolioTool(container=test_container_full)
        definition = tool.get_tool_definition().model_dump(mode="json", by_alias=True, exclude_none=True)

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

    def test_execute_success(self, test_container_full, mock_optimization_result):
        """Test successful execution."""
        from src.optimization.optimizer import OptunaOptimizer

        tool = OptimizePortfolioTool(container=test_container_full)

        mock_optimizer = MagicMock(spec=OptunaOptimizer)
        mock_optimizer.optimize.return_value = mock_optimization_result
        test_container_full.optuna_optimizer.override(mock_optimizer)

        result = tool.execute(symbol="AAPL", start_date="2023-01-01", end_date="2024-01-01")

        assert "AAPL" in result
        assert "momentum" in result
        assert "1.87" in result  # sharpe
        assert "50" in result  # trials
        assert "42.3" in result  # time
        mock_optimizer.optimize.assert_called_once_with("AAPL", "2023-01-01", "2024-01-01", "momentum")

    def test_execute_uppercase_symbol(self, test_container_full, mock_optimization_result):
        """Test that symbol is uppercased."""
        from src.optimization.optimizer import OptunaOptimizer

        tool = OptimizePortfolioTool(container=test_container_full)

        mock_optimizer = MagicMock(spec=OptunaOptimizer)
        mock_optimizer.optimize.return_value = mock_optimization_result
        test_container_full.optuna_optimizer.override(mock_optimizer)

        tool.execute(symbol="aapl", start_date="2023-01-01", end_date="2024-01-01")

        mock_optimizer.optimize.assert_called_once_with("AAPL", "2023-01-01", "2024-01-01", "momentum")

    def test_execute_custom_strategy(self, test_container_full, mock_optimization_result):
        """Test execution with custom strategy."""
        from src.optimization.optimizer import OptunaOptimizer

        tool = OptimizePortfolioTool(container=test_container_full)

        mock_optimizer = MagicMock(spec=OptunaOptimizer)
        mock_optimizer.optimize.return_value = mock_optimization_result
        test_container_full.optuna_optimizer.override(mock_optimizer)

        tool.execute(
            symbol="AAPL", start_date="2023-01-01", end_date="2024-01-01", strategy="trend_following"
        )

        mock_optimizer.optimize.assert_called_once_with("AAPL", "2023-01-01", "2024-01-01", "trend_following")

    def test_execute_custom_trials(self, test_container_full, mock_optimization_result):
        """Test execution with custom trial count."""
        from dependency_injector import providers

        from src.optimization.optimizer import OptunaOptimizer

        tool = OptimizePortfolioTool(container=test_container_full)

        mock_optimizer = MagicMock(spec=OptunaOptimizer)
        mock_optimizer.optimize.return_value = mock_optimization_result

        factory_called = False
        original_trials = None

        def mock_factory(n_trials=50):
            nonlocal factory_called, original_trials
            factory_called = True
            original_trials = n_trials
            return mock_optimizer

        test_container_full.optuna_optimizer.override(providers.Factory(mock_factory))

        tool.execute(symbol="AAPL", start_date="2023-01-01", end_date="2024-01-01", n_trials=100)

        assert factory_called
        assert original_trials == 100

    def test_execute_error_handling(self, test_container_full):
        """Test error handling on failure."""
        from src.optimization.optimizer import OptunaOptimizer

        tool = OptimizePortfolioTool(container=test_container_full)

        mock_optimizer = MagicMock(spec=OptunaOptimizer)
        mock_optimizer.optimize.side_effect = Exception("Optimization failed")
        test_container_full.optuna_optimizer.override(mock_optimizer)

        result = tool.execute(symbol="INVALID", start_date="2023-01-01", end_date="2024-01-01")

        assert "Optimization failed" in result

    def test_format_result_float_params(self, test_container_full):
        """Test formatting with float parameters."""
        tool = OptimizePortfolioTool(container=test_container_full)
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

    def test_repr(self, test_container_full):
        """Test string representation."""
        tool = OptimizePortfolioTool(container=test_container_full)
        repr_str = repr(tool)
        assert "OptimizePortfolioTool" in repr_str
