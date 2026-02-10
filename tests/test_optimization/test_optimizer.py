"""Tests for Optuna optimizer module."""

import pytest

from src.optimization.optimizer import OptunaOptimizer
from src.optimization.results import OptimizationResult
from src.optimization.validation import WalkForwardValidator


class TestOptunaOptimizer:
    """Tests for OptunaOptimizer."""

    def test_init_defaults(self, mock_backtest_runner):
        """Test default initialization."""
        optimizer = OptunaOptimizer(runner=mock_backtest_runner)

        assert optimizer.n_trials == 100
        assert optimizer.directions == ["maximize"]
        assert optimizer._multi_objective is False
        assert optimizer.validator is None

    def test_init_multi_objective(self, mock_backtest_runner):
        """Test multi-objective initialization."""
        optimizer = OptunaOptimizer(
            runner=mock_backtest_runner,
            directions=["maximize", "maximize", "minimize"],
        )

        assert optimizer._multi_objective is True
        assert len(optimizer.directions) == 3

    def test_init_with_validator(self, mock_backtest_runner):
        """Test initialization with validator."""
        validator = WalkForwardValidator(n_splits=3)
        optimizer = OptunaOptimizer(
            runner=mock_backtest_runner,
            validator=validator,
        )

        assert optimizer.validator is not None

    def test_optimize_momentum(self, mock_backtest_runner, sample_dates):
        """Test optimization with momentum strategy."""
        optimizer = OptunaOptimizer(
            runner=mock_backtest_runner,
            n_trials=5,
        )

        result = optimizer.optimize(
            symbol="AAPL",
            start_date=sample_dates["start_date"],
            end_date=sample_dates["end_date"],
            strategy_name="momentum",
        )

        assert isinstance(result, OptimizationResult)
        assert result.strategy_name == "momentum"
        assert result.symbol == "AAPL"
        assert result.total_trials == 5
        assert "sharpe_ratio" in result.best_metrics

    def test_optimize_trend_following(self, mock_backtest_runner, sample_dates):
        """Test optimization with trend following strategy."""
        optimizer = OptunaOptimizer(
            runner=mock_backtest_runner,
            n_trials=5,
        )

        result = optimizer.optimize(
            symbol="AAPL",
            start_date=sample_dates["start_date"],
            end_date=sample_dates["end_date"],
            strategy_name="trend_following",
        )

        assert result.strategy_name == "trend_following"
        assert result.total_trials == 5

    def test_optimize_mean_reversion(self, mock_backtest_runner, sample_dates):
        """Test optimization with mean reversion strategy."""
        optimizer = OptunaOptimizer(
            runner=mock_backtest_runner,
            n_trials=5,
        )

        result = optimizer.optimize(
            symbol="AAPL",
            start_date=sample_dates["start_date"],
            end_date=sample_dates["end_date"],
            strategy_name="mean_reversion",
        )

        assert result.strategy_name == "mean_reversion"
        assert "bb_period" in result.best_params or "bb_std" in result.best_params

    def test_optimize_ensemble(self, mock_backtest_runner, sample_dates):
        """Test optimization with ensemble strategy."""
        optimizer = OptunaOptimizer(
            runner=mock_backtest_runner,
            n_trials=5,
        )

        result = optimizer.optimize(
            symbol="AAPL",
            start_date=sample_dates["start_date"],
            end_date=sample_dates["end_date"],
            strategy_name="ensemble",
        )

        assert result.strategy_name == "ensemble"

    def test_optimize_multi_objective(self, mock_backtest_runner, sample_dates):
        """Test multi-objective optimization."""
        optimizer = OptunaOptimizer(
            runner=mock_backtest_runner,
            n_trials=5,
            directions=["maximize", "maximize", "minimize"],
        )

        result = optimizer.optimize(
            symbol="AAPL",
            start_date=sample_dates["start_date"],
            end_date=sample_dates["end_date"],
            strategy_name="momentum",
        )

        assert result.pareto_front is not None
        assert "sharpe_ratio" in result.best_metrics
        assert "total_return" in result.best_metrics
        assert "max_drawdown" in result.best_metrics

    def test_optimize_with_validation(self, mock_backtest_runner, sample_dates):
        """Test optimization with walk-forward validation."""
        validator = WalkForwardValidator(n_splits=2)
        optimizer = OptunaOptimizer(
            runner=mock_backtest_runner,
            n_trials=3,
            validator=validator,
        )

        result = optimizer.optimize(
            symbol="AAPL",
            start_date=sample_dates["start_date"],
            end_date=sample_dates["end_date"],
            strategy_name="momentum",
        )

        assert result.total_trials == 3

    def test_invalid_strategy(self, mock_backtest_runner, sample_dates):
        """Test error on invalid strategy."""
        optimizer = OptunaOptimizer(
            runner=mock_backtest_runner,
            n_trials=5,
        )

        with pytest.raises(ValueError, match="Unknown strategy"):
            optimizer.optimize(
                symbol="AAPL",
                start_date=sample_dates["start_date"],
                end_date=sample_dates["end_date"],
                strategy_name="invalid_strategy",
            )

    def test_repr(self, mock_backtest_runner):
        """Test string representation."""
        optimizer = OptunaOptimizer(
            runner=mock_backtest_runner,
            n_trials=50,
        )

        repr_str = repr(optimizer)
        assert "trials=50" in repr_str
        assert "multi_objective=False" in repr_str

    def test_run_backtest_safe_expected_errors(self, mock_backtest_runner, sample_dates):
        """Verify expected errors (ValueError, KeyError, IndexError) return None with warning."""
        from unittest.mock import MagicMock

        from src.optimization.search_space import StrategyType

        optimizer = OptunaOptimizer(runner=mock_backtest_runner, n_trials=5)

        # Mock backtest to raise ValueError (expected error)
        mock_backtest_runner.run_backtest = MagicMock(side_effect=ValueError("Insufficient data"))

        # Get the strategy class
        strategy_class = optimizer._create_strategy_class(
            StrategyType.MOMENTUM, {"rsi_period": 14, "rsi_oversold": 30.0, "rsi_overbought": 70.0}
        )

        # Should return None for expected errors
        result = optimizer._run_backtest_safe(
            symbol="AAPL",
            start_date=sample_dates["start_date"],
            end_date=sample_dates["end_date"],
            strategy_class=strategy_class,
        )

        assert result is None

        # Test KeyError
        mock_backtest_runner.run_backtest = MagicMock(side_effect=KeyError("missing column"))
        result = optimizer._run_backtest_safe(
            symbol="AAPL",
            start_date=sample_dates["start_date"],
            end_date=sample_dates["end_date"],
            strategy_class=strategy_class,
        )
        assert result is None

        # Test IndexError
        mock_backtest_runner.run_backtest = MagicMock(side_effect=IndexError("out of bounds"))
        result = optimizer._run_backtest_safe(
            symbol="AAPL",
            start_date=sample_dates["start_date"],
            end_date=sample_dates["end_date"],
            strategy_class=strategy_class,
        )
        assert result is None

    def test_run_backtest_safe_unexpected_errors(self, mock_backtest_runner, sample_dates):
        """Verify unexpected errors (RuntimeError, etc.) propagate."""
        from unittest.mock import MagicMock

        from src.optimization.search_space import StrategyType

        optimizer = OptunaOptimizer(runner=mock_backtest_runner, n_trials=5)

        # Mock backtest to raise RuntimeError (unexpected error)
        mock_backtest_runner.run_backtest = MagicMock(side_effect=RuntimeError("Strategy bug"))

        # Get the strategy class
        strategy_class = optimizer._create_strategy_class(
            StrategyType.MOMENTUM, {"rsi_period": 14, "rsi_oversold": 30.0, "rsi_overbought": 70.0}
        )

        # Should propagate unexpected errors
        with pytest.raises(RuntimeError, match="Strategy bug"):
            optimizer._run_backtest_safe(
                symbol="AAPL",
                start_date=sample_dates["start_date"],
                end_date=sample_dates["end_date"],
                strategy_class=strategy_class,
            )


class TestOptimizationResult:
    """Tests for OptimizationResult."""

    def test_result_creation(self):
        """Test creating an optimization result."""
        result = OptimizationResult(
            strategy_name="momentum",
            symbol="AAPL",
            best_params={"rsi_period": 14, "rsi_oversold": 30.0},
            best_metrics={"sharpe_ratio": 1.5, "total_return": 0.2, "max_drawdown": 0.1},
            total_trials=100,
            optimization_time_seconds=45.5,
        )

        assert result.strategy_name == "momentum"
        assert result.symbol == "AAPL"
        assert result.best_params["rsi_period"] == 14
        assert result.best_metrics["sharpe_ratio"] == 1.5
        assert result.total_trials == 100

    def test_result_with_pareto(self):
        """Test result with pareto front."""
        pareto = [
            {"rsi_period": 14, "sharpe_ratio": 1.5, "total_return": 0.2, "max_drawdown": 0.1},
            {"rsi_period": 12, "sharpe_ratio": 1.3, "total_return": 0.25, "max_drawdown": 0.15},
        ]

        result = OptimizationResult(
            strategy_name="momentum",
            symbol="AAPL",
            best_params={"rsi_period": 14},
            best_metrics={"sharpe_ratio": 1.5},
            pareto_front=pareto,
            total_trials=100,
            optimization_time_seconds=45.5,
        )

        assert result.pareto_front is not None
        assert len(result.pareto_front) == 2

    def test_repr(self):
        """Test string representation."""
        result = OptimizationResult(
            strategy_name="momentum",
            symbol="AAPL",
            best_params={},
            best_metrics={"sharpe_ratio": 1.45},
            total_trials=100,
            optimization_time_seconds=45.5,
        )

        repr_str = repr(result)
        assert "momentum" in repr_str
        assert "AAPL" in repr_str
        assert "trials=100" in repr_str
        assert "sharpe=1.45" in repr_str
