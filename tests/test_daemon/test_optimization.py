"""Tests for daemon optimization orchestrator."""

import tempfile
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

from src.backtesting.runner import BacktestResult
from src.daemon.optimization import DaemonOptimizer
from src.optimization.param_store import OptimizedParamStore, SymbolStrategyParams
from src.optimization.results import OptimizationResult


def _mock_optimization_result(strategy_name: str = "momentum", total_trades: int = 150) -> OptimizationResult:
    return OptimizationResult(
        strategy_name=strategy_name,
        symbol="AAPL",
        best_params={"rsi_period": 14, "rsi_oversold": 30.0},
        best_metrics={
            "sharpe_ratio": 1.5,
            "total_return": 0.12,
            "max_drawdown": 0.08,
            "total_trades": total_trades,
        },
        total_trials=100,
        optimization_time_seconds=60.0,
    )


def _mock_backtest_result(total_trades: int = 150) -> BacktestResult:
    return BacktestResult(
        symbol="AAPL",
        start_date=datetime(2022, 1, 1),
        end_date=datetime(2024, 1, 1),
        total_return=0.15,
        sharpe_ratio=1.2,
        max_drawdown=-0.08,
        win_rate=0.55,
        total_trades=total_trades,
        avg_return_per_trade=0.015,
        trades=[],
    )


class TestDaemonOptimizer:
    def test_optimize_symbol_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = OptimizedParamStore(f"{tmpdir}/params.json")

            optimizer = DaemonOptimizer(store, n_trials=10, min_trades=50)

            with patch.object(
                optimizer._optimizer, "optimize", return_value=_mock_optimization_result(total_trades=150)
            ):
                result = optimizer.optimize_symbol("AAPL", "momentum")

            assert result is not None
            assert result.symbol == "AAPL"
            assert result.strategy_name == "momentum"
            assert result.validation_trades == 150
            assert store.get("AAPL", "momentum") is not None

    def test_optimize_symbol_insufficient_trades(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = OptimizedParamStore(f"{tmpdir}/params.json")

            optimizer = DaemonOptimizer(store, n_trials=10, min_trades=200)

            with patch.object(
                optimizer._optimizer, "optimize", return_value=_mock_optimization_result(total_trades=50)
            ):
                result = optimizer.optimize_symbol("AAPL", "momentum")

            assert result is None
            assert store.get("AAPL", "momentum") is None

    def test_optimize_symbol_optimization_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = OptimizedParamStore(f"{tmpdir}/params.json")

            optimizer = DaemonOptimizer(store, n_trials=10)

            with patch.object(
                optimizer._optimizer, "optimize", side_effect=RuntimeError("Optimization failed")
            ):
                result = optimizer.optimize_symbol("AAPL", "momentum")

            assert result is None

    def test_optimize_watchlist_all_fresh(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = OptimizedParamStore(f"{tmpdir}/params.json")

            # Pre-populate with fresh params
            for symbol in ["AAPL", "TSLA"]:
                for strategy in ["momentum", "mean_reversion", "trend_following"]:
                    store.save(
                        SymbolStrategyParams(
                            symbol=symbol,
                            strategy_name=strategy,
                            params={"rsi_period": 14},
                            metrics={"sharpe_ratio": 1.0},
                            optimized_at=datetime.now(UTC),
                            trials_count=100,
                            validation_trades=150,
                        )
                    )

            optimizer = DaemonOptimizer(store, n_trials=10)
            optimized, skipped, failed = optimizer.optimize_watchlist(["AAPL", "TSLA"], refresh_days=30)

            assert optimized == []
            assert sorted(skipped) == ["AAPL", "TSLA"]
            assert failed == []

    def test_optimize_watchlist_stale_params(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = OptimizedParamStore(f"{tmpdir}/params.json")

            # Pre-populate with stale params for AAPL
            old_time = datetime.now(UTC) - timedelta(days=31)
            store.save(
                SymbolStrategyParams(
                    symbol="AAPL",
                    strategy_name="momentum",
                    params={"rsi_period": 14},
                    metrics={"sharpe_ratio": 1.0},
                    optimized_at=old_time,
                    trials_count=100,
                    validation_trades=150,
                )
            )

            optimizer = DaemonOptimizer(store, n_trials=10, min_trades=50)

            with patch.object(
                optimizer._optimizer, "optimize", return_value=_mock_optimization_result(total_trades=150)
            ):
                optimized, _skipped, _failed = optimizer.optimize_watchlist(
                    ["AAPL"], strategies=["momentum"], refresh_days=30
                )

            assert "AAPL" in optimized

    def test_optimize_watchlist_all_failed(self):
        """Test that symbols with all failed optimizations are tracked in failed list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = OptimizedParamStore(f"{tmpdir}/params.json")

            # Pre-populate with stale params for TSLA
            old_time = datetime.now(UTC) - timedelta(days=31)
            for strategy in ["momentum", "mean_reversion", "trend_following"]:
                store.save(
                    SymbolStrategyParams(
                        symbol="TSLA",
                        strategy_name=strategy,
                        params={"rsi_period": 14},
                        metrics={"sharpe_ratio": 1.0},
                        optimized_at=old_time,
                        trials_count=100,
                        validation_trades=150,
                    )
                )

            optimizer = DaemonOptimizer(store, n_trials=10, min_trades=50)

            # Mock all optimizations to return insufficient trades (fail)
            with patch.object(
                optimizer._optimizer, "optimize", return_value=_mock_optimization_result(total_trades=10)
            ):
                optimized, skipped, failed = optimizer.optimize_watchlist(
                    ["TSLA"], strategies=["momentum", "mean_reversion", "trend_following"], refresh_days=30
                )

            assert optimized == []
            assert skipped == []
            assert len(failed) == 1
            assert failed[0][0] == "TSLA"
            assert "momentum" in failed[0][1]
            assert "mean_reversion" in failed[0][1]
            assert "trend_following" in failed[0][1]

    def test_repr(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = OptimizedParamStore(f"{tmpdir}/params.json")
            optimizer = DaemonOptimizer(store, n_trials=50, min_trades=200)

            repr_str = repr(optimizer)
            assert "trials=50" in repr_str
            assert "min_trades=200" in repr_str
