"""Tests for optimized parameter store."""

import tempfile
from datetime import UTC, datetime, timedelta

from src.optimization.param_store import OptimizedParamStore, SymbolStrategyParams


def _make_params(
    symbol: str = "AAPL",
    strategy: str = "momentum",
    optimized_at: datetime | None = None,
) -> SymbolStrategyParams:
    return SymbolStrategyParams(
        symbol=symbol,
        strategy_name=strategy,
        params={"rsi_period": 14, "rsi_oversold": 30.0},
        metrics={"sharpe_ratio": 1.5, "total_return": 0.12},
        optimized_at=optimized_at or datetime.now(UTC),
        trials_count=100,
        validation_trades=150,
    )


class TestSymbolStrategyParams:
    def test_creation(self):
        params = _make_params()

        assert params.symbol == "AAPL"
        assert params.strategy_name == "momentum"
        assert params.params["rsi_period"] == 14
        assert params.metrics["sharpe_ratio"] == 1.5
        assert params.trials_count == 100
        assert params.validation_trades == 150


class TestOptimizedParamStore:
    def test_empty_store(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = OptimizedParamStore(f"{tmpdir}/params.json")

            assert store.load_all() == {}
            assert store.get("AAPL", "momentum") is None

    def test_save_and_get(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = OptimizedParamStore(f"{tmpdir}/params.json")
            params = _make_params()

            store.save(params)

            loaded = store.get("AAPL", "momentum")
            assert loaded is not None
            assert loaded.symbol == "AAPL"
            assert loaded.params["rsi_period"] == 14
            assert loaded.metrics["sharpe_ratio"] == 1.5

    def test_save_multiple_strategies(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = OptimizedParamStore(f"{tmpdir}/params.json")

            store.save(_make_params(strategy="momentum"))
            store.save(_make_params(strategy="mean_reversion"))
            store.save(_make_params(symbol="TSLA", strategy="trend_following"))

            assert store.get("AAPL", "momentum") is not None
            assert store.get("AAPL", "mean_reversion") is not None
            assert store.get("TSLA", "trend_following") is not None
            assert store.get("TSLA", "momentum") is None

    def test_save_overwrites_existing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = OptimizedParamStore(f"{tmpdir}/params.json")

            store.save(_make_params())
            updated = _make_params()
            updated.params["rsi_period"] = 21
            store.save(updated)

            loaded = store.get("AAPL", "momentum")
            assert loaded.params["rsi_period"] == 21

    def test_persistence_across_instances(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/params.json"

            store1 = OptimizedParamStore(path)
            store1.save(_make_params())

            store2 = OptimizedParamStore(path)
            loaded = store2.get("AAPL", "momentum")

            assert loaded is not None
            assert loaded.symbol == "AAPL"

    def test_load_all(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = OptimizedParamStore(f"{tmpdir}/params.json")
            store.save(_make_params(symbol="AAPL", strategy="momentum"))
            store.save(_make_params(symbol="TSLA", strategy="trend_following"))

            all_params = store.load_all()

            assert "AAPL" in all_params
            assert "TSLA" in all_params
            assert "momentum" in all_params["AAPL"]

    def test_is_stale_no_params(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = OptimizedParamStore(f"{tmpdir}/params.json")

            assert store.is_stale("AAPL", "momentum") is True

    def test_is_stale_fresh(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = OptimizedParamStore(f"{tmpdir}/params.json")
            store.save(_make_params())

            assert store.is_stale("AAPL", "momentum", max_age_days=30) is False

    def test_is_stale_old(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = OptimizedParamStore(f"{tmpdir}/params.json")
            old_time = datetime.now(UTC) - timedelta(days=31)
            store.save(_make_params(optimized_at=old_time))

            assert store.is_stale("AAPL", "momentum", max_age_days=30) is True

    def test_load_nonexistent_path(self):
        store = OptimizedParamStore("/nonexistent/path/params.json")

        assert store.load_all() == {}

    def test_repr(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = OptimizedParamStore(f"{tmpdir}/params.json")
            store.save(_make_params())

            repr_str = repr(store)
            assert "symbols=1" in repr_str
            assert "entries=1" in repr_str
