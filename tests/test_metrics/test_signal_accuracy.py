from datetime import UTC, datetime

import pytest

from src.cache.historical import HistoricalCache
from src.metrics.signal_accuracy import SignalAccuracyCalculator


@pytest.fixture
def cache(tmp_path):
    db_path = str(tmp_path / "test.db")
    c = HistoricalCache(db_path=db_path)
    yield c
    c.close()


@pytest.fixture
def calculator(cache):
    return SignalAccuracyCalculator(cache)


class TestSignalAccuracyCalculator:
    def test_calculate_hit_rate_buy_sell(self, cache, calculator):
        from datetime import timedelta

        now = datetime.now(UTC)

        # BUY signal that succeeds (price goes up)
        cache.record_signal_outcome(
            symbol="AAPL",
            timestamp=now,
            signal="BUY",
            confidence=0.85,
            price_at_signal=100.0,
        )

        # BUY signal that fails (price goes down)
        cache.record_signal_outcome(
            symbol="TSLA",
            timestamp=now - timedelta(seconds=1),
            signal="BUY",
            confidence=0.80,
            price_at_signal=200.0,
        )

        # SELL signal that succeeds (price goes down)
        cache.record_signal_outcome(
            symbol="MSFT",
            timestamp=now - timedelta(seconds=2),
            signal="SELL",
            confidence=0.90,
            price_at_signal=300.0,
        )

        # SELL signal that fails (price goes up)
        cache.record_signal_outcome(
            symbol="GOOGL",
            timestamp=now - timedelta(seconds=3),
            signal="SELL",
            confidence=0.75,
            price_at_signal=150.0,
        )

        signals = cache.get_signal_outcomes(window="all")
        signal_ids = {s["symbol"]: s["id"] for s in signals}

        cache.update_signal_outcome(signal_ids["AAPL"], price_at_1d=105.0, outcome_updated_at=now.isoformat())
        cache.update_signal_outcome(signal_ids["TSLA"], price_at_1d=195.0, outcome_updated_at=now.isoformat())
        cache.update_signal_outcome(signal_ids["MSFT"], price_at_1d=295.0, outcome_updated_at=now.isoformat())
        cache.update_signal_outcome(
            signal_ids["GOOGL"], price_at_1d=155.0, outcome_updated_at=now.isoformat()
        )

        metrics = calculator.calculate(window="all")

        # BUY: 1 hit / 2 total = 50%
        assert metrics.buy_hit_rate_1d == 0.5

        # SELL: 1 hit / 2 total = 50%
        assert metrics.sell_hit_rate_1d == 0.5

    def test_hold_excluded_from_metrics(self, cache, calculator):
        from datetime import timedelta

        now = datetime.now(UTC)

        # HOLD signal should be excluded
        cache.record_signal_outcome(
            symbol="AAPL",
            timestamp=now,
            signal="HOLD",
            confidence=0.60,
            price_at_signal=100.0,
        )

        # BUY signal that succeeds
        cache.record_signal_outcome(
            symbol="TSLA",
            timestamp=now - timedelta(seconds=1),
            signal="BUY",
            confidence=0.85,
            price_at_signal=200.0,
        )

        signals = cache.get_signal_outcomes(window="all")
        signal_ids = {s["symbol"]: s["id"] for s in signals}

        cache.update_signal_outcome(signal_ids["AAPL"], price_at_1d=105.0, outcome_updated_at=now.isoformat())
        cache.update_signal_outcome(signal_ids["TSLA"], price_at_1d=205.0, outcome_updated_at=now.isoformat())

        metrics = calculator.calculate(window="all")

        # Total should exclude HOLD
        assert metrics.total_signals == 2

        # BUY hit rate: 1/1 = 100% (HOLD excluded)
        assert metrics.buy_hit_rate_1d == 1.0

        # SELL hit rate: 0/0 = 0% (no SELL signals)
        assert metrics.sell_hit_rate_1d == 0.0

    def test_sell_return_inversion(self, cache, calculator):
        now = datetime.now(UTC)

        # SELL signal: price at signal 100, price at 1d 95 (5% drop)
        # Expected return: +5% (inverted for SELL)
        cache.record_signal_outcome(
            symbol="AAPL",
            timestamp=now,
            signal="SELL",
            confidence=0.85,
            price_at_signal=100.0,
        )
        signals = cache.get_signal_outcomes(window="all")
        cache.update_signal_outcome(signals[0]["id"], price_at_1d=95.0, outcome_updated_at=now.isoformat())

        metrics = calculator.calculate(window="all")

        # Return should be +5% (inverted)
        assert metrics.avg_return_1d == 5.0

    def test_calibration_bucketing(self, cache, calculator):
        from datetime import timedelta

        now = datetime.now(UTC)

        # 0.5-0.6 bucket: 1 correct, 1 incorrect
        cache.record_signal_outcome(
            symbol="A",
            timestamp=now,
            signal="BUY",
            confidence=0.55,
            price_at_signal=100.0,
        )

        cache.record_signal_outcome(
            symbol="B",
            timestamp=now - timedelta(seconds=1),
            signal="BUY",
            confidence=0.58,
            price_at_signal=100.0,
        )

        # 0.9-1.0 bucket: 2 correct, 0 incorrect
        cache.record_signal_outcome(
            symbol="C",
            timestamp=now - timedelta(seconds=2),
            signal="BUY",
            confidence=0.95,
            price_at_signal=100.0,
        )

        cache.record_signal_outcome(
            symbol="D",
            timestamp=now - timedelta(seconds=3),
            signal="SELL",
            confidence=0.92,
            price_at_signal=100.0,
        )

        signals = cache.get_signal_outcomes(window="all")
        signal_ids = {s["symbol"]: s["id"] for s in signals}

        cache.update_signal_outcome(signal_ids["A"], price_at_5d=105.0, outcome_updated_at=now.isoformat())
        cache.update_signal_outcome(signal_ids["B"], price_at_5d=95.0, outcome_updated_at=now.isoformat())
        cache.update_signal_outcome(signal_ids["C"], price_at_5d=110.0, outcome_updated_at=now.isoformat())
        cache.update_signal_outcome(signal_ids["D"], price_at_5d=90.0, outcome_updated_at=now.isoformat())

        metrics = calculator.calculate(window="all")

        # 0.5-0.6: 50% hit rate
        assert metrics.calibration_curve["0.5-0.6"] == 0.5

        # 0.9-1.0: 100% hit rate
        assert metrics.calibration_curve["0.9-1.0"] == 1.0

    def test_strategy_grouping(self, cache, calculator):
        from datetime import timedelta

        now = datetime.now(UTC)

        # Momentum strategy: 2 correct, 0 incorrect
        cache.record_signal_outcome(
            symbol="A",
            timestamp=now,
            signal="BUY",
            confidence=0.85,
            price_at_signal=100.0,
            strategy_used="momentum",
        )

        cache.record_signal_outcome(
            symbol="B",
            timestamp=now - timedelta(seconds=1),
            signal="SELL",
            confidence=0.80,
            price_at_signal=100.0,
            strategy_used="momentum",
        )

        # Mean reversion: 1 correct, 1 incorrect
        cache.record_signal_outcome(
            symbol="C",
            timestamp=now - timedelta(seconds=2),
            signal="BUY",
            confidence=0.75,
            price_at_signal=100.0,
            strategy_used="mean_reversion",
        )

        cache.record_signal_outcome(
            symbol="D",
            timestamp=now - timedelta(seconds=3),
            signal="BUY",
            confidence=0.70,
            price_at_signal=100.0,
            strategy_used="mean_reversion",
        )

        signals = cache.get_signal_outcomes(window="all")
        signal_ids = {s["symbol"]: s["id"] for s in signals}

        cache.update_signal_outcome(signal_ids["A"], price_at_5d=105.0, outcome_updated_at=now.isoformat())
        cache.update_signal_outcome(signal_ids["B"], price_at_5d=95.0, outcome_updated_at=now.isoformat())
        cache.update_signal_outcome(signal_ids["C"], price_at_5d=110.0, outcome_updated_at=now.isoformat())
        cache.update_signal_outcome(signal_ids["D"], price_at_5d=95.0, outcome_updated_at=now.isoformat())

        metrics = calculator.calculate(window="all")

        # Momentum: 100% hit rate
        assert metrics.strategy_accuracy["momentum"] == 1.0

        # Mean reversion: 50% hit rate
        assert metrics.strategy_accuracy["mean_reversion"] == 0.5

    def test_regime_grouping(self, cache, calculator):
        from datetime import timedelta

        now = datetime.now(UTC)

        # BULL regime: 2 correct, 0 incorrect
        cache.record_signal_outcome(
            symbol="A",
            timestamp=now,
            signal="BUY",
            confidence=0.85,
            price_at_signal=100.0,
            regime="BULL",
        )

        cache.record_signal_outcome(
            symbol="B",
            timestamp=now - timedelta(seconds=1),
            signal="BUY",
            confidence=0.80,
            price_at_signal=100.0,
            regime="BULL",
        )

        # BEAR regime: 0 correct, 2 incorrect
        cache.record_signal_outcome(
            symbol="C",
            timestamp=now - timedelta(seconds=2),
            signal="BUY",
            confidence=0.75,
            price_at_signal=100.0,
            regime="BEAR",
        )

        cache.record_signal_outcome(
            symbol="D",
            timestamp=now - timedelta(seconds=3),
            signal="BUY",
            confidence=0.70,
            price_at_signal=100.0,
            regime="BEAR",
        )

        signals = cache.get_signal_outcomes(window="all")
        signal_ids = {s["symbol"]: s["id"] for s in signals}

        cache.update_signal_outcome(signal_ids["A"], price_at_5d=105.0, outcome_updated_at=now.isoformat())
        cache.update_signal_outcome(signal_ids["B"], price_at_5d=110.0, outcome_updated_at=now.isoformat())
        cache.update_signal_outcome(signal_ids["C"], price_at_5d=95.0, outcome_updated_at=now.isoformat())
        cache.update_signal_outcome(signal_ids["D"], price_at_5d=90.0, outcome_updated_at=now.isoformat())

        metrics = calculator.calculate(window="all")

        # BULL: 100% hit rate
        assert metrics.regime_accuracy["BULL"] == 1.0

        # BEAR: 0% hit rate
        assert metrics.regime_accuracy["BEAR"] == 0.0

    def test_missing_horizon_prices_skipped(self, cache, calculator):
        from datetime import timedelta

        now = datetime.now(UTC)

        # Signal with only 1d price (no 5d, 20d)
        cache.record_signal_outcome(
            symbol="AAPL",
            timestamp=now,
            signal="BUY",
            confidence=0.85,
            price_at_signal=100.0,
        )

        # Signal with all prices
        cache.record_signal_outcome(
            symbol="TSLA",
            timestamp=now - timedelta(seconds=1),
            signal="BUY",
            confidence=0.80,
            price_at_signal=200.0,
        )

        signals = cache.get_signal_outcomes(window="all")
        signal_ids = {s["symbol"]: s["id"] for s in signals}

        cache.update_signal_outcome(signal_ids["AAPL"], price_at_1d=105.0, outcome_updated_at=now.isoformat())
        cache.update_signal_outcome(
            signal_ids["TSLA"],
            price_at_1d=205.0,
            price_at_5d=210.0,
            price_at_20d=220.0,
            outcome_updated_at=now.isoformat(),
        )

        metrics = calculator.calculate(window="all")

        # 1d: both signals counted = 100%
        assert metrics.buy_hit_rate_1d == 1.0

        # 5d: only TSLA counted = 100%
        assert metrics.buy_hit_rate_5d == 1.0

        # 20d: only TSLA counted = 100%
        assert metrics.buy_hit_rate_20d == 1.0

    def test_early_exit_price_priority(self, cache, calculator):
        from datetime import timedelta

        now = datetime.now(UTC)

        # Signal with early exit (actual_exit_price should be used)
        cache.record_signal_outcome(
            symbol="AAPL",
            timestamp=now,
            signal="BUY",
            confidence=0.85,
            price_at_signal=100.0,
        )

        # Signal without early exit (price_at_1d should be used)
        cache.record_signal_outcome(
            symbol="TSLA",
            timestamp=now - timedelta(seconds=1),
            signal="BUY",
            confidence=0.80,
            price_at_signal=200.0,
        )

        signals = cache.get_signal_outcomes(window="all")
        signal_ids = {s["symbol"]: s["id"] for s in signals}

        cache.update_signal_outcome(
            signal_ids["AAPL"],
            price_at_1d=105.0,
            actual_exit_price=103.0,
            actual_exit_date=now.isoformat(),
            outcome_updated_at=now.isoformat(),
        )
        cache.update_signal_outcome(signal_ids["TSLA"], price_at_1d=205.0, outcome_updated_at=now.isoformat())

        metrics = calculator.calculate(window="all")

        # Both signals succeeded (early exit: 103 > 100, normal: 205 > 200)
        assert metrics.buy_hit_rate_1d == 1.0

        # Average return: ((103-100)/100 * 100 + (205-200)/200 * 100) / 2 = (3.0 + 2.5) / 2 = 2.75
        assert abs(metrics.avg_return_1d - 2.75) < 0.01
