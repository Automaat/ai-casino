"""Signal accuracy metrics calculator."""

from collections import defaultdict

from pydantic import BaseModel

from src.cache.historical import HistoricalCache

# Confidence bucket thresholds
CONF_BUCKET_0_5 = 0.5
CONF_BUCKET_0_6 = 0.6
CONF_BUCKET_0_7 = 0.7
CONF_BUCKET_0_8 = 0.8
CONF_BUCKET_0_9 = 0.9
CONF_BUCKET_1_0 = 1.0


class SignalAccuracyMetrics(BaseModel):
    """Signal accuracy metrics for a time window."""

    window: str
    total_signals: int
    buy_hit_rate_1d: float
    buy_hit_rate_5d: float
    buy_hit_rate_20d: float
    sell_hit_rate_1d: float
    sell_hit_rate_5d: float
    sell_hit_rate_20d: float
    calibration_curve: dict[str, float]
    strategy_accuracy: dict[str, float]
    regime_accuracy: dict[str, float] | None
    avg_return_1d: float
    avg_return_5d: float
    avg_return_20d: float


class SignalAccuracyCalculator:
    """Calculate signal accuracy metrics from historical outcomes."""

    def __init__(self, historical_cache: HistoricalCache) -> None:
        """Initialize calculator.

        Args:
            historical_cache: Historical cache with signal outcomes
        """
        self._cache = historical_cache

    def calculate(self, window: str = "all") -> SignalAccuracyMetrics:
        """Calculate accuracy metrics for time window.

        Args:
            window: Time window (7d/30d/90d/all)

        Returns:
            SignalAccuracyMetrics with hit rates and calibration
        """
        signals = self._cache.get_signal_outcomes(window=window)

        return SignalAccuracyMetrics(
            window=window,
            total_signals=len(signals),
            buy_hit_rate_1d=self._calculate_hit_rate(signals, "BUY", "1d"),
            buy_hit_rate_5d=self._calculate_hit_rate(signals, "BUY", "5d"),
            buy_hit_rate_20d=self._calculate_hit_rate(signals, "BUY", "20d"),
            sell_hit_rate_1d=self._calculate_hit_rate(signals, "SELL", "1d"),
            sell_hit_rate_5d=self._calculate_hit_rate(signals, "SELL", "5d"),
            sell_hit_rate_20d=self._calculate_hit_rate(signals, "SELL", "20d"),
            calibration_curve=self._calculate_calibration(signals, "5d"),
            strategy_accuracy=self._calculate_strategy_accuracy(signals, "5d"),
            regime_accuracy=self._calculate_regime_accuracy(signals, "5d"),
            avg_return_1d=self._calculate_avg_return(signals, "1d"),
            avg_return_5d=self._calculate_avg_return(signals, "5d"),
            avg_return_20d=self._calculate_avg_return(signals, "20d"),
        )

    def _calculate_hit_rate(self, signals: list[dict], signal_type: str, horizon: str) -> float:
        """Calculate hit rate for signal type at horizon.

        Args:
            signals: List of signal outcome records
            signal_type: Signal type (BUY/SELL)
            horizon: Time horizon (1d/5d/20d)

        Returns:
            Hit rate (0.0-1.0)
        """
        filtered = [s for s in signals if s["signal"] == signal_type]
        if not filtered:
            return 0.0

        hits = 0
        total = 0

        for sig in filtered:
            # Use actual exit if closed early, else market price at T+Nd
            price_future = sig["actual_exit_price"] or sig.get(f"price_at_{horizon}")
            if price_future is None:
                continue

            if (signal_type == "BUY" and price_future > sig["price_at_signal"]) or (
                signal_type == "SELL" and price_future < sig["price_at_signal"]
            ):
                hits += 1

            total += 1

        return (hits / total) if total > 0 else 0.0

    def _get_confidence_bucket(self, confidence: float) -> str | None:
        """Get confidence bucket label for a confidence value.

        Args:
            confidence: Confidence value (0.0-1.0)

        Returns:
            Bucket label or None if out of range
        """
        if CONF_BUCKET_0_5 <= confidence < CONF_BUCKET_0_6:
            return "0.5-0.6"
        if CONF_BUCKET_0_6 <= confidence < CONF_BUCKET_0_7:
            return "0.6-0.7"
        if CONF_BUCKET_0_7 <= confidence < CONF_BUCKET_0_8:
            return "0.7-0.8"
        if CONF_BUCKET_0_8 <= confidence < CONF_BUCKET_0_9:
            return "0.8-0.9"
        if CONF_BUCKET_0_9 <= confidence <= CONF_BUCKET_1_0:
            return "0.9-1.0"
        return None

    def _is_signal_correct(self, signal: dict, horizon: str) -> bool | None:
        """Check if signal prediction was correct.

        Args:
            signal: Signal record
            horizon: Time horizon (1d/5d/20d)

        Returns:
            True if correct, False if incorrect, None if no data
        """
        price_future = signal["actual_exit_price"] or signal.get(f"price_at_{horizon}")
        if price_future is None:
            return None

        signal_type = signal["signal"]
        if signal_type == "BUY":
            return price_future > signal["price_at_signal"]
        if signal_type == "SELL":
            return price_future < signal["price_at_signal"]
        return None

    def _calculate_calibration(self, signals: list[dict], horizon: str) -> dict[str, float]:
        """Bucket by confidence, compute hit rate per bucket.

        Args:
            signals: List of signal outcome records
            horizon: Time horizon (1d/5d/20d)

        Returns:
            Dict mapping confidence bucket to hit rate
        """
        buckets: dict[str, list[dict]] = defaultdict(list)

        for sig in signals:
            if sig["signal"] == "HOLD":
                continue

            bucket = self._get_confidence_bucket(sig["confidence"])
            if bucket:
                buckets[bucket].append(sig)

        result = {}
        for bucket, sigs in buckets.items():
            hits = sum(1 for sig in sigs if self._is_signal_correct(sig, horizon) is True)
            total = sum(1 for sig in sigs if self._is_signal_correct(sig, horizon) is not None)
            result[bucket] = (hits / total) if total > 0 else 0.0

        return result

    def _calculate_strategy_accuracy(self, signals: list[dict], horizon: str) -> dict[str, float]:
        """Hit rate grouped by strategy.

        Args:
            signals: List of signal outcome records
            horizon: Time horizon (1d/5d/20d)

        Returns:
            Dict mapping strategy name to hit rate
        """
        by_strategy: dict[str, list[dict]] = defaultdict(list)

        for sig in signals:
            if sig["signal"] == "HOLD" or not sig["strategy_used"]:
                continue
            by_strategy[sig["strategy_used"]].append(sig)

        result = {}
        for strategy, sigs in by_strategy.items():
            hits = sum(1 for sig in sigs if self._is_signal_correct(sig, horizon) is True)
            total = sum(1 for sig in sigs if self._is_signal_correct(sig, horizon) is not None)
            result[strategy] = (hits / total) if total > 0 else 0.0

        return result

    def _calculate_regime_accuracy(self, signals: list[dict], horizon: str) -> dict[str, float] | None:
        """Hit rate grouped by regime.

        Args:
            signals: List of signal outcome records
            horizon: Time horizon (1d/5d/20d)

        Returns:
            Dict mapping regime to hit rate, or None if no regime data
        """
        by_regime: dict[str, list[dict]] = defaultdict(list)

        for sig in signals:
            if sig["signal"] == "HOLD" or not sig["regime"]:
                continue
            by_regime[sig["regime"]].append(sig)

        if not by_regime:
            return None

        result = {}
        for regime, sigs in by_regime.items():
            hits = sum(1 for sig in sigs if self._is_signal_correct(sig, horizon) is True)
            total = sum(1 for sig in sigs if self._is_signal_correct(sig, horizon) is not None)
            result[regime] = (hits / total) if total > 0 else 0.0

        return result

    def _calculate_avg_return(self, signals: list[dict], horizon: str) -> float:
        """Average return % from signal to T+Nd.

        Args:
            signals: List of signal outcome records
            horizon: Time horizon (1d/5d/20d)

        Returns:
            Average return percentage
        """
        returns = []

        for sig in signals:
            if sig["signal"] == "HOLD":
                continue

            price_future = sig["actual_exit_price"] or sig.get(f"price_at_{horizon}")
            if price_future is None or sig["price_at_signal"] == 0:
                continue

            ret = ((price_future - sig["price_at_signal"]) / sig["price_at_signal"]) * 100

            # Invert for SELL signals (short position)
            if sig["signal"] == "SELL":
                ret = -ret

            returns.append(ret)

        return sum(returns) / len(returns) if returns else 0.0

    def __repr__(self) -> str:
        """Return string representation."""
        return "SignalAccuracyCalculator()"
