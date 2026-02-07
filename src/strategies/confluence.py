"""Multi-timeframe confluence calculation."""

from typing import ClassVar

from loguru import logger

from src.strategies.momentum import Signal
from src.strategies.timeframe import Timeframe, TimeframeResult


class ConfluenceCalculator:
    """Calculate signal confluence across multiple timeframes."""

    TIMEFRAME_WEIGHTS: ClassVar[dict[Timeframe, float]] = {
        Timeframe.DAILY: 0.60,
        Timeframe.HOURLY: 0.40,
        Timeframe.FIFTEEN_MIN: 0.15,
    }

    SIGNAL_VALUES: ClassVar[dict[Signal, float]] = {Signal.BUY: 1.0, Signal.HOLD: 0.0, Signal.SELL: -1.0}

    CONFLICT_MARGIN_THRESHOLD = 0.15
    HIGH_CONFLUENCE_THRESHOLD = 0.8
    LOW_CONFLUENCE_THRESHOLD = 0.5
    SIGNAL_THRESHOLD = 0.15

    CONFIDENCE_BOOST_HIGH_CONFLUENCE = 0.10
    CONFIDENCE_PENALTY_LOW_CONFLUENCE = 0.20

    def calculate_confluence(self, results: dict[Timeframe, TimeframeResult]) -> tuple[Signal, float, bool]:
        """Calculate weighted signal confluence.

        Args:
            results: Timeframe analysis results

        Returns:
            Tuple of (final_signal, confluence_score, conflict_detected)
        """
        if not results:
            msg = "No timeframe results provided"
            raise ValueError(msg)

        weighted_sum = 0.0
        total_weight = 0.0

        for timeframe, result in results.items():
            weight = self.TIMEFRAME_WEIGHTS.get(timeframe, 0.0)
            signal_value = self.SIGNAL_VALUES[result.signal]
            weighted_sum += weight * signal_value
            total_weight += weight

        if total_weight == 0:
            msg = "No valid timeframe weights"
            raise ValueError(msg)

        weighted_avg = weighted_sum / total_weight

        final_signal = self._weighted_avg_to_signal(weighted_avg)

        conflict_detected = self._detect_conflict(results, weighted_avg)

        if conflict_detected and abs(weighted_avg) < self.CONFLICT_MARGIN_THRESHOLD:
            logger.warning(f"Conflict detected with margin {abs(weighted_avg):.2f}, forcing HOLD")
            final_signal = Signal.HOLD

        confluence_score = self._calculate_agreement_score(results, final_signal)

        logger.info(
            f"Confluence: signal={final_signal}, score={confluence_score:.2f}, "
            f"conflict={conflict_detected}, weighted_avg={weighted_avg:.2f}"
        )

        return final_signal, confluence_score, conflict_detected

    def adjust_confidence(self, base_confidence: float, confluence_score: float) -> float:
        """Adjust confidence based on confluence score.

        Args:
            base_confidence: Original confidence value
            confluence_score: Confluence score (0-1)

        Returns:
            Adjusted confidence value (0-1)
        """
        if confluence_score >= self.HIGH_CONFLUENCE_THRESHOLD:
            adjusted = min(1.0, base_confidence + self.CONFIDENCE_BOOST_HIGH_CONFLUENCE)
            logger.debug(f"High confluence boost: {base_confidence:.2f} -> {adjusted:.2f}")
            return adjusted
        if confluence_score < self.LOW_CONFLUENCE_THRESHOLD:
            adjusted = max(0.0, base_confidence - self.CONFIDENCE_PENALTY_LOW_CONFLUENCE)
            logger.debug(f"Low confluence penalty: {base_confidence:.2f} -> {adjusted:.2f}")
            return adjusted
        return base_confidence

    def _weighted_avg_to_signal(self, weighted_avg: float) -> Signal:
        """Convert weighted average to signal."""
        if weighted_avg > self.SIGNAL_THRESHOLD:
            return Signal.BUY
        if weighted_avg < -self.SIGNAL_THRESHOLD:
            return Signal.SELL
        return Signal.HOLD

    def _detect_conflict(self, results: dict[Timeframe, TimeframeResult], weighted_avg: float) -> bool:
        """Detect conflicting signals (e.g., daily BUY + hourly SELL)."""
        signals = {result.signal for result in results.values()}

        if Signal.BUY in signals and Signal.SELL in signals:
            return True

        primary = results.get(Timeframe.DAILY)
        if primary and primary.signal != Signal.HOLD:
            primary_value = self.SIGNAL_VALUES[primary.signal]
            if (primary_value > 0 and weighted_avg < 0) or (primary_value < 0 and weighted_avg > 0):
                return True

        return False

    def _calculate_agreement_score(
        self, results: dict[Timeframe, TimeframeResult], final_signal: Signal
    ) -> float:
        """Calculate agreement score between timeframes and final signal."""
        if not results:
            return 0.0

        final_value = self.SIGNAL_VALUES[final_signal]

        weighted_agreement = 0.0
        total_weight = 0.0

        for timeframe, result in results.items():
            weight = self.TIMEFRAME_WEIGHTS.get(timeframe, 0.0)
            signal_value = self.SIGNAL_VALUES[result.signal]

            if final_value == 0:
                agreement = 1.0 if signal_value == 0 else 0.5
            else:
                agreement = 1.0 if signal_value * final_value > 0 else 0.0

            weighted_agreement += weight * agreement
            total_weight += weight

        return weighted_agreement / total_weight if total_weight > 0 else 0.0

    def __repr__(self) -> str:
        """String representation."""
        return f"ConfluenceCalculator(weights={self.TIMEFRAME_WEIGHTS})"
