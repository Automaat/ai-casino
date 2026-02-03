"""Ensemble strategy combining multiple trading strategies with weighted voting."""

from enum import StrEnum

import pandas as pd
from loguru import logger
from pydantic import BaseModel

from src.strategies.mean_reversion import MeanReversionIndicators, MeanReversionStrategy
from src.strategies.momentum import MomentumIndicators, MomentumStrategy, Signal
from src.strategies.trend_following import TrendFollowingIndicators, TrendFollowingStrategy

DEFAULT_WEIGHTS = {
    "momentum": 0.40,
    "mean_reversion": 0.25,
    "trend_following": 0.35,
}

CONFLICT_MARGIN_THRESHOLD = 0.10
WEIGHT_NORMALIZATION_TOLERANCE = 0.01


class AggregationMethod(StrEnum):
    """Aggregation method for combining strategy signals."""

    WEIGHTED_VOTING = "WEIGHTED_VOTING"
    MAJORITY_VOTE = "MAJORITY_VOTE"
    UNANIMOUS = "UNANIMOUS"


class StrategyResult(BaseModel):
    """Result from individual strategy."""

    name: str
    signal: Signal
    weight: float
    indicators: MomentumIndicators | MeanReversionIndicators | TrendFollowingIndicators


class EnsembleResult(BaseModel):
    """Aggregated result from ensemble of strategies."""

    signal: Signal
    confidence: float
    agreement_ratio: float
    strategy_results: list[StrategyResult]
    conflict_resolved: bool


class EnsembleStrategy:
    """Multi-strategy ensemble with weighted voting and conflict resolution."""

    def __init__(
        self,
        momentum: MomentumStrategy | None = None,
        mean_reversion: MeanReversionStrategy | None = None,
        trend_following: TrendFollowingStrategy | None = None,
        weights: dict[str, float] | None = None,
        aggregation: AggregationMethod = AggregationMethod.WEIGHTED_VOTING,
    ) -> None:
        """Initialize ensemble strategy.

        Args:
            momentum: Momentum strategy (default: new instance)
            mean_reversion: Mean reversion strategy (default: new instance)
            trend_following: Trend following strategy (default: new instance)
            weights: Strategy weights (default: momentum=0.4, mean_rev=0.25, trend=0.35)
            aggregation: Aggregation method (default: weighted voting)
        """
        self.momentum = momentum or MomentumStrategy()
        self.mean_reversion = mean_reversion or MeanReversionStrategy()
        # Use shorter SMAs (20/50) for ensemble - requires less data, more responsive
        self.trend_following = trend_following or TrendFollowingStrategy(sma_fast=20, sma_slow=50)
        self.weights = weights or DEFAULT_WEIGHTS.copy()
        self.aggregation = aggregation

        total_weight = sum(self.weights.values())
        if abs(total_weight - 1.0) > WEIGHT_NORMALIZATION_TOLERANCE:
            logger.warning(f"Weights sum to {total_weight}, normalizing to 1.0")
            self.weights = {k: v / total_weight for k, v in self.weights.items()}

        logger.info(f"Initialized EnsembleStrategy: {self.aggregation.value}, weights={self.weights}")

    def _run_strategies(self, data: pd.DataFrame) -> list[StrategyResult]:
        """Run all strategies and collect results.

        Args:
            data: OHLCV dataframe

        Returns:
            List of StrategyResult from each strategy
        """
        results = []

        signal, indicators = self.momentum.generate_signal(data)
        results.append(
            StrategyResult(
                name="momentum",
                signal=signal,
                weight=self.weights["momentum"],
                indicators=indicators,
            )
        )

        signal, indicators = self.mean_reversion.generate_signal(data)
        results.append(
            StrategyResult(
                name="mean_reversion",
                signal=signal,
                weight=self.weights["mean_reversion"],
                indicators=indicators,
            )
        )

        signal, indicators = self.trend_following.generate_signal(data)
        results.append(
            StrategyResult(
                name="trend_following",
                signal=signal,
                weight=self.weights["trend_following"],
                indicators=indicators,
            )
        )

        return results

    def _weighted_voting(self, results: list[StrategyResult]) -> tuple[Signal, float, bool]:
        """Aggregate signals using weighted voting.

        Args:
            results: Strategy results

        Returns:
            Tuple of (winning signal, weighted score, conflict_resolved flag)
        """
        signal_weights: dict[Signal, float] = {Signal.BUY: 0.0, Signal.SELL: 0.0, Signal.HOLD: 0.0}

        for result in results:
            signal_weights[result.signal] += result.weight

        sorted_signals = sorted(signal_weights.items(), key=lambda x: x[1], reverse=True)
        winner, winner_score = sorted_signals[0]
        runner_up_score = sorted_signals[1][1]

        margin = winner_score - runner_up_score
        conflict_resolved = False

        if margin < CONFLICT_MARGIN_THRESHOLD:
            conflict_resolved = True
            winner = Signal.HOLD

        if (
            signal_weights[Signal.BUY] > 0
            and signal_weights[Signal.SELL] > 0
            and abs(signal_weights[Signal.BUY] - signal_weights[Signal.SELL]) < CONFLICT_MARGIN_THRESHOLD
        ):
            conflict_resolved = True
            winner = Signal.HOLD

        return winner, winner_score, conflict_resolved

    def _majority_vote(self, results: list[StrategyResult]) -> tuple[Signal, float, bool]:
        """Aggregate signals using majority vote.

        Args:
            results: Strategy results

        Returns:
            Tuple of (winning signal, vote count ratio, conflict_resolved flag)
        """
        signal_counts: dict[Signal, int] = {Signal.BUY: 0, Signal.SELL: 0, Signal.HOLD: 0}

        for result in results:
            signal_counts[result.signal] += 1

        sorted_signals = sorted(signal_counts.items(), key=lambda x: x[1], reverse=True)
        winner, winner_count = sorted_signals[0]
        runner_up_count = sorted_signals[1][1]

        conflict_resolved = False
        if winner_count == runner_up_count:
            conflict_resolved = True
            winner = Signal.HOLD

        score = winner_count / len(results)
        return winner, score, conflict_resolved

    def _unanimous(self, results: list[StrategyResult]) -> tuple[Signal, float, bool]:
        """Require unanimous agreement for non-HOLD signal.

        Args:
            results: Strategy results

        Returns:
            Tuple of (signal, score, conflict_resolved flag)
        """
        signals = {r.signal for r in results}

        if len(signals) == 1:
            return results[0].signal, 1.0, False

        return Signal.HOLD, 0.0, True

    def _calculate_agreement_ratio(self, results: list[StrategyResult], winning_signal: Signal) -> float:
        """Calculate ratio of strategies agreeing with winning signal.

        Args:
            results: Strategy results
            winning_signal: The winning signal

        Returns:
            Agreement ratio (0.0-1.0)
        """
        return sum(r.weight for r in results if r.signal == winning_signal)

    def _calculate_signal_strength(self, results: list[StrategyResult]) -> float:
        """Calculate overall signal strength from indicators.

        Args:
            results: Strategy results

        Returns:
            Signal strength (0.0-1.0)
        """
        strength_sum = 0.0
        weight_sum = 0.0

        for result in results:
            indicators = result.indicators
            strength = 0.5

            if isinstance(indicators, MomentumIndicators):
                if indicators.rsi_oversold or indicators.rsi_overbought:
                    strength += 0.2
                if indicators.macd_bullish or indicators.macd_bearish:
                    strength += 0.2

            elif isinstance(indicators, MeanReversionIndicators):
                if indicators.oversold or indicators.overbought:
                    strength += 0.3
                bb_dist = abs(indicators.bb_percent - 0.5)
                strength += min(bb_dist * 0.4, 0.2)

            elif isinstance(indicators, TrendFollowingIndicators):
                if indicators.strong_trend:
                    strength += 0.3
                if indicators.sma_bullish_cross or indicators.sma_bearish_cross:
                    strength += 0.2

            strength_sum += min(strength, 1.0) * result.weight
            weight_sum += result.weight

        return strength_sum / weight_sum if weight_sum > 0 else 0.5

    def _calculate_confidence(
        self, agreement_ratio: float, weighted_score: float, signal_strength: float
    ) -> float:
        """Calculate overall confidence score.

        Formula: agreement_ratio * 0.4 + normalized_weighted_score * 0.4 + signal_strength * 0.2

        Args:
            agreement_ratio: Ratio of strategies agreeing
            weighted_score: Normalized weighted score of winning signal
            signal_strength: Aggregate signal strength from indicators

        Returns:
            Confidence score (0.0-1.0)
        """
        confidence = agreement_ratio * 0.4 + weighted_score * 0.4 + signal_strength * 0.2
        return min(max(confidence, 0.0), 1.0)

    def generate_signal(self, data: pd.DataFrame) -> tuple[Signal, EnsembleResult]:
        """Generate trading signal from ensemble of strategies.

        Args:
            data: OHLCV dataframe

        Returns:
            Tuple of (Signal, EnsembleResult)
        """
        logger.info(f"Running ensemble analysis with {self.aggregation.value}")

        results = self._run_strategies(data)

        if self.aggregation == AggregationMethod.WEIGHTED_VOTING:
            signal, weighted_score, conflict_resolved = self._weighted_voting(results)
        elif self.aggregation == AggregationMethod.MAJORITY_VOTE:
            signal, weighted_score, conflict_resolved = self._majority_vote(results)
        else:
            signal, weighted_score, conflict_resolved = self._unanimous(results)

        agreement_ratio = self._calculate_agreement_ratio(results, signal)
        signal_strength = self._calculate_signal_strength(results)
        confidence = self._calculate_confidence(agreement_ratio, weighted_score, signal_strength)

        ensemble_result = EnsembleResult(
            signal=signal,
            confidence=confidence,
            agreement_ratio=agreement_ratio,
            strategy_results=results,
            conflict_resolved=conflict_resolved,
        )

        strategy_signals = ", ".join(f"{r.name}={r.signal.value}" for r in results)
        logger.info(
            f"Ensemble signal: {signal.value} | confidence={confidence:.2f} | "
            f"agreement={agreement_ratio:.2f} | conflict_resolved={conflict_resolved} | [{strategy_signals}]"
        )

        return signal, ensemble_result

    def __repr__(self) -> str:
        """String representation."""
        return f"EnsembleStrategy(aggregation={self.aggregation.value}, weights={self.weights})"
