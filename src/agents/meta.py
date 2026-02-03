"""Meta-agent for dynamic strategy selection based on market regime."""

import pandas as pd
from loguru import logger
from pydantic import BaseModel

from src.metrics.tracker import MetricsTracker
from src.models.llm import LLMClient
from src.strategies.ensemble import EnsembleStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.momentum import MomentumStrategy
from src.strategies.regime import MarketRegime, MarketRegimeDetector, RegimeAnalysis
from src.strategies.trend_following import TrendFollowingStrategy

StrategyType = MomentumStrategy | MeanReversionStrategy | TrendFollowingStrategy | EnsembleStrategy

STRATEGY_REGIME_MAP: dict[MarketRegime, str] = {
    MarketRegime.TRENDING_BULLISH: "trend_following",
    MarketRegime.TRENDING_BEARISH: "trend_following",
    MarketRegime.RANGING: "mean_reversion",
    MarketRegime.HIGH_VOLATILITY: "momentum",
}

DEFAULT_WEIGHTS = {
    "momentum": 0.33,
    "mean_reversion": 0.33,
    "trend_following": 0.34,
}

LOW_CONFIDENCE_THRESHOLD = 0.5
REGIME_WEIGHT_BOOST = 0.2
PERFORMANCE_WEIGHT_BOOST = 0.1
WIN_RATE_THRESHOLD = 0.5


class StrategySelection(BaseModel):
    """Result of meta-agent strategy selection."""

    model_config = {"arbitrary_types_allowed": True}

    strategy_name: str
    strategy_instance: StrategyType
    ensemble_weights: dict[str, float] | None
    regime: MarketRegime
    regime_confidence: float
    reasoning: str
    confidence: float


class MetaAgent:
    """Agent that selects optimal trading strategy based on market regime and performance."""

    def __init__(
        self,
        llm_client: LLMClient,
        regime_detector: MarketRegimeDetector,
        metrics_tracker: MetricsTracker | None = None,
    ) -> None:
        """Initialize meta-agent.

        Args:
            llm_client: LLM client (for future use)
            regime_detector: Market regime detector
            metrics_tracker: Optional metrics tracker for performance-based selection
        """
        self.llm = llm_client
        self.regime_detector = regime_detector
        self.metrics_tracker = metrics_tracker

        logger.info("Initialized MetaAgent")

    def _create_strategy(self, strategy_name: str) -> StrategyType:
        """Create strategy instance by name.

        Args:
            strategy_name: Name of strategy to create

        Returns:
            Strategy instance
        """
        if strategy_name == "momentum":
            return MomentumStrategy()
        if strategy_name == "mean_reversion":
            return MeanReversionStrategy()
        if strategy_name == "trend_following":
            return TrendFollowingStrategy(sma_fast=20, sma_slow=50)
        msg = f"Unknown strategy: {strategy_name}"
        raise ValueError(msg)

    def _calculate_ensemble_weights(
        self,
        regime: MarketRegime,
    ) -> dict[str, float]:
        """Calculate ensemble weights based on regime and performance.

        Args:
            regime: Detected market regime
            regime_confidence: Confidence in regime detection

        Returns:
            Normalized strategy weights
        """
        weights = DEFAULT_WEIGHTS.copy()

        # Boost regime-matched strategy
        matched_strategy = STRATEGY_REGIME_MAP[regime]
        weights[matched_strategy] += REGIME_WEIGHT_BOOST

        # Performance-based boost if tracker available
        if self.metrics_tracker:
            self._apply_performance_boost(weights)

        # Normalize to sum=1.0
        total = sum(weights.values())
        return {k: round(v / total, 3) for k, v in weights.items()}

    def _apply_performance_boost(self, weights: dict[str, float]) -> None:
        """Apply performance-based weight adjustments.

        Args:
            weights: Current weights dict (modified in place)
        """
        try:
            metrics = self.metrics_tracker.calculate_metrics("30d")
            if metrics.total_decisions == 0:
                return

            # Boost based on win rate if above threshold
            if metrics.win_rate > WIN_RATE_THRESHOLD * 100:
                recent_score = (metrics.win_rate / 100 - WIN_RATE_THRESHOLD) * 2
                # Apply boost to all strategies proportionally
                # Future: track per-strategy performance
                for key in weights:
                    weights[key] += PERFORMANCE_WEIGHT_BOOST * recent_score

        except Exception as e:
            logger.warning(f"Failed to calculate performance metrics: {e}")

    async def select_strategy(
        self,
        symbol: str,
        market_data: pd.DataFrame,
    ) -> StrategySelection:
        """Select optimal strategy for current market conditions.

        Args:
            symbol: Stock ticker symbol
            market_data: OHLCV dataframe

        Returns:
            StrategySelection with chosen strategy and reasoning
        """
        logger.info(f"Selecting strategy for {symbol}")

        regime_analysis: RegimeAnalysis = self.regime_detector.detect_regime(market_data)
        regime = regime_analysis.regime
        regime_confidence = regime_analysis.confidence

        # Low confidence -> use ensemble with adjusted weights
        if regime_confidence < LOW_CONFIDENCE_THRESHOLD:
            weights = self._calculate_ensemble_weights(regime)
            strategy = EnsembleStrategy(weights=weights)
            reasoning = (
                f"Low regime confidence ({regime_confidence:.2f}), using ensemble. "
                f"Detected {regime.value}, weights adjusted: {weights}"
            )
            logger.info("Strategy: ensemble (low confidence fallback)")

            return StrategySelection(
                strategy_name="ensemble",
                strategy_instance=strategy,
                ensemble_weights=weights,
                regime=regime,
                regime_confidence=regime_confidence,
                reasoning=reasoning,
                confidence=regime_confidence,
            )

        # High confidence -> use regime-matched strategy
        strategy_name = STRATEGY_REGIME_MAP[regime]
        strategy = self._create_strategy(strategy_name)
        reasoning = (
            f"Regime: {regime.value} (confidence={regime_confidence:.2f}). "
            f"Selected {strategy_name} strategy. {regime_analysis.reasoning}"
        )
        logger.info(f"Strategy: {strategy_name} for {regime.value}")

        return StrategySelection(
            strategy_name=strategy_name,
            strategy_instance=strategy,
            ensemble_weights=None,
            regime=regime,
            regime_confidence=regime_confidence,
            reasoning=reasoning,
            confidence=regime_confidence,
        )

    def __repr__(self) -> str:
        """String representation."""
        has_tracker = self.metrics_tracker is not None
        return f"MetaAgent(has_metrics_tracker={has_tracker})"
