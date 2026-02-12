"""Adaptive confidence threshold manager."""

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from loguru import logger
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from src.coordinator.models import AdaptiveThresholdConfig
    from src.daemon.state.models import SignalOutcome
    from src.database.repositories.signal_outcome import SignalOutcomeRepository


class AdaptiveThresholds(BaseModel):
    """Adaptive threshold state."""

    buy_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    sell_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    last_updated: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_reset: datetime = Field(default_factory=lambda: datetime.now(UTC))
    adaptation_count: int = Field(default=0, ge=0)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"AdaptiveThresholds(buy={self.buy_threshold:.2f}, sell={self.sell_threshold:.2f}, "
            f"adaptations={self.adaptation_count})"
        )


class AdaptiveThresholdManager:
    """Self-regulating threshold manager based on rolling accuracy."""

    def __init__(
        self,
        config: AdaptiveThresholdConfig,
        signal_outcome_repo: SignalOutcomeRepository,
    ) -> None:
        """Initialize threshold manager.

        Args:
            config: Adaptive threshold configuration
            signal_outcome_repo: Repository for signal outcomes
        """
        self._config = config
        self._repo = signal_outcome_repo
        self._thresholds = AdaptiveThresholds(
            buy_threshold=config.min_threshold,
            sell_threshold=config.min_threshold,
        )
        logger.info(f"Initialized AdaptiveThresholdManager: {self._thresholds}")

    async def update_thresholds(self) -> AdaptiveThresholds:
        """Update thresholds based on recent accuracy.

        Returns:
            Updated thresholds
        """
        # Check if weekly reset needed
        if self._should_reset_weekly():
            self._reset_thresholds()
            return self._thresholds

        # Query recent outcomes for BUY and SELL
        try:
            buy_outcomes = await self._repo.get_recent_outcomes(
                window=self._config.min_sample_size,
                signal_type="BUY",
            )
            sell_outcomes = await self._repo.get_recent_outcomes(
                window=self._config.min_sample_size,
                signal_type="SELL",
            )
        except Exception as e:
            logger.opt(exception=True).error(f"Failed to query signal outcomes: {e}")
            return self._thresholds

        # Calculate accuracies
        buy_accuracy = self._calculate_accuracy(buy_outcomes, "BUY")
        sell_accuracy = self._calculate_accuracy(sell_outcomes, "SELL")

        # Log current state
        logger.info(
            f"Accuracy metrics: BUY={buy_accuracy:.1%} ({len(buy_outcomes)} samples), "
            f"SELL={sell_accuracy:.1%} ({len(sell_outcomes)} samples)"
        )

        # Apply adjustment rules
        old_buy = self._thresholds.buy_threshold
        old_sell = self._thresholds.sell_threshold

        if buy_accuracy is not None:
            self._adjust_buy_threshold(buy_accuracy)

        if sell_accuracy is not None:
            self._adjust_sell_threshold(sell_accuracy)

        # Update metadata
        if old_buy != self._thresholds.buy_threshold or old_sell != self._thresholds.sell_threshold:
            self._thresholds.last_updated = datetime.now(UTC)
            self._thresholds.adaptation_count += 1

            logger.info(
                f"Thresholds adjusted: BUY {old_buy:.2f}→{self._thresholds.buy_threshold:.2f}, "
                f"SELL {old_sell:.2f}→{self._thresholds.sell_threshold:.2f}"
            )

        return self._thresholds

    def get_threshold(self, signal_type: str) -> float:
        """Get threshold for signal type.

        Args:
            signal_type: BUY/SELL/HOLD

        Returns:
            Threshold value (0.0-1.0)
        """
        if signal_type == "BUY":
            return self._thresholds.buy_threshold
        if signal_type == "SELL":
            return self._thresholds.sell_threshold
        return self._config.min_threshold

    def get_thresholds(self) -> AdaptiveThresholds:
        """Get full threshold state.

        Returns:
            Current thresholds
        """
        return self._thresholds

    def set_thresholds(self, thresholds: AdaptiveThresholds) -> None:
        """Restore thresholds from persistence.

        Args:
            thresholds: Threshold state to restore
        """
        self._thresholds = thresholds
        logger.info(f"Restored thresholds: {thresholds}")

    def _calculate_accuracy(
        self,
        outcomes: list[SignalOutcome],
        signal_type: str,
    ) -> float | None:
        """Calculate hit rate for signal type.

        Args:
            outcomes: List of signal outcomes
            signal_type: BUY/SELL

        Returns:
            Accuracy (0.0-1.0) or None if insufficient data
        """
        if len(outcomes) < self._config.min_sample_size:
            logger.debug(
                f"Insufficient data for {signal_type}: {len(outcomes)} < {self._config.min_sample_size}"
            )
            return None

        hits = 0
        total = 0

        for outcome in outcomes:
            # Skip pending outcomes (no price_at_5d yet)
            if outcome.price_at_5d is None:
                continue

            # Calculate hit based on direction
            if signal_type == "BUY":
                is_hit = outcome.price_at_5d > outcome.price_at_signal
            elif signal_type == "SELL":
                is_hit = outcome.price_at_5d < outcome.price_at_signal
            else:
                continue

            if is_hit:
                hits += 1
            total += 1

        if total == 0:
            logger.warning(f"No completed outcomes for {signal_type} (all pending)")
            return None

        accuracy = hits / total
        logger.debug(f"{signal_type} accuracy: {accuracy:.1%} ({hits}/{total} hits)")
        return accuracy

    def _adjust_buy_threshold(self, accuracy: float) -> None:
        """Adjust BUY threshold based on accuracy.

        Args:
            accuracy: Current BUY accuracy (0.0-1.0)
        """
        if accuracy < self._config.buy_accuracy_threshold:
            # Increase threshold when accuracy is low
            new_threshold = min(
                self._thresholds.buy_threshold + self._config.buy_increase_step,
                self._config.max_threshold,
            )
            self._thresholds.buy_threshold = new_threshold
            logger.info(
                f"BUY accuracy {accuracy:.1%} < {self._config.buy_accuracy_threshold:.1%}, "
                f"increasing threshold"
            )

    def _adjust_sell_threshold(self, accuracy: float) -> None:
        """Adjust SELL threshold based on accuracy.

        Args:
            accuracy: Current SELL accuracy (0.0-1.0)
        """
        if accuracy > self._config.sell_accuracy_threshold:
            # Decrease threshold when accuracy is high
            new_threshold = max(
                self._thresholds.sell_threshold - self._config.sell_decrease_step,
                self._config.min_threshold,
            )
            self._thresholds.sell_threshold = new_threshold
            logger.info(
                f"SELL accuracy {accuracy:.1%} > {self._config.sell_accuracy_threshold:.1%}, "
                f"decreasing threshold"
            )

    def _should_reset_weekly(self) -> bool:
        """Check if weekly reset is needed.

        Returns:
            True if 7+ days since last reset
        """
        if not self._config.weekly_reset_enabled:
            return False

        days_since_reset = (datetime.now(UTC) - self._thresholds.last_reset).days
        return days_since_reset >= 7

    def _reset_thresholds(self) -> None:
        """Reset thresholds to base values."""
        self._thresholds.buy_threshold = self._config.min_threshold
        self._thresholds.sell_threshold = self._config.min_threshold
        self._thresholds.last_reset = datetime.now(UTC)
        self._thresholds.adaptation_count = 0
        logger.info("Weekly reset: thresholds reset to base values")

    def __repr__(self) -> str:
        """String representation."""
        return f"AdaptiveThresholdManager(config={self._config}, thresholds={self._thresholds})"
