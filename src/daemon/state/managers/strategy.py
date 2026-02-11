"""Strategy state manager."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from pydantic import Field

from src.daemon.state.managers.base import StateManager
from src.daemon.state.models import DegradationRecord, GamePlanRecord

if TYPE_CHECKING:
    from src.daemon.degradation import DegradationContext


class StrategyStateManager(StateManager):
    """Daily planning, degradation tracking, error logging."""

    # Game plan
    last_game_plan: datetime | None = None
    game_plan_history: list[GamePlanRecord] = Field(default_factory=list)

    # Degradation
    last_degradation: datetime | None = None
    degradation_history: list[DegradationRecord] = Field(default_factory=list)

    # Events
    market_events: list[dict] = Field(default_factory=list)

    # Health
    last_health_check: datetime | None = None

    # Errors
    errors: list[str] = Field(default_factory=list)

    def record_game_plan(
        self,
        priority_symbols: list[str],
        risk_stance: str,
        sector_focus: list[str],
    ) -> None:
        """Record game plan generation.

        Args:
            priority_symbols: Priority symbols for the day
            risk_stance: Risk stance (AGGRESSIVE/DEFENSIVE/NEUTRAL)
            sector_focus: Sector focus list
        """
        now = datetime.now(UTC)

        self.game_plan_history.append(
            GamePlanRecord(
                timestamp=now,
                priority_symbols=priority_symbols,
                risk_stance=risk_stance,
                sector_focus=sector_focus,
            )
        )
        self.last_game_plan = now
        self.game_plan_history = self._cap_history(self.game_plan_history, 30, 30)

    def record_degradation(self, context: DegradationContext) -> None:
        """Record degradation event.

        Args:
            context: Degradation context
        """
        now = datetime.now(UTC)
        self.degradation_history.append(
            DegradationRecord(
                timestamp=now,
                tier=context.tier.value,
                unavailable_services=context.unavailable_services,
                confidence_adjustment=context.confidence_adjustment,
                halt_reason=context.halt_reason,
            )
        )
        self.last_degradation = now
        self.degradation_history = self._cap_history(self.degradation_history, 100, 100)

    def record_error(self, error: str) -> None:
        """Record an error.

        Args:
            error: Error message
        """
        timestamp = datetime.now(tz=UTC).isoformat()
        self.errors.append(f"{timestamp}: {error}")
        self.errors = self._cap_history(self.errors, 100, 50)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"StrategyStateManager(errors={len(self.errors)})"
