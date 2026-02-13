"""Strategy state manager."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from loguru import logger
from pydantic import PrivateAttr

from src.daemon.state.managers.base import StateManager
from src.daemon.state.models import DegradationRecord, GamePlanRecord

if TYPE_CHECKING:
    from src.daemon.degradation import DegradationContext
    from src.database.repositories.degradation import DegradationRecordRepository
    from src.database.repositories.game_plan import GamePlanRecordRepository
    from src.database.repositories.metadata import MetadataRepository


class StrategyStateManager(StateManager):
    """Daily planning, degradation tracking, error logging."""

    _metadata_repository: MetadataRepository | None = PrivateAttr(default=None)
    _game_plan_repository: GamePlanRecordRepository | None = PrivateAttr(default=None)
    _degradation_repository: DegradationRecordRepository | None = PrivateAttr(default=None)

    _game_plan_cache: list[GamePlanRecord] | None = PrivateAttr(default=None)
    _degradation_cache: list[DegradationRecord] | None = PrivateAttr(default=None)

    def set_repositories(
        self,
        metadata_repository: MetadataRepository,
        game_plan_repository: GamePlanRecordRepository,
        degradation_repository: DegradationRecordRepository,
    ) -> None:
        """Inject repositories."""
        self._metadata_repository = metadata_repository
        self._game_plan_repository = game_plan_repository
        self._degradation_repository = degradation_repository
        logger.debug("StrategyStateManager repositories injected")

    async def get_last_game_plan(self) -> datetime | None:
        """Get last game plan timestamp from DB."""
        if not self._metadata_repository:
            return None
        return await self._metadata_repository.get_datetime("strategy.last_game_plan")

    async def get_last_degradation(self) -> datetime | None:
        """Get last degradation timestamp from DB."""
        if not self._metadata_repository:
            return None
        return await self._metadata_repository.get_datetime("strategy.last_degradation")

    async def get_last_health_check(self) -> datetime | None:
        """Get last health check timestamp from DB."""
        if not self._metadata_repository:
            return None
        return await self._metadata_repository.get_datetime("strategy.last_health_check")

    async def set_last_health_check(self, value: datetime | None) -> None:
        """Set last health check timestamp in DB."""
        if self._metadata_repository and value is not None:
            await self._metadata_repository.set("strategy.last_health_check", value)

    async def get_market_events(self, limit: int | None = None) -> list[dict]:
        """Get market events from DB metadata.

        Args:
            limit: Max number of events to return (optional)

        Returns:
            List of market events
        """
        if not self._metadata_repository:
            return []
        value = await self._metadata_repository.get("strategy.market_events")
        events = value if isinstance(value, list) else []
        if limit is not None and limit > 0:
            return events[-limit:]
        return events

    async def get_errors(self) -> list[str]:
        """Get errors from DB metadata."""
        if not self._metadata_repository:
            return []
        value = await self._metadata_repository.get("strategy.errors")
        return value if isinstance(value, list) else []

    async def get_game_plan_history(self, limit: int = 30) -> list[GamePlanRecord]:
        """Get game plan history with lazy loading."""
        if not self._game_plan_repository:
            return []
        if self._game_plan_cache is None:
            self._game_plan_cache = await self._game_plan_repository.get_recent(limit)
        return self._game_plan_cache

    async def get_degradation_history(self, limit: int = 100) -> list[DegradationRecord]:
        """Get degradation history with lazy loading."""
        if not self._degradation_repository:
            return []
        if self._degradation_cache is None:
            self._degradation_cache = await self._degradation_repository.get_recent(limit)
        return self._degradation_cache

    async def record_game_plan(
        self,
        priority_symbols: list[str],
        risk_stance: str,
        sector_focus: list[str],
    ) -> None:
        """Record game plan generation."""
        now = datetime.now(UTC)
        record = GamePlanRecord(
            timestamp=now,
            priority_symbols=priority_symbols,
            risk_stance=risk_stance,
            sector_focus=sector_focus,
        )

        if self._game_plan_repository:
            await self._game_plan_repository.create(record)
        if self._metadata_repository:
            await self._metadata_repository.set("strategy.last_game_plan", now)

        self._game_plan_cache = None

    async def record_degradation(self, context: DegradationContext) -> None:
        """Record degradation event."""
        now = datetime.now(UTC)
        record = DegradationRecord(
            timestamp=now,
            tier=context.tier.value,
            unavailable_services=context.unavailable_services,
            confidence_adjustment=context.confidence_adjustment,
            halt_reason=context.halt_reason,
        )

        if self._degradation_repository:
            await self._degradation_repository.create(record)
        if self._metadata_repository:
            await self._metadata_repository.set("strategy.last_degradation", now)

        self._degradation_cache = None

    async def record_error(self, error: str) -> None:
        """Record an error to metadata."""
        if not self._metadata_repository:
            return

        timestamp = datetime.now(tz=UTC).isoformat()
        error_entry = f"{timestamp}: {error}"

        # Get existing errors
        errors = await self.get_errors()
        errors.append(error_entry)

        # Cap at 100 errors
        if len(errors) > 100:
            errors = errors[-50:]

        await self._metadata_repository.set("strategy.errors", errors)

    def __repr__(self) -> str:
        """Return string representation."""
        return "StrategyStateManager()"
