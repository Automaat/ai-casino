"""Economic calendar signal repository for database operations."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import select

from src.daemon.events import (
    EconomicEvent,
    EconomicEventSignal,
    EconomicImpact,
    EconomicRecommendation,
    EconomicRiskLevel,
)
from src.database.models import EconomicCalendarSignalORM
from src.database.repositories.base import BaseRepository

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


class EconomicCalendarSignalRepository(BaseRepository[EconomicEventSignal]):
    """Repository for economic calendar signal persistence."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with database session.

        Args:
            session: SQLAlchemy async session
        """
        super().__init__(session)

    async def create(self, entity: EconomicEventSignal) -> EconomicEventSignal:
        """Create new economic calendar signal record.

        Args:
            entity: EconomicEventSignal to persist

        Returns:
            Created EconomicEventSignal
        """
        events_json = [
            {
                "event_id": e.event_id,
                "country": e.country,
                "event": e.event,
                "impact": e.impact.value,
                "scheduled_at": e.scheduled_at.isoformat(),
                "actual": e.actual,
                "estimate": e.estimate,
                "prev": e.prev,
            }
            for e in entity.upcoming_events
        ]

        orm = EconomicCalendarSignalORM(
            id=uuid.uuid4(),
            risk_level=entity.risk_level.value,
            recommendation=entity.recommendation.value,
            reason=entity.reason,
            upcoming_events=events_json,
            avoid_until=entity.avoid_until,
            computed_at=entity.computed_at,
        )
        self._session.add(orm)
        await self._session.commit()
        return entity

    async def get_by_id(self, entity_id: str) -> EconomicEventSignal | None:
        """Get signal by ID.

        Args:
            entity_id: UUID string

        Returns:
            EconomicEventSignal if found, None otherwise
        """
        result = await self._session.execute(
            select(EconomicCalendarSignalORM).where(EconomicCalendarSignalORM.id == uuid.UUID(entity_id))
        )
        orm = result.scalar_one_or_none()
        return self._to_domain(orm) if orm else None

    async def get_latest(self) -> EconomicEventSignal | None:
        """Get most recent economic calendar signal.

        Returns:
            Most recent EconomicEventSignal or None
        """
        result = await self._session.execute(
            select(EconomicCalendarSignalORM).order_by(EconomicCalendarSignalORM.computed_at.desc()).limit(1)
        )
        orm = result.scalar_one_or_none()
        return self._to_domain(orm) if orm else None

    def _to_domain(self, orm: EconomicCalendarSignalORM) -> EconomicEventSignal:
        """Convert ORM model to EconomicEventSignal.

        Args:
            orm: EconomicCalendarSignalORM instance

        Returns:
            EconomicEventSignal domain model
        """
        events: list[EconomicEvent] = []
        if isinstance(orm.upcoming_events, list):
            for e_dict in orm.upcoming_events:
                scheduled_raw = e_dict.get("scheduled_at", "")
                scheduled_at = datetime.fromisoformat(scheduled_raw) if scheduled_raw else datetime.now(UTC)
                events.append(
                    EconomicEvent(
                        event_id=e_dict.get("event_id", ""),
                        country=e_dict.get("country", ""),
                        event=e_dict.get("event", ""),
                        impact=EconomicImpact(e_dict.get("impact", "low")),
                        scheduled_at=scheduled_at,
                        actual=e_dict.get("actual"),
                        estimate=e_dict.get("estimate"),
                        prev=e_dict.get("prev"),
                    )
                )

        return EconomicEventSignal(
            upcoming_events=events,
            risk_level=EconomicRiskLevel(orm.risk_level),
            recommendation=EconomicRecommendation(orm.recommendation),
            reason=orm.reason,
            computed_at=orm.computed_at,
            avoid_until=orm.avoid_until,
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return "EconomicCalendarSignalRepository()"
