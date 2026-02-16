"""Health report repository for database operations."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from loguru import logger
from sqlalchemy import select

from src.daemon.state.models import HealthReportRecord
from src.database.models import HealthReportORM
from src.database.repositories.base import BaseRepository

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


class HealthReportRepository(BaseRepository[HealthReportRecord]):
    """Repository for health report persistence."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with database session.

        Args:
            session: SQLAlchemy async session
        """
        super().__init__(session)

    async def create(self, entity: HealthReportRecord) -> HealthReportRecord:
        """Create new health report record.

        Args:
            entity: HealthReportRecord to persist

        Returns:
            Created HealthReportRecord with ID
        """
        orm = HealthReportORM(
            id=uuid.uuid4(),
            timestamp=entity.timestamp,
            overall_status=entity.overall_status,
            service_checks=entity.service_checks,
            cleanup_results=entity.cleanup_results,
            total_duration_ms=Decimal(str(entity.total_duration_ms)),
            created_at=datetime.now(UTC),
        )
        self._session.add(orm)
        await self._session.commit()
        logger.info(f"Created health report: {orm.id} (status={entity.overall_status})")
        entity.id = str(orm.id)
        return entity

    async def get_by_id(self, entity_id: str) -> HealthReportRecord | None:
        """Get health report by ID.

        Args:
            entity_id: Health report UUID string

        Returns:
            HealthReportRecord if found, None otherwise
        """
        result = await self._session.execute(
            select(HealthReportORM).where(HealthReportORM.id == uuid.UUID(entity_id))
        )
        orm = result.scalar_one_or_none()
        return self._to_record(orm) if orm else None

    async def get_recent(self, limit: int = 100) -> list[HealthReportRecord]:
        """Get recent health reports.

        Args:
            limit: Maximum number of reports to return

        Returns:
            List of HealthReportRecords ordered by timestamp desc
        """
        result = await self._session.execute(
            select(HealthReportORM).order_by(HealthReportORM.timestamp.desc()).limit(limit)
        )
        return [self._to_record(orm) for orm in result.scalars().all()]

    async def get_by_status(self, status: str, limit: int = 100) -> list[HealthReportRecord]:
        """Get health reports by status.

        Args:
            status: Health status to filter by (HEALTHY, DEGRADED, UNHEALTHY)
            limit: Maximum number of reports to return

        Returns:
            List of HealthReportRecords with given status
        """
        result = await self._session.execute(
            select(HealthReportORM)
            .where(HealthReportORM.overall_status == status)
            .order_by(HealthReportORM.timestamp.desc())
            .limit(limit)
        )
        return [self._to_record(orm) for orm in result.scalars().all()]

    def _to_record(self, orm: HealthReportORM) -> HealthReportRecord:
        """Convert ORM model to HealthReportRecord.

        Args:
            orm: HealthReportORM instance

        Returns:
            HealthReportRecord
        """
        return HealthReportRecord(
            id=str(orm.id),
            timestamp=orm.timestamp,
            overall_status=orm.overall_status,
            service_checks=orm.service_checks,
            cleanup_results=orm.cleanup_results,
            total_duration_ms=float(orm.total_duration_ms),
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return "HealthReportRepository()"
