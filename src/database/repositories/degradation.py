"""Degradation record repository for database operations."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from loguru import logger
from sqlalchemy import select

from src.daemon.state.models import DegradationRecord
from src.database.models import DegradationRecordORM
from src.database.repositories.base import BaseRepository

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


class DegradationRecordRepository(BaseRepository[DegradationRecord]):
    """Repository for degradation record persistence."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with database session.

        Args:
            session: SQLAlchemy async session
        """
        super().__init__(session)
        logger.debug("Initialized DegradationRecordRepository")

    async def create(self, entity: DegradationRecord) -> DegradationRecord:
        """Create new degradation record.

        Args:
            entity: DegradationRecord to persist

        Returns:
            Created DegradationRecord
        """
        orm = DegradationRecordORM(
            id=uuid.uuid4(),
            timestamp=entity.timestamp,
            tier=entity.tier,
            unavailable_services=entity.unavailable_services,
            confidence_adjustment=Decimal(str(entity.confidence_adjustment)),
            halt_reason=entity.halt_reason,
            created_at=datetime.now(UTC),
        )
        self._session.add(orm)
        await self._session.commit()
        logger.info(f"Created degradation record: tier={entity.tier}")
        return entity

    async def get_by_id(self, entity_id: str) -> DegradationRecord | None:
        """Get degradation record by ID.

        Args:
            entity_id: Degradation record UUID string

        Returns:
            DegradationRecord if found, None otherwise
        """
        result = await self._session.execute(
            select(DegradationRecordORM).where(DegradationRecordORM.id == uuid.UUID(entity_id))
        )
        orm = result.scalar_one_or_none()
        return self._to_record(orm) if orm else None

    async def get_recent(self, limit: int = 100) -> list[DegradationRecord]:
        """Get recent degradation records.

        Args:
            limit: Maximum number of records to return

        Returns:
            List of recent DegradationRecords
        """
        result = await self._session.execute(
            select(DegradationRecordORM).order_by(DegradationRecordORM.timestamp.desc()).limit(limit)
        )
        return [self._to_record(orm) for orm in result.scalars().all()]

    def _to_record(self, orm: DegradationRecordORM) -> DegradationRecord:
        """Convert ORM model to DegradationRecord.

        Args:
            orm: DegradationRecordORM instance

        Returns:
            DegradationRecord
        """
        return DegradationRecord(
            timestamp=orm.timestamp,
            tier=orm.tier,
            unavailable_services=orm.unavailable_services if isinstance(orm.unavailable_services, list) else [],
            confidence_adjustment=float(orm.confidence_adjustment),
            halt_reason=orm.halt_reason,
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return "DegradationRecordRepository()"
