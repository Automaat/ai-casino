"""Sector rotation record repository for database operations."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from loguru import logger
from sqlalchemy import select

from src.daemon.state.models import SectorRotationRecord
from src.database.models import SectorRotationRecordORM
from src.database.repositories.base import BaseRepository

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


class SectorRotationRecordRepository(BaseRepository[SectorRotationRecord]):
    """Repository for sector rotation record persistence."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with database session.

        Args:
            session: SQLAlchemy async session
        """
        super().__init__(session)

    async def create(self, entity: SectorRotationRecord) -> SectorRotationRecord:
        """Create new sector rotation record.

        Args:
            entity: SectorRotationRecord to persist

        Returns:
            Created SectorRotationRecord
        """
        orm = SectorRotationRecordORM(
            id=uuid.uuid4(),
            timestamp=entity.timestamp,
            leading_sectors=entity.leading_sectors,
            lagging_sectors=entity.lagging_sectors,
            sector_strengths=entity.sector_strengths,
            sector_momenta=entity.sector_momenta,
            flagged_positions=entity.flagged_positions,
            created_at=datetime.now(UTC),
        )
        self._session.add(orm)
        await self._session.commit()
        logger.info(f"Created sector rotation record: {len(entity.leading_sectors)} leading sectors")
        return entity

    async def get_by_id(self, entity_id: str) -> SectorRotationRecord | None:
        """Get sector rotation record by ID.

        Args:
            entity_id: Sector rotation record UUID string

        Returns:
            SectorRotationRecord if found, None otherwise
        """
        result = await self._session.execute(
            select(SectorRotationRecordORM).where(SectorRotationRecordORM.id == uuid.UUID(entity_id))
        )
        orm = result.scalar_one_or_none()
        return self._to_record(orm) if orm else None

    async def get_recent(self, limit: int = 100) -> list[SectorRotationRecord]:
        """Get recent sector rotation records.

        Args:
            limit: Maximum number of records to return

        Returns:
            List of recent SectorRotationRecords
        """
        result = await self._session.execute(
            select(SectorRotationRecordORM).order_by(SectorRotationRecordORM.timestamp.desc()).limit(limit)
        )
        return [self._to_record(orm) for orm in result.scalars().all()]

    def _to_record(self, orm: SectorRotationRecordORM) -> SectorRotationRecord:
        """Convert ORM model to SectorRotationRecord.

        Args:
            orm: SectorRotationRecordORM instance

        Returns:
            SectorRotationRecord
        """
        return SectorRotationRecord(
            timestamp=orm.timestamp,
            leading_sectors=orm.leading_sectors if isinstance(orm.leading_sectors, list) else [],
            lagging_sectors=orm.lagging_sectors if isinstance(orm.lagging_sectors, list) else [],
            sector_strengths=orm.sector_strengths if isinstance(orm.sector_strengths, dict) else {},
            sector_momenta=orm.sector_momenta if isinstance(orm.sector_momenta, dict) else {},
            flagged_positions=orm.flagged_positions if isinstance(orm.flagged_positions, list) else [],
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return "SectorRotationRecordRepository()"
