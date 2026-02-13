"""Profiling record repository for database operations."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from loguru import logger
from sqlalchemy import select

from src.daemon.state.models import ProfilingRecord
from src.database.models import ProfilingRecordORM
from src.database.repositories.base import BaseRepository

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


class ProfilingRecordRepository(BaseRepository[ProfilingRecord]):
    """Repository for profiling record persistence."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with database session.

        Args:
            session: SQLAlchemy async session
        """
        super().__init__(session)
        logger.debug("Initialized ProfilingRecordRepository")

    async def create(self, entity: ProfilingRecord) -> ProfilingRecord:
        """Create new profiling record.

        Args:
            entity: ProfilingRecord to persist

        Returns:
            Created ProfilingRecord
        """
        orm = ProfilingRecordORM(
            id=uuid.uuid4(),
            cycle_number=entity.cycle_number,
            timestamp=entity.timestamp,
            duration_seconds=Decimal(str(entity.duration_seconds)),
            profiling_overhead_percent=Decimal(str(entity.profiling_overhead_percent)),
            top_function=entity.top_function,
            top_function_cumtime=(
                Decimal(str(entity.top_function_cumtime)) if entity.top_function_cumtime is not None else None
            ),
            created_at=datetime.now(UTC),
        )
        self._session.add(orm)
        await self._session.commit()
        logger.info(f"Created profiling record: cycle {entity.cycle_number}")
        return entity

    async def get_by_id(self, entity_id: str) -> ProfilingRecord | None:
        """Get profiling record by ID.

        Args:
            entity_id: Profiling record UUID string

        Returns:
            ProfilingRecord if found, None otherwise
        """
        result = await self._session.execute(
            select(ProfilingRecordORM).where(ProfilingRecordORM.id == uuid.UUID(entity_id))
        )
        orm = result.scalar_one_or_none()
        return self._to_record(orm) if orm else None

    async def get_recent(self, limit: int = 100) -> list[ProfilingRecord]:
        """Get recent profiling records.

        Args:
            limit: Maximum number of records to return

        Returns:
            List of recent ProfilingRecords
        """
        result = await self._session.execute(
            select(ProfilingRecordORM).order_by(ProfilingRecordORM.timestamp.desc()).limit(limit)
        )
        return [self._to_record(orm) for orm in result.scalars().all()]

    def _to_record(self, orm: ProfilingRecordORM) -> ProfilingRecord:
        """Convert ORM model to ProfilingRecord.

        Args:
            orm: ProfilingRecordORM instance

        Returns:
            ProfilingRecord
        """
        return ProfilingRecord(
            cycle_number=orm.cycle_number,
            timestamp=orm.timestamp,
            duration_seconds=float(orm.duration_seconds),
            profiling_overhead_percent=float(orm.profiling_overhead_percent),
            top_function=orm.top_function,
            top_function_cumtime=float(orm.top_function_cumtime) if orm.top_function_cumtime is not None else None,
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return "ProfilingRecordRepository()"
