"""Optimization record repository for database operations."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from loguru import logger
from sqlalchemy import select

from src.daemon.state.models import OptimizationRecord
from src.database.models import OptimizationRecordORM
from src.database.repositories.base import BaseRepository

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


class OptimizationRecordRepository(BaseRepository[OptimizationRecord]):
    """Repository for optimization record persistence."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with database session.

        Args:
            session: SQLAlchemy async session
        """
        super().__init__(session)
        logger.debug("Initialized OptimizationRecordRepository")

    async def create(self, entity: OptimizationRecord) -> OptimizationRecord:
        """Create new optimization record.

        Args:
            entity: OptimizationRecord to persist

        Returns:
            Created OptimizationRecord
        """
        orm = OptimizationRecordORM(
            id=uuid.uuid4(),
            timestamp=entity.timestamp,
            symbols_optimized=entity.symbols_optimized,
            symbols_skipped=entity.symbols_skipped,
            total_time_seconds=Decimal(str(entity.total_time_seconds)),
            created_at=datetime.now(UTC),
        )
        self._session.add(orm)
        await self._session.commit()
        logger.info(f"Created optimization record: {len(entity.symbols_optimized)} symbols")
        return entity

    async def get_by_id(self, entity_id: str) -> OptimizationRecord | None:
        """Get optimization record by ID.

        Args:
            entity_id: Optimization record UUID string

        Returns:
            OptimizationRecord if found, None otherwise
        """
        result = await self._session.execute(
            select(OptimizationRecordORM).where(OptimizationRecordORM.id == uuid.UUID(entity_id))
        )
        orm = result.scalar_one_or_none()
        return self._to_record(orm) if orm else None

    async def get_recent(self, limit: int = 100) -> list[OptimizationRecord]:
        """Get recent optimization records.

        Args:
            limit: Maximum number of records to return

        Returns:
            List of recent OptimizationRecords
        """
        result = await self._session.execute(
            select(OptimizationRecordORM).order_by(OptimizationRecordORM.timestamp.desc()).limit(limit)
        )
        return [self._to_record(orm) for orm in result.scalars().all()]

    def _to_record(self, orm: OptimizationRecordORM) -> OptimizationRecord:
        """Convert ORM model to OptimizationRecord.

        Args:
            orm: OptimizationRecordORM instance

        Returns:
            OptimizationRecord
        """
        return OptimizationRecord(
            timestamp=orm.timestamp,
            symbols_optimized=orm.symbols_optimized if isinstance(orm.symbols_optimized, list) else [],
            symbols_skipped=orm.symbols_skipped if isinstance(orm.symbols_skipped, list) else [],
            total_time_seconds=float(orm.total_time_seconds),
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return "OptimizationRecordRepository()"
