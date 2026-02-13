"""Prefetch record repository for database operations."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from loguru import logger
from sqlalchemy import select

from src.daemon.state.models import PrefetchRecord
from src.database.models import PrefetchRecordORM
from src.database.repositories.base import BaseRepository

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


class PrefetchRecordRepository(BaseRepository[PrefetchRecord]):
    """Repository for prefetch record persistence."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with database session.

        Args:
            session: SQLAlchemy async session
        """
        super().__init__(session)
        logger.debug("Initialized PrefetchRecordRepository")

    async def create(self, entity: PrefetchRecord) -> PrefetchRecord:
        """Create new prefetch record.

        Args:
            entity: PrefetchRecord to persist

        Returns:
            Created PrefetchRecord
        """
        orm = PrefetchRecordORM(
            id=uuid.uuid4(),
            timestamp=entity.timestamp,
            symbols_prefetched=entity.symbols_prefetched,
            symbols_failed=entity.symbols_failed,
            finbert_ready=entity.finbert_ready,
            total_duration_seconds=Decimal(str(entity.total_duration_seconds)),
            created_at=datetime.now(UTC),
        )
        self._session.add(orm)
        await self._session.commit()
        logger.info(f"Created prefetch record: {entity.symbols_prefetched} symbols")
        return entity

    async def get_by_id(self, entity_id: str) -> PrefetchRecord | None:
        """Get prefetch record by ID.

        Args:
            entity_id: Prefetch record UUID string

        Returns:
            PrefetchRecord if found, None otherwise
        """
        result = await self._session.execute(
            select(PrefetchRecordORM).where(PrefetchRecordORM.id == uuid.UUID(entity_id))
        )
        orm = result.scalar_one_or_none()
        return self._to_record(orm) if orm else None

    async def get_recent(self, limit: int = 100) -> list[PrefetchRecord]:
        """Get recent prefetch records.

        Args:
            limit: Maximum number of records to return

        Returns:
            List of recent PrefetchRecords
        """
        result = await self._session.execute(
            select(PrefetchRecordORM).order_by(PrefetchRecordORM.timestamp.desc()).limit(limit)
        )
        return [self._to_record(orm) for orm in result.scalars().all()]

    def _to_record(self, orm: PrefetchRecordORM) -> PrefetchRecord:
        """Convert ORM model to PrefetchRecord.

        Args:
            orm: PrefetchRecordORM instance

        Returns:
            PrefetchRecord
        """
        return PrefetchRecord(
            timestamp=orm.timestamp,
            symbols_prefetched=orm.symbols_prefetched,
            symbols_failed=orm.symbols_failed,
            finbert_ready=orm.finbert_ready,
            total_duration_seconds=float(orm.total_duration_seconds),
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return "PrefetchRecordRepository()"
