"""Correlation audit record repository for database operations."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from loguru import logger
from sqlalchemy import select

from src.daemon.state.models import CorrelationAuditRecord
from src.database.models import CorrelationAuditRecordORM
from src.database.repositories.base import BaseRepository

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


class CorrelationAuditRecordRepository(BaseRepository[CorrelationAuditRecord]):
    """Repository for correlation audit record persistence."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with database session.

        Args:
            session: SQLAlchemy async session
        """
        super().__init__(session)
        logger.debug("Initialized CorrelationAuditRecordRepository")

    async def create(self, entity: CorrelationAuditRecord) -> CorrelationAuditRecord:
        """Create new correlation audit record.

        Args:
            entity: CorrelationAuditRecord to persist

        Returns:
            Created CorrelationAuditRecord
        """
        orm = CorrelationAuditRecordORM(
            id=uuid.uuid4(),
            timestamp=entity.timestamp,
            num_positions=entity.num_positions,
            num_correlated_pairs=entity.num_correlated_pairs,
            max_correlation=Decimal(str(entity.max_correlation)),
            avg_correlation=Decimal(str(entity.avg_correlation)),
            diversification_ratio=Decimal(str(entity.diversification_ratio)),
            num_substitutions=entity.num_substitutions,
            total_duration_seconds=Decimal(str(entity.total_duration_seconds)),
            created_at=datetime.now(UTC),
        )
        self._session.add(orm)
        await self._session.commit()
        logger.info(f"Created correlation audit record: {entity.num_positions} positions")
        return entity

    async def get_by_id(self, entity_id: str) -> CorrelationAuditRecord | None:
        """Get correlation audit record by ID.

        Args:
            entity_id: Correlation audit record UUID string

        Returns:
            CorrelationAuditRecord if found, None otherwise
        """
        result = await self._session.execute(
            select(CorrelationAuditRecordORM).where(CorrelationAuditRecordORM.id == uuid.UUID(entity_id))
        )
        orm = result.scalar_one_or_none()
        return self._to_record(orm) if orm else None

    async def get_recent(self, limit: int = 100) -> list[CorrelationAuditRecord]:
        """Get recent correlation audit records.

        Args:
            limit: Maximum number of records to return

        Returns:
            List of recent CorrelationAuditRecords
        """
        result = await self._session.execute(
            select(CorrelationAuditRecordORM).order_by(CorrelationAuditRecordORM.timestamp.desc()).limit(limit)
        )
        return [self._to_record(orm) for orm in result.scalars().all()]

    def _to_record(self, orm: CorrelationAuditRecordORM) -> CorrelationAuditRecord:
        """Convert ORM model to CorrelationAuditRecord.

        Args:
            orm: CorrelationAuditRecordORM instance

        Returns:
            CorrelationAuditRecord
        """
        return CorrelationAuditRecord(
            timestamp=orm.timestamp,
            num_positions=orm.num_positions,
            num_correlated_pairs=orm.num_correlated_pairs,
            max_correlation=float(orm.max_correlation),
            avg_correlation=float(orm.avg_correlation),
            diversification_ratio=float(orm.diversification_ratio),
            num_substitutions=orm.num_substitutions,
            total_duration_seconds=float(orm.total_duration_seconds),
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return "CorrelationAuditRecordRepository()"
