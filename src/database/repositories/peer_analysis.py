"""Peer analysis record repository for database operations."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from loguru import logger
from sqlalchemy import select

from src.daemon.state.models import PeerAnalysisRecord
from src.database.models import PeerAnalysisRecordORM
from src.database.repositories.base import BaseRepository

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


class PeerAnalysisRecordRepository(BaseRepository[PeerAnalysisRecord]):
    """Repository for peer analysis record persistence."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with database session.

        Args:
            session: SQLAlchemy async session
        """
        super().__init__(session)
        logger.debug("Initialized PeerAnalysisRecordRepository")

    async def create(self, entity: PeerAnalysisRecord) -> PeerAnalysisRecord:
        """Create new peer analysis record.

        Args:
            entity: PeerAnalysisRecord to persist

        Returns:
            Created PeerAnalysisRecord
        """
        orm = PeerAnalysisRecordORM(
            id=uuid.uuid4(),
            timestamp=entity.timestamp,
            symbols_analyzed=entity.symbols_analyzed,
            rankings=entity.rankings,
            swap_recommendations=entity.swap_recommendations,
            total_peers=entity.total_peers,
            total_duration_seconds=Decimal(str(entity.total_duration_seconds)),
            created_at=datetime.now(UTC),
        )
        self._session.add(orm)
        await self._session.commit()
        logger.info(f"Created peer analysis record: {entity.total_peers} peers")
        return entity

    async def get_by_id(self, entity_id: str) -> PeerAnalysisRecord | None:
        """Get peer analysis record by ID.

        Args:
            entity_id: Peer analysis record UUID string

        Returns:
            PeerAnalysisRecord if found, None otherwise
        """
        result = await self._session.execute(
            select(PeerAnalysisRecordORM).where(PeerAnalysisRecordORM.id == uuid.UUID(entity_id))
        )
        orm = result.scalar_one_or_none()
        return self._to_record(orm) if orm else None

    async def get_recent(self, limit: int = 100) -> list[PeerAnalysisRecord]:
        """Get recent peer analysis records.

        Args:
            limit: Maximum number of records to return

        Returns:
            List of recent PeerAnalysisRecords
        """
        result = await self._session.execute(
            select(PeerAnalysisRecordORM).order_by(PeerAnalysisRecordORM.timestamp.desc()).limit(limit)
        )
        return [self._to_record(orm) for orm in result.scalars().all()]

    def _to_record(self, orm: PeerAnalysisRecordORM) -> PeerAnalysisRecord:
        """Convert ORM model to PeerAnalysisRecord.

        Args:
            orm: PeerAnalysisRecordORM instance

        Returns:
            PeerAnalysisRecord
        """
        return PeerAnalysisRecord(
            timestamp=orm.timestamp,
            symbols_analyzed=orm.symbols_analyzed if isinstance(orm.symbols_analyzed, list) else [],
            rankings=orm.rankings if isinstance(orm.rankings, dict) else {},
            swap_recommendations=orm.swap_recommendations
            if isinstance(orm.swap_recommendations, list)
            else [],
            total_peers=orm.total_peers,
            total_duration_seconds=float(orm.total_duration_seconds),
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return "PeerAnalysisRecordRepository()"
