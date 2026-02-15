"""Sector attribution record repository for database operations."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from loguru import logger
from sqlalchemy import select

from src.daemon.state.models import SectorAttributionRecord
from src.database.models import SectorAttributionRecordORM
from src.database.repositories.base import BaseRepository

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


class SectorAttributionRecordRepository(BaseRepository[SectorAttributionRecord]):
    """Repository for sector attribution record persistence."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with database session.

        Args:
            session: SQLAlchemy async session
        """
        super().__init__(session)

    async def create(self, entity: SectorAttributionRecord) -> SectorAttributionRecord:
        """Create new sector attribution record.

        Args:
            entity: SectorAttributionRecord to persist

        Returns:
            Created SectorAttributionRecord
        """
        orm = SectorAttributionRecordORM(
            id=uuid.uuid4(),
            timestamp=entity.timestamp,
            total_portfolio_value=entity.total_portfolio_value,
            benchmark_name=entity.benchmark_name,
            contributions={"contributions": entity.contributions},
            created_at=datetime.now(UTC),
        )
        self._session.add(orm)
        await self._session.commit()
        logger.info(
            f"Created sector attribution record: "
            f"{len(entity.contributions)} sectors, value=${entity.total_portfolio_value:,.2f}"
        )
        return entity

    async def get_by_id(self, entity_id: str) -> SectorAttributionRecord | None:
        """Get sector attribution record by ID.

        Args:
            entity_id: Sector attribution record UUID string

        Returns:
            SectorAttributionRecord if found, None otherwise
        """
        result = await self._session.execute(
            select(SectorAttributionRecordORM).where(SectorAttributionRecordORM.id == uuid.UUID(entity_id))
        )
        orm = result.scalar_one_or_none()
        return self._to_record(orm) if orm else None

    async def get_latest(self) -> SectorAttributionRecord | None:
        """Get most recent sector attribution record.

        Returns:
            Latest SectorAttributionRecord or None if no records exist
        """
        result = await self._session.execute(
            select(SectorAttributionRecordORM).order_by(SectorAttributionRecordORM.timestamp.desc()).limit(1)
        )
        orm = result.scalar_one_or_none()
        return self._to_record(orm) if orm else None

    async def get_history(self, limit: int = 30) -> list[SectorAttributionRecord]:
        """Get historical sector attribution records.

        Args:
            limit: Maximum number of records to return

        Returns:
            List of SectorAttributionRecords ordered by timestamp descending
        """
        result = await self._session.execute(
            select(SectorAttributionRecordORM)
            .order_by(SectorAttributionRecordORM.timestamp.desc())
            .limit(limit)
        )
        return [self._to_record(orm) for orm in result.scalars().all()]

    def _to_record(self, orm: SectorAttributionRecordORM) -> SectorAttributionRecord:
        """Convert ORM model to SectorAttributionRecord.

        Args:
            orm: SectorAttributionRecordORM instance

        Returns:
            SectorAttributionRecord
        """
        contributions = orm.contributions.get("contributions", []) if orm.contributions else []
        return SectorAttributionRecord(
            timestamp=orm.timestamp,
            total_portfolio_value=float(orm.total_portfolio_value),
            benchmark_name=orm.benchmark_name,
            contributions=contributions if isinstance(contributions, list) else [],
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return "SectorAttributionRecordRepository()"
