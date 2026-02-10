"""Discovery history repository for database operations."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from loguru import logger
from sqlalchemy import delete, select

from src.daemon.state import DiscoveryHistoryRecord
from src.database.models import DiscoveryHistoryRecordORM
from src.database.repositories.base import BaseRepository

if TYPE_CHECKING:
    from sqlalchemy.engine import Result
    from sqlalchemy.ext.asyncio import AsyncSession


class DiscoveryHistoryRepository(BaseRepository[DiscoveryHistoryRecord]):
    """Repository for discovery history record persistence."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with database session.

        Args:
            session: SQLAlchemy async session
        """
        super().__init__(session)
        logger.debug("Initialized DiscoveryHistoryRepository")

    async def create(self, entity: DiscoveryHistoryRecord) -> DiscoveryHistoryRecord:
        """Create new discovery history record.

        Args:
            entity: DiscoveryHistoryRecord to persist

        Returns:
            Created DiscoveryHistoryRecord
        """
        # Convert sources list to JSONB-compatible format
        sources_data = [{"source": s.value if hasattr(s, "value") else str(s)} for s in entity.sources]

        orm = DiscoveryHistoryRecordORM(
            id=uuid.uuid4(),
            symbol=entity.symbol,
            discovered_at=entity.discovered_at,
            composite_score=Decimal(str(entity.composite_score)),
            sources=sources_data,
            added_to_watchlist=entity.added_to_watchlist,
            ttl_expires_at=entity.ttl_expires_at,
            first_signal=entity.first_signal,
            first_signal_date=entity.first_signal_date,
            outcome_7d=Decimal(str(entity.outcome_7d)) if entity.outcome_7d is not None else None,
            outcome_30d=Decimal(str(entity.outcome_30d)) if entity.outcome_30d is not None else None,
            created_at=datetime.now(UTC),
        )
        self._session.add(orm)
        await self._session.commit()
        logger.info(f"Created discovery history record: {entity.symbol} score={entity.composite_score}")
        return entity

    async def get_by_id(self, entity_id: str) -> DiscoveryHistoryRecord | None:
        """Get discovery history record by ID.

        Args:
            entity_id: Discovery history record UUID string

        Returns:
            DiscoveryHistoryRecord if found, None otherwise
        """
        result = await self._session.execute(
            select(DiscoveryHistoryRecordORM).where(DiscoveryHistoryRecordORM.id == uuid.UUID(entity_id))
        )
        orm = result.scalar_one_or_none()
        return self._to_record(orm) if orm else None

    async def get_by_symbol(self, symbol: str, limit: int = 100) -> list[DiscoveryHistoryRecord]:
        """Get discovery history records for specific symbol.

        Args:
            symbol: Stock ticker symbol
            limit: Maximum number of records to return

        Returns:
            List of DiscoveryHistoryRecords for symbol
        """
        result = await self._session.execute(
            select(DiscoveryHistoryRecordORM)
            .where(DiscoveryHistoryRecordORM.symbol == symbol)
            .order_by(DiscoveryHistoryRecordORM.discovered_at.desc())
            .limit(limit)
        )
        return [self._to_record(orm) for orm in result.scalars().all()]

    async def get_recent_discoveries(self, days: int = 30) -> list[DiscoveryHistoryRecord]:
        """Get recent discoveries across all symbols.

        Args:
            days: Number of days to look back

        Returns:
            List of recent DiscoveryHistoryRecords
        """
        from datetime import timedelta

        cutoff = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        cutoff = cutoff - timedelta(days=days)

        result = await self._session.execute(
            select(DiscoveryHistoryRecordORM)
            .where(DiscoveryHistoryRecordORM.discovered_at >= cutoff)
            .order_by(DiscoveryHistoryRecordORM.discovered_at.desc())
        )
        return [self._to_record(orm) for orm in result.scalars().all()]

    async def update_outcome(
        self,
        symbol: str,
        outcome_7d: float | None = None,
        outcome_30d: float | None = None,
    ) -> DiscoveryHistoryRecord | None:
        """Update outcome metrics for most recent discovery of symbol.

        Args:
            symbol: Stock ticker symbol
            outcome_7d: 7-day outcome metric
            outcome_30d: 30-day outcome metric

        Returns:
            Updated DiscoveryHistoryRecord if found, None otherwise
        """
        result = await self._session.execute(
            select(DiscoveryHistoryRecordORM)
            .where(DiscoveryHistoryRecordORM.symbol == symbol)
            .order_by(DiscoveryHistoryRecordORM.discovered_at.desc())
            .limit(1)
        )
        orm = result.scalar_one_or_none()
        if not orm:
            return None

        if outcome_7d is not None:
            orm.outcome_7d = Decimal(str(outcome_7d))
        if outcome_30d is not None:
            orm.outcome_30d = Decimal(str(outcome_30d))

        await self._session.commit()
        logger.info(f"Updated discovery outcome for {symbol}: 7d={outcome_7d}, 30d={outcome_30d}")
        return self._to_record(orm)

    async def delete_before(self, cutoff: datetime) -> int:
        """Delete discovery history records older than cutoff date.

        Args:
            cutoff: Delete records with created_at < cutoff

        Returns:
            Number of records deleted
        """
        result: Result = await self._session.execute(
            delete(DiscoveryHistoryRecordORM).where(DiscoveryHistoryRecordORM.created_at < cutoff)
        )
        await self._session.commit()
        deleted_count = result.rowcount or 0  # type: ignore[missing-attribute]
        logger.info(f"Deleted {deleted_count} discovery history records before {cutoff}")
        return deleted_count

    def _to_record(self, orm: DiscoveryHistoryRecordORM) -> DiscoveryHistoryRecord:
        """Convert ORM model to DiscoveryHistoryRecord.

        Args:
            orm: DiscoveryHistoryRecordORM instance

        Returns:
            DiscoveryHistoryRecord
        """
        from src.discovery.models import DiscoverySource

        # Convert JSONB sources back to DiscoverySource enum
        sources_list = []
        if isinstance(orm.sources, list):
            for s in orm.sources:
                source_str = s.get("source", "") if isinstance(s, dict) else str(s)
                try:
                    sources_list.append(DiscoverySource(source_str))
                except ValueError:
                    logger.warning(f"Unknown discovery source: {source_str}")

        return DiscoveryHistoryRecord(
            symbol=orm.symbol,
            discovered_at=orm.discovered_at,
            composite_score=float(orm.composite_score),
            sources=sources_list,
            added_to_watchlist=orm.added_to_watchlist,
            ttl_expires_at=orm.ttl_expires_at,
            first_signal=orm.first_signal,
            first_signal_date=orm.first_signal_date,
            outcome_7d=float(orm.outcome_7d) if orm.outcome_7d is not None else None,
            outcome_30d=float(orm.outcome_30d) if orm.outcome_30d is not None else None,
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return "DiscoveryHistoryRepository()"
