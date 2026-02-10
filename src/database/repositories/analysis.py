"""Analysis record repository for database operations."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from loguru import logger
from sqlalchemy import delete, func, select

from src.daemon.state import AnalysisRecord
from src.database.models import AnalysisRecordORM
from src.database.repositories.base import BaseRepository

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


class AnalysisRecordRepository(BaseRepository[AnalysisRecord]):
    """Repository for analysis record persistence."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with database session.

        Args:
            session: SQLAlchemy async session
        """
        super().__init__(session)
        logger.debug("Initialized AnalysisRecordRepository")

    async def create(self, entity: AnalysisRecord) -> AnalysisRecord:
        """Create new analysis record.

        Args:
            entity: AnalysisRecord to persist

        Returns:
            Created AnalysisRecord
        """
        orm = AnalysisRecordORM(
            id=uuid.uuid4(),
            symbol=entity.symbol,
            timestamp=entity.timestamp,
            signal=entity.signal,
            confidence=Decimal(str(entity.confidence)),
            executed_trade=entity.executed_trade,
            trading_session=entity.trading_session.value,
            is_paper_trade=entity.is_paper_trade,
            rsi=Decimal(str(entity.rsi)) if entity.rsi is not None else None,
            macd_hist=Decimal(str(entity.macd_hist)) if entity.macd_hist is not None else None,
            reasoning=entity.reasoning,  # list stored as JSONB
            created_at=datetime.now(UTC),
        )
        self._session.add(orm)
        await self._session.commit()
        logger.info(f"Created analysis record: {entity.symbol} {entity.signal} @ {entity.timestamp}")
        return entity

    async def get_by_id(self, entity_id: str) -> AnalysisRecord | None:
        """Get analysis record by ID.

        Args:
            entity_id: Analysis record UUID string

        Returns:
            AnalysisRecord if found, None otherwise
        """
        result = await self._session.execute(
            select(AnalysisRecordORM).where(AnalysisRecordORM.id == uuid.UUID(entity_id))
        )
        orm = result.scalar_one_or_none()
        return self._to_record(orm) if orm else None

    async def get_by_symbol(self, symbol: str, limit: int = 100) -> list[AnalysisRecord]:
        """Get analysis records for specific symbol.

        Args:
            symbol: Stock ticker symbol
            limit: Maximum number of records to return

        Returns:
            List of AnalysisRecords for symbol
        """
        result = await self._session.execute(
            select(AnalysisRecordORM)
            .where(AnalysisRecordORM.symbol == symbol)
            .order_by(AnalysisRecordORM.timestamp.desc())
            .limit(limit)
        )
        return [self._to_record(orm) for orm in result.scalars().all()]

    async def get_by_date_range(
        self,
        start: datetime,
        end: datetime,
        symbol: str | None = None,
    ) -> list[AnalysisRecord]:
        """Get analysis records within date range.

        Args:
            start: Start datetime (inclusive)
            end: End datetime (inclusive)
            symbol: Optional symbol filter

        Returns:
            List of AnalysisRecords in date range
        """
        stmt = select(AnalysisRecordORM).where(
            AnalysisRecordORM.timestamp >= start,
            AnalysisRecordORM.timestamp <= end,
        )
        if symbol:
            stmt = stmt.where(AnalysisRecordORM.symbol == symbol)
        stmt = stmt.order_by(AnalysisRecordORM.timestamp.desc())

        result = await self._session.execute(stmt)
        return [self._to_record(orm) for orm in result.scalars().all()]

    async def get_signal_stats(self, symbol: str | None = None, days: int = 30) -> dict[str, int]:
        """Get signal distribution statistics.

        Args:
            symbol: Optional symbol filter
            days: Number of days to look back

        Returns:
            Dict mapping signal to count
        """
        cutoff = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        cutoff = cutoff.replace(day=cutoff.day - days)

        stmt = (
            select(AnalysisRecordORM.signal, func.count(AnalysisRecordORM.id))
            .where(AnalysisRecordORM.created_at >= cutoff)
            .group_by(AnalysisRecordORM.signal)
        )
        if symbol:
            stmt = stmt.where(AnalysisRecordORM.symbol == symbol)

        result = await self._session.execute(stmt)
        return dict(result.all())

    async def delete_before(self, cutoff: datetime) -> int:
        """Delete analysis records older than cutoff date.

        Args:
            cutoff: Delete records with created_at < cutoff

        Returns:
            Number of records deleted
        """
        result = await self._session.execute(
            delete(AnalysisRecordORM).where(AnalysisRecordORM.created_at < cutoff)
        )
        await self._session.commit()
        deleted_count = result.rowcount if result.rowcount else 0
        logger.info(f"Deleted {deleted_count} analysis records before {cutoff}")
        return deleted_count

    def _to_record(self, orm: AnalysisRecordORM) -> AnalysisRecord:
        """Convert ORM model to AnalysisRecord.

        Args:
            orm: AnalysisRecordORM instance

        Returns:
            AnalysisRecord
        """
        from src.strategies.session import TradingSession

        return AnalysisRecord(
            symbol=orm.symbol,
            timestamp=orm.timestamp,
            signal=orm.signal,
            confidence=float(orm.confidence),
            executed_trade=orm.executed_trade,
            trading_session=TradingSession(orm.trading_session),
            is_paper_trade=orm.is_paper_trade,
            rsi=float(orm.rsi) if orm.rsi is not None else None,
            macd_hist=float(orm.macd_hist) if orm.macd_hist is not None else None,
            reasoning=orm.reasoning if isinstance(orm.reasoning, list) else [],
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return "AnalysisRecordRepository()"
