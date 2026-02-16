"""Trade journal repository for database operations."""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from loguru import logger
from sqlalchemy import select

from src.daemon.state.models import TradeJournalRecord
from src.database.models import TradeJournalORM
from src.database.repositories.base import BaseRepository

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


class TradeJournalRepository(BaseRepository[TradeJournalRecord]):
    """Repository for trade journal persistence."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with database session.

        Args:
            session: SQLAlchemy async session
        """
        super().__init__(session)

    async def create(self, entity: TradeJournalRecord) -> TradeJournalRecord:
        """Create new trade journal record.

        Args:
            entity: TradeJournalRecord to persist

        Returns:
            Created TradeJournalRecord with ID
        """
        orm = TradeJournalORM(
            id=uuid.uuid4(),
            date=entity.date,
            outcomes=entity.outcomes,
            winners=entity.winners,
            losers=entity.losers,
            lessons=entity.lessons,
            tomorrows_focus=entity.tomorrows_focus,
            overall_assessment=entity.overall_assessment,
            markdown_content=entity.markdown_content,
            total_signals=entity.total_signals,
            correct_signals=entity.correct_signals,
            accuracy_pct=Decimal(str(entity.accuracy_pct)),
            created_at=datetime.now(UTC),
        )
        self._session.add(orm)
        await self._session.commit()
        logger.info(f"Created trade journal: {orm.id} (date={entity.date}, accuracy={entity.accuracy_pct}%)")
        entity.id = str(orm.id)
        return entity

    async def get_by_id(self, entity_id: str) -> TradeJournalRecord | None:
        """Get trade journal by ID.

        Args:
            entity_id: Trade journal UUID string

        Returns:
            TradeJournalRecord if found, None otherwise
        """
        result = await self._session.execute(
            select(TradeJournalORM).where(TradeJournalORM.id == uuid.UUID(entity_id))
        )
        orm = result.scalar_one_or_none()
        return self._to_record(orm) if orm else None

    async def get_by_date(self, journal_date: date) -> TradeJournalRecord | None:
        """Get trade journal by date.

        Args:
            journal_date: Journal date

        Returns:
            TradeJournalRecord if found, None otherwise
        """
        result = await self._session.execute(
            select(TradeJournalORM).where(TradeJournalORM.date == journal_date)
        )
        orm = result.scalar_one_or_none()
        return self._to_record(orm) if orm else None

    async def get_recent(self, limit: int = 30) -> list[TradeJournalRecord]:
        """Get recent trade journals.

        Args:
            limit: Maximum number of journals to return

        Returns:
            List of TradeJournalRecords ordered by date desc
        """
        result = await self._session.execute(
            select(TradeJournalORM).order_by(TradeJournalORM.date.desc()).limit(limit)
        )
        return [self._to_record(orm) for orm in result.scalars().all()]

    async def get_date_range(self, start_date: date, end_date: date) -> list[TradeJournalRecord]:
        """Get trade journals within date range.

        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)

        Returns:
            List of TradeJournalRecords in date range
        """
        result = await self._session.execute(
            select(TradeJournalORM)
            .where(TradeJournalORM.date >= start_date, TradeJournalORM.date <= end_date)
            .order_by(TradeJournalORM.date.desc())
        )
        return [self._to_record(orm) for orm in result.scalars().all()]

    def _to_record(self, orm: TradeJournalORM) -> TradeJournalRecord:
        """Convert ORM model to TradeJournalRecord.

        Args:
            orm: TradeJournalORM instance

        Returns:
            TradeJournalRecord
        """
        return TradeJournalRecord(
            id=str(orm.id),
            date=orm.date,
            outcomes=orm.outcomes,
            winners=orm.winners,
            losers=orm.losers,
            lessons=orm.lessons,
            tomorrows_focus=orm.tomorrows_focus,
            overall_assessment=orm.overall_assessment,
            markdown_content=orm.markdown_content,
            total_signals=orm.total_signals,
            correct_signals=orm.correct_signals,
            accuracy_pct=float(orm.accuracy_pct),
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return "TradeJournalRepository()"
