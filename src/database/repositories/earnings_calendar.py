"""Earnings calendar record repository for database operations."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from loguru import logger
from sqlalchemy import select

from src.daemon.state.models import EarningsCalendarRecord, EarningsEventRecord
from src.database.models import EarningsCalendarRecordORM
from src.database.repositories.base import BaseRepository

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


class EarningsCalendarRecordRepository(BaseRepository[EarningsCalendarRecord]):
    """Repository for earnings calendar record persistence."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with database session.

        Args:
            session: SQLAlchemy async session
        """
        super().__init__(session)
        logger.debug("Initialized EarningsCalendarRecordRepository")

    async def create(self, entity: EarningsCalendarRecord) -> EarningsCalendarRecord:
        """Create new earnings calendar record.

        Args:
            entity: EarningsCalendarRecord to persist

        Returns:
            Created EarningsCalendarRecord
        """
        # Serialize events to JSONB
        events_json = [
            {
                "symbol": e.symbol,
                "earnings_date": e.earnings_date,
                "estimate_eps": e.estimate_eps,
            }
            for e in entity.events
        ]

        orm = EarningsCalendarRecordORM(
            id=uuid.uuid4(),
            timestamp=entity.timestamp,
            events=events_json,
            symbols_fetched=entity.symbols_fetched,
            symbols_failed=entity.symbols_failed,
            created_at=datetime.now(UTC),
        )
        self._session.add(orm)
        await self._session.commit()
        logger.info(f"Created earnings calendar record: {len(entity.events)} events")
        return entity

    async def get_by_id(self, entity_id: str) -> EarningsCalendarRecord | None:
        """Get earnings calendar record by ID.

        Args:
            entity_id: Earnings calendar record UUID string

        Returns:
            EarningsCalendarRecord if found, None otherwise
        """
        result = await self._session.execute(
            select(EarningsCalendarRecordORM).where(EarningsCalendarRecordORM.id == uuid.UUID(entity_id))
        )
        orm = result.scalar_one_or_none()
        return self._to_record(orm) if orm else None

    async def get_recent(self, limit: int = 100) -> list[EarningsCalendarRecord]:
        """Get recent earnings calendar records.

        Args:
            limit: Maximum number of records to return

        Returns:
            List of recent EarningsCalendarRecords
        """
        result = await self._session.execute(
            select(EarningsCalendarRecordORM)
            .order_by(EarningsCalendarRecordORM.timestamp.desc())
            .limit(limit)
        )
        return [self._to_record(orm) for orm in result.scalars().all()]

    def _to_record(self, orm: EarningsCalendarRecordORM) -> EarningsCalendarRecord:
        """Convert ORM model to EarningsCalendarRecord.

        Args:
            orm: EarningsCalendarRecordORM instance

        Returns:
            EarningsCalendarRecord
        """
        events = []
        if isinstance(orm.events, list):
            for e_dict in orm.events:
                events.append(
                    EarningsEventRecord(
                        symbol=e_dict["symbol"],
                        earnings_date=e_dict["earnings_date"],
                        estimate_eps=e_dict.get("estimate_eps"),
                    )
                )

        return EarningsCalendarRecord(
            timestamp=orm.timestamp,
            events=events,
            symbols_fetched=orm.symbols_fetched,
            symbols_failed=orm.symbols_failed,
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return "EarningsCalendarRecordRepository()"
