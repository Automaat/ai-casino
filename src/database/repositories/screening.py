"""Screening record repository for database operations."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from loguru import logger
from sqlalchemy import select

from src.daemon.state.models import ScreeningRecord
from src.database.models import ScreeningRecordORM
from src.database.repositories.base import BaseRepository

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


class ScreeningRecordRepository(BaseRepository[ScreeningRecord]):
    """Repository for screening record persistence."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with database session.

        Args:
            session: SQLAlchemy async session
        """
        super().__init__(session)

    async def create(self, entity: ScreeningRecord) -> ScreeningRecord:
        """Create new screening record.

        Args:
            entity: ScreeningRecord to persist

        Returns:
            Created ScreeningRecord
        """
        # Serialize candidates (ScreeningResult objects) to JSONB
        candidates_json = [
            {
                "symbol": c.symbol,
                "name": c.name,
                "sector": c.sector,
                "score": c.score,
                "signal": c.signal,
                "metrics": c.metrics,
                "reason": c.reason,
            }
            for c in entity.candidates
        ]

        orm = ScreeningRecordORM(
            id=uuid.uuid4(),
            timestamp=entity.timestamp,
            criteria=entity.criteria,
            universe=entity.universe,
            top_symbols=entity.top_symbols,
            candidates=candidates_json,
            screened_at=entity.screened_at,
            created_at=datetime.now(UTC),
        )
        self._session.add(orm)
        await self._session.commit()
        logger.info(f"Created screening record: {len(entity.candidates)} candidates")
        return entity

    async def get_by_id(self, entity_id: str) -> ScreeningRecord | None:
        """Get screening record by ID.

        Args:
            entity_id: Screening record UUID string

        Returns:
            ScreeningRecord if found, None otherwise
        """
        result = await self._session.execute(
            select(ScreeningRecordORM).where(ScreeningRecordORM.id == uuid.UUID(entity_id))
        )
        orm = result.scalar_one_or_none()
        return self._to_record(orm) if orm else None

    async def get_recent(self, limit: int = 100) -> list[ScreeningRecord]:
        """Get recent screening records.

        Args:
            limit: Maximum number of records to return

        Returns:
            List of recent ScreeningRecords
        """
        result = await self._session.execute(
            select(ScreeningRecordORM).order_by(ScreeningRecordORM.timestamp.desc()).limit(limit)
        )
        return [self._to_record(orm) for orm in result.scalars().all()]

    def _to_record(self, orm: ScreeningRecordORM) -> ScreeningRecord:
        """Convert ORM model to ScreeningRecord.

        Args:
            orm: ScreeningRecordORM instance

        Returns:
            ScreeningRecord
        """
        from src.screening.screener import ScreeningResult

        candidates = []
        if isinstance(orm.candidates, list):
            for c_dict in orm.candidates:
                candidates.append(
                    ScreeningResult(
                        symbol=c_dict.get("symbol", "UNKNOWN"),
                        name=c_dict.get("name", "Unknown"),
                        sector=c_dict.get("sector", "Unknown"),
                        score=c_dict.get("score", 0.0),
                        signal=c_dict.get("signal", "HOLD"),
                        metrics=c_dict.get("metrics", {}),
                        reason=c_dict.get("reason", "Legacy data"),
                    )
                )

        return ScreeningRecord(
            id=str(orm.id),
            timestamp=orm.timestamp,
            criteria=orm.criteria,
            universe=orm.universe,
            top_symbols=orm.top_symbols if isinstance(orm.top_symbols, list) else [],
            candidates=candidates,
            screened_at=orm.screened_at,
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return "ScreeningRecordRepository()"
