"""Game plan record repository for database operations."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from loguru import logger
from sqlalchemy import select

from src.daemon.state.models import GamePlanRecord
from src.database.models import GamePlanRecordORM
from src.database.repositories.base import BaseRepository

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


class GamePlanRecordRepository(BaseRepository[GamePlanRecord]):
    """Repository for game plan record persistence."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with database session.

        Args:
            session: SQLAlchemy async session
        """
        super().__init__(session)

    async def create(self, entity: GamePlanRecord) -> GamePlanRecord:
        """Create new game plan record.

        Args:
            entity: GamePlanRecord to persist

        Returns:
            Created GamePlanRecord
        """
        orm = GamePlanRecordORM(
            id=uuid.uuid4(),
            timestamp=entity.timestamp,
            priority_symbols=entity.priority_symbols,
            risk_stance=entity.risk_stance,
            sector_focus=entity.sector_focus,
            reasoning=entity.reasoning,
            confidence=entity.confidence,
            overnight_summary=entity.overnight_summary,
            key_levels=entity.key_levels,
            generated_at=entity.generated_at or datetime.now(UTC),
            created_at=datetime.now(UTC),
        )
        self._session.add(orm)
        await self._session.commit()
        logger.info(f"Created game plan record: {entity.risk_stance}")
        return entity

    async def get_by_id(self, entity_id: str) -> GamePlanRecord | None:
        """Get game plan record by ID.

        Args:
            entity_id: Game plan record UUID string

        Returns:
            GamePlanRecord if found, None otherwise
        """
        result = await self._session.execute(
            select(GamePlanRecordORM).where(GamePlanRecordORM.id == uuid.UUID(entity_id))
        )
        orm = result.scalar_one_or_none()
        return self._to_record(orm) if orm else None

    async def get_recent(self, limit: int = 100) -> list[GamePlanRecord]:
        """Get recent game plan records.

        Args:
            limit: Maximum number of records to return

        Returns:
            List of recent GamePlanRecords
        """
        result = await self._session.execute(
            select(GamePlanRecordORM).order_by(GamePlanRecordORM.timestamp.desc()).limit(limit)
        )
        return [self._to_record(orm) for orm in result.scalars().all()]

    def _to_record(self, orm: GamePlanRecordORM) -> GamePlanRecord:
        """Convert ORM model to GamePlanRecord.

        Args:
            orm: GamePlanRecordORM instance

        Returns:
            GamePlanRecord
        """
        return GamePlanRecord(
            timestamp=orm.timestamp,
            priority_symbols=orm.priority_symbols if isinstance(orm.priority_symbols, list) else [],
            risk_stance=orm.risk_stance,
            sector_focus=orm.sector_focus if isinstance(orm.sector_focus, list) else [],
            reasoning=orm.reasoning,
            confidence=float(orm.confidence) if orm.confidence is not None else None,
            overnight_summary=orm.overnight_summary,
            key_levels=orm.key_levels if isinstance(orm.key_levels, dict) else {},
            generated_at=orm.generated_at,
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return "GamePlanRecordRepository()"
