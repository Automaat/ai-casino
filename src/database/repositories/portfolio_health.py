"""Portfolio health record repository."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from loguru import logger
from sqlalchemy import select

from src.daemon.state.models import PortfolioHealthRecord
from src.database.models.metrics import PortfolioHealthRecordORM
from src.database.repositories.base import BaseRepository

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


class PortfolioHealthRecordRepository(BaseRepository[PortfolioHealthRecord]):
    """Repository for portfolio health record persistence."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with database session.

        Args:
            session: SQLAlchemy async session
        """
        super().__init__(session)

    async def create(self, entity: PortfolioHealthRecord) -> PortfolioHealthRecord:
        """Create new portfolio health record.

        Args:
            entity: PortfolioHealthRecord to persist

        Returns:
            Created PortfolioHealthRecord
        """
        orm = PortfolioHealthRecordORM(
            id=uuid.uuid4(),
            timestamp=entity.timestamp,
            total_positions=entity.total_positions,
            portfolio_value=Decimal(str(entity.portfolio_value)),
            cash_percent=Decimal(str(entity.cash_percent)),
            max_concentration_percent=Decimal(str(entity.max_concentration_percent)),
            max_concentration_symbol=entity.max_concentration_symbol,
            total_pnl_percent=Decimal(str(entity.total_pnl_percent)),
            biggest_drawdown_symbol=entity.biggest_drawdown_symbol,
            biggest_drawdown_percent=Decimal(str(entity.biggest_drawdown_percent)),
            health_status=entity.health_status,
            recommendations=entity.recommendations,
            constraints=entity.constraints,
            created_at=datetime.now(UTC),
        )
        self._session.add(orm)
        await self._session.commit()
        logger.info(f"Created portfolio health record: {entity.health_status}")
        return entity

    async def get_by_id(self, entity_id: str) -> PortfolioHealthRecord | None:
        """Get portfolio health record by ID.

        Args:
            entity_id: Record UUID string

        Returns:
            PortfolioHealthRecord if found, None otherwise
        """
        result = await self._session.execute(
            select(PortfolioHealthRecordORM).where(PortfolioHealthRecordORM.id == uuid.UUID(entity_id))
        )
        orm = result.scalar_one_or_none()
        return self._to_record(orm) if orm else None

    async def get_latest(self) -> PortfolioHealthRecord | None:
        """Get most recent portfolio health record.

        Returns:
            Latest PortfolioHealthRecord or None
        """
        result = await self._session.execute(
            select(PortfolioHealthRecordORM).order_by(PortfolioHealthRecordORM.timestamp.desc()).limit(1)
        )
        orm = result.scalar_one_or_none()
        return self._to_record(orm) if orm else None

    async def get_recent(self, limit: int = 10) -> list[PortfolioHealthRecord]:
        """Get recent portfolio health records.

        Args:
            limit: Maximum number of records

        Returns:
            List of recent PortfolioHealthRecords
        """
        result = await self._session.execute(
            select(PortfolioHealthRecordORM).order_by(PortfolioHealthRecordORM.timestamp.desc()).limit(limit)
        )
        return [self._to_record(orm) for orm in result.scalars().all()]

    def _to_record(self, orm: PortfolioHealthRecordORM) -> PortfolioHealthRecord:
        """Convert ORM to domain model.

        Args:
            orm: ORM instance

        Returns:
            PortfolioHealthRecord
        """
        return PortfolioHealthRecord(
            timestamp=orm.timestamp,
            total_positions=orm.total_positions,
            portfolio_value=float(orm.portfolio_value),
            cash_percent=float(orm.cash_percent),
            max_concentration_percent=float(orm.max_concentration_percent),
            max_concentration_symbol=orm.max_concentration_symbol,
            total_pnl_percent=float(orm.total_pnl_percent),
            biggest_drawdown_symbol=orm.biggest_drawdown_symbol,
            biggest_drawdown_percent=float(orm.biggest_drawdown_percent),
            health_status=orm.health_status,
            recommendations=orm.recommendations,
            constraints=orm.constraints,
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return "PortfolioHealthRecordRepository()"
