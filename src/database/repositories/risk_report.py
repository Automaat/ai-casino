"""Risk report record repository for database operations."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from loguru import logger
from sqlalchemy import select

from src.daemon.state.models import RiskReportRecord
from src.database.models import RiskReportRecordORM
from src.database.repositories.base import BaseRepository

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


class RiskReportRecordRepository(BaseRepository[RiskReportRecord]):
    """Repository for risk report record persistence."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with database session.

        Args:
            session: SQLAlchemy async session
        """
        super().__init__(session)
        logger.debug("Initialized RiskReportRecordRepository")

    async def create(self, entity: RiskReportRecord) -> RiskReportRecord:
        """Create new risk report record.

        Args:
            entity: RiskReportRecord to persist

        Returns:
            Created RiskReportRecord
        """
        orm = RiskReportRecordORM(
            id=uuid.uuid4(),
            timestamp=entity.timestamp,
            var_95=Decimal(str(entity.var_95)),
            var_99=Decimal(str(entity.var_99)),
            cvar_95=Decimal(str(entity.cvar_95)),
            cvar_99=Decimal(str(entity.cvar_99)),
            cdar_95=Decimal(str(entity.cdar_95)),
            max_drawdown=Decimal(str(entity.max_drawdown)),
            portfolio_volatility=Decimal(str(entity.portfolio_volatility)),
            current_exposure_percent=Decimal(str(entity.current_exposure_percent)),
            num_positions=entity.num_positions,
            var_limit_breached=entity.var_limit_breached,
            cvar_limit_breached=entity.cvar_limit_breached,
            risk_status=entity.risk_status,
            created_at=datetime.now(UTC),
        )
        self._session.add(orm)
        await self._session.commit()
        logger.info(f"Created risk report record: {entity.risk_status}")
        return entity

    async def get_by_id(self, entity_id: str) -> RiskReportRecord | None:
        """Get risk report record by ID.

        Args:
            entity_id: Risk report record UUID string

        Returns:
            RiskReportRecord if found, None otherwise
        """
        result = await self._session.execute(
            select(RiskReportRecordORM).where(RiskReportRecordORM.id == uuid.UUID(entity_id))
        )
        orm = result.scalar_one_or_none()
        return self._to_record(orm) if orm else None

    async def get_recent(self, limit: int = 100) -> list[RiskReportRecord]:
        """Get recent risk report records.

        Args:
            limit: Maximum number of records to return

        Returns:
            List of recent RiskReportRecords
        """
        result = await self._session.execute(
            select(RiskReportRecordORM).order_by(RiskReportRecordORM.timestamp.desc()).limit(limit)
        )
        return [self._to_record(orm) for orm in result.scalars().all()]

    def _to_record(self, orm: RiskReportRecordORM) -> RiskReportRecord:
        """Convert ORM model to RiskReportRecord.

        Args:
            orm: RiskReportRecordORM instance

        Returns:
            RiskReportRecord
        """
        return RiskReportRecord(
            timestamp=orm.timestamp,
            var_95=float(orm.var_95),
            var_99=float(orm.var_99),
            cvar_95=float(orm.cvar_95),
            cvar_99=float(orm.cvar_99),
            cdar_95=float(orm.cdar_95),
            max_drawdown=float(orm.max_drawdown),
            portfolio_volatility=float(orm.portfolio_volatility),
            current_exposure_percent=float(orm.current_exposure_percent),
            num_positions=orm.num_positions,
            var_limit_breached=orm.var_limit_breached,
            cvar_limit_breached=orm.cvar_limit_breached,
            risk_status=orm.risk_status,
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return "RiskReportRecordRepository()"
