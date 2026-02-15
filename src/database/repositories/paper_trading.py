"""Paper trading report repository for database operations."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from loguru import logger
from sqlalchemy import select

from src.daemon.state.models import PaperTradingReportRecord
from src.database.models import PaperTradingReportORM
from src.database.repositories.base import BaseRepository

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


class PaperTradingReportRepository(BaseRepository[PaperTradingReportRecord]):
    """Repository for paper trading report persistence."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with database session.

        Args:
            session: SQLAlchemy async session
        """
        super().__init__(session)

    async def create(self, entity: PaperTradingReportRecord) -> PaperTradingReportRecord:
        """Create new paper trading report record.

        Args:
            entity: PaperTradingReportRecord to persist

        Returns:
            Created PaperTradingReportRecord with ID
        """
        orm = PaperTradingReportORM(
            id=uuid.uuid4(),
            assessment_date=entity.assessment_date,
            ready_for_live=entity.ready_for_live,
            paper_trading_duration_days=entity.paper_trading_duration_days,
            total_paper_trades=entity.total_paper_trades,
            criteria=entity.criteria,
            total_pnl=Decimal(str(entity.total_pnl)),
            sharpe_ratio=Decimal(str(entity.sharpe_ratio)),
            sortino_ratio=Decimal(str(entity.sortino_ratio)),
            max_drawdown=Decimal(str(entity.max_drawdown)),
            win_rate=Decimal(str(entity.win_rate)),
            simulated_live=entity.simulated_live,
            recommendations=entity.recommendations,
            created_at=datetime.now(UTC),
        )
        self._session.add(orm)
        await self._session.commit()
        logger.info(f"Created paper trading report: {orm.id} (ready={entity.ready_for_live})")
        entity.id = str(orm.id)
        return entity

    async def get_by_id(self, entity_id: str) -> PaperTradingReportRecord | None:
        """Get paper trading report by ID.

        Args:
            entity_id: Paper trading report UUID string

        Returns:
            PaperTradingReportRecord if found, None otherwise
        """
        result = await self._session.execute(
            select(PaperTradingReportORM).where(PaperTradingReportORM.id == uuid.UUID(entity_id))
        )
        orm = result.scalar_one_or_none()
        return self._to_record(orm) if orm else None

    async def get_latest(self) -> PaperTradingReportRecord | None:
        """Get latest paper trading report.

        Returns:
            Latest PaperTradingReportRecord if exists, None otherwise
        """
        result = await self._session.execute(
            select(PaperTradingReportORM).order_by(PaperTradingReportORM.assessment_date.desc()).limit(1)
        )
        orm = result.scalar_one_or_none()
        return self._to_record(orm) if orm else None

    async def get_recent(self, limit: int = 10) -> list[PaperTradingReportRecord]:
        """Get recent paper trading reports.

        Args:
            limit: Maximum number of reports to return

        Returns:
            List of PaperTradingReportRecords ordered by assessment_date desc
        """
        result = await self._session.execute(
            select(PaperTradingReportORM).order_by(PaperTradingReportORM.assessment_date.desc()).limit(limit)
        )
        return [self._to_record(orm) for orm in result.scalars().all()]

    async def get_ready_reports(self) -> list[PaperTradingReportRecord]:
        """Get all reports that passed readiness validation.

        Returns:
            List of PaperTradingReportRecords where ready_for_live is True
        """
        result = await self._session.execute(
            select(PaperTradingReportORM)
            .where(PaperTradingReportORM.ready_for_live.is_(True))
            .order_by(PaperTradingReportORM.assessment_date.desc())
        )
        return [self._to_record(orm) for orm in result.scalars().all()]

    def _to_record(self, orm: PaperTradingReportORM) -> PaperTradingReportRecord:
        """Convert ORM model to PaperTradingReportRecord.

        Args:
            orm: PaperTradingReportORM instance

        Returns:
            PaperTradingReportRecord
        """
        return PaperTradingReportRecord(
            id=str(orm.id),
            assessment_date=orm.assessment_date,
            ready_for_live=orm.ready_for_live,
            paper_trading_duration_days=orm.paper_trading_duration_days,
            total_paper_trades=orm.total_paper_trades,
            criteria=orm.criteria,
            total_pnl=float(orm.total_pnl),
            sharpe_ratio=float(orm.sharpe_ratio),
            sortino_ratio=float(orm.sortino_ratio),
            max_drawdown=float(orm.max_drawdown),
            win_rate=float(orm.win_rate),
            simulated_live=orm.simulated_live,
            recommendations=orm.recommendations,
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return "PaperTradingReportRepository()"
