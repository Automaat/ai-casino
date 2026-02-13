"""Monte Carlo record repository for database operations."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from loguru import logger
from sqlalchemy import select

from src.daemon.state.models import MonteCarloRecord
from src.database.models import MonteCarloRecordORM
from src.database.repositories.base import BaseRepository

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


class MonteCarloRecordRepository(BaseRepository[MonteCarloRecord]):
    """Repository for Monte Carlo record persistence."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with database session.

        Args:
            session: SQLAlchemy async session
        """
        super().__init__(session)
        logger.debug("Initialized MonteCarloRecordRepository")

    async def create(self, entity: MonteCarloRecord) -> MonteCarloRecord:
        """Create new Monte Carlo record.

        Args:
            entity: MonteCarloRecord to persist

        Returns:
            Created MonteCarloRecord
        """
        orm = MonteCarloRecordORM(
            id=uuid.uuid4(),
            timestamp=entity.timestamp,
            simulation_method=entity.simulation_method,
            num_simulations=entity.num_simulations,
            horizon_days=entity.horizon_days,
            prob_loss_gt_threshold=Decimal(str(entity.prob_loss_gt_threshold)),
            expected_worst_drawdown=Decimal(str(entity.expected_worst_drawdown)),
            var_95=Decimal(str(entity.var_95)),
            cvar_95=Decimal(str(entity.cvar_95)),
            median_recovery_days=(
                Decimal(str(entity.median_recovery_days)) if entity.median_recovery_days is not None else None
            ),
            exceeds_risk_tolerance=entity.exceeds_risk_tolerance,
            alert_message=entity.alert_message,
            portfolio_symbols=entity.portfolio_symbols,
            total_market_value=Decimal(str(entity.total_market_value)),
            created_at=datetime.now(UTC),
        )
        self._session.add(orm)
        await self._session.commit()
        logger.info(f"Created Monte Carlo record: {entity.simulation_method}")
        return entity

    async def get_by_id(self, entity_id: str) -> MonteCarloRecord | None:
        """Get Monte Carlo record by ID.

        Args:
            entity_id: Monte Carlo record UUID string

        Returns:
            MonteCarloRecord if found, None otherwise
        """
        result = await self._session.execute(
            select(MonteCarloRecordORM).where(MonteCarloRecordORM.id == uuid.UUID(entity_id))
        )
        orm = result.scalar_one_or_none()
        return self._to_record(orm) if orm else None

    async def get_recent(self, limit: int = 100) -> list[MonteCarloRecord]:
        """Get recent Monte Carlo records.

        Args:
            limit: Maximum number of records to return

        Returns:
            List of recent MonteCarloRecords
        """
        result = await self._session.execute(
            select(MonteCarloRecordORM).order_by(MonteCarloRecordORM.timestamp.desc()).limit(limit)
        )
        return [self._to_record(orm) for orm in result.scalars().all()]

    def _to_record(self, orm: MonteCarloRecordORM) -> MonteCarloRecord:
        """Convert ORM model to MonteCarloRecord.

        Args:
            orm: MonteCarloRecordORM instance

        Returns:
            MonteCarloRecord
        """
        return MonteCarloRecord(
            timestamp=orm.timestamp,
            simulation_method=orm.simulation_method,
            num_simulations=orm.num_simulations,
            horizon_days=orm.horizon_days,
            prob_loss_gt_threshold=float(orm.prob_loss_gt_threshold),
            expected_worst_drawdown=float(orm.expected_worst_drawdown),
            var_95=float(orm.var_95),
            cvar_95=float(orm.cvar_95),
            median_recovery_days=float(orm.median_recovery_days) if orm.median_recovery_days is not None else None,
            exceeds_risk_tolerance=orm.exceeds_risk_tolerance,
            alert_message=orm.alert_message,
            portfolio_symbols=orm.portfolio_symbols if isinstance(orm.portfolio_symbols, list) else [],
            total_market_value=float(orm.total_market_value),
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return "MonteCarloRecordRepository()"
