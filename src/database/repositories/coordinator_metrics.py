"""Coordinator metrics repository for database operations."""

import uuid
from datetime import datetime
from decimal import Decimal

from loguru import logger
from sqlalchemy import select

from src.coordinator.metrics import CoordinatorCycleMetrics
from src.database.models import CoordinatorMetricsORM
from src.database.repositories.base import BaseRepository


class CoordinatorMetricsRepository(BaseRepository[CoordinatorCycleMetrics]):
    """Repository for coordinator cycle metrics."""

    async def create(self, entity: CoordinatorCycleMetrics) -> CoordinatorCycleMetrics:
        """Insert coordinator metrics record.

        Args:
            entity: CoordinatorCycleMetrics to persist

        Returns:
            Created CoordinatorCycleMetrics with ID
        """
        orm = CoordinatorMetricsORM(
            id=uuid.uuid4(),
            cycle_num=entity.cycle_num,
            timestamp=entity.timestamp,
            symbols_analyzed=entity.symbols_analyzed,
            tool_calls_made=entity.tool_calls_made,
            trades_proposed=entity.trades_proposed,
            trades_executed=entity.trades_executed,
            trades_pending=entity.trades_pending,
            game_plan_generated=entity.game_plan_generated,
            cycle_duration_seconds=Decimal(str(entity.cycle_duration_seconds)),
            patterns_detected=entity.patterns_detected,
        )
        self._session.add(orm)
        await self._session.commit()
        await self._session.refresh(orm)
        entity.id = str(orm.id)
        entity.created_at = orm.created_at
        logger.debug(f"Created coordinator metrics: cycle={entity.cycle_num}")
        return entity

    async def get_by_id(self, entity_id: str) -> CoordinatorCycleMetrics | None:
        """Get coordinator metrics by ID.

        Args:
            entity_id: Coordinator metrics UUID string

        Returns:
            CoordinatorCycleMetrics if found, None otherwise
        """
        result = await self._session.execute(
            select(CoordinatorMetricsORM).where(CoordinatorMetricsORM.id == uuid.UUID(entity_id))
        )
        orm = result.scalar_one_or_none()
        return self._to_metrics(orm) if orm else None

    async def get_recent(self, limit: int = 50) -> list[CoordinatorCycleMetrics]:
        """Get recent coordinator metrics.

        Args:
            limit: Maximum records to return

        Returns:
            List of recent CoordinatorCycleMetrics (newest first)
        """
        result = await self._session.execute(
            select(CoordinatorMetricsORM).order_by(CoordinatorMetricsORM.timestamp.desc()).limit(limit)
        )
        return [self._to_metrics(orm) for orm in result.scalars().all()]

    async def get_by_cycle_num(self, cycle_num: int, limit: int = 100) -> list[CoordinatorCycleMetrics]:
        """Get metrics for specific cycle number.

        Args:
            cycle_num: Cycle number to filter by
            limit: Maximum records to return

        Returns:
            List of CoordinatorCycleMetrics for cycle
        """
        result = await self._session.execute(
            select(CoordinatorMetricsORM)
            .where(CoordinatorMetricsORM.cycle_num == cycle_num)
            .order_by(CoordinatorMetricsORM.timestamp.desc())
            .limit(limit)
        )
        return [self._to_metrics(orm) for orm in result.scalars().all()]

    async def get_date_range(
        self, start: datetime, end: datetime, limit: int = 1000
    ) -> list[CoordinatorCycleMetrics]:
        """Get metrics within date range.

        Args:
            start: Start datetime
            end: End datetime
            limit: Maximum records to return

        Returns:
            List of CoordinatorCycleMetrics in range
        """
        result = await self._session.execute(
            select(CoordinatorMetricsORM)
            .where(CoordinatorMetricsORM.timestamp >= start)
            .where(CoordinatorMetricsORM.timestamp <= end)
            .order_by(CoordinatorMetricsORM.timestamp.desc())
            .limit(limit)
        )
        return [self._to_metrics(orm) for orm in result.scalars().all()]

    def _to_metrics(self, orm: CoordinatorMetricsORM) -> CoordinatorCycleMetrics:
        """Convert ORM to domain model.

        Args:
            orm: CoordinatorMetricsORM instance

        Returns:
            CoordinatorCycleMetrics
        """
        return CoordinatorCycleMetrics(
            id=str(orm.id),
            created_at=orm.created_at,
            cycle_num=orm.cycle_num,
            timestamp=orm.timestamp,
            symbols_analyzed=list(orm.symbols_analyzed),
            tool_calls_made=orm.tool_calls_made,
            trades_proposed=orm.trades_proposed,
            trades_executed=orm.trades_executed,
            trades_pending=orm.trades_pending,
            game_plan_generated=orm.game_plan_generated,
            cycle_duration_seconds=float(orm.cycle_duration_seconds),
            patterns_detected=orm.patterns_detected,
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return "CoordinatorMetricsRepository()"
