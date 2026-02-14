"""Supervisor metrics repository for database operations."""

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from loguru import logger
from sqlalchemy import Integer, delete, func, select

from src.agents.supervisor.metrics import SupervisorCycleMetrics
from src.database.models import SupervisorMetricsORM
from src.database.repositories.base import BaseRepository


class SupervisorMetricsRepository(BaseRepository[SupervisorCycleMetrics]):
    """Repository for supervisor cycle metrics."""

    async def create(self, entity: SupervisorCycleMetrics) -> SupervisorCycleMetrics:
        """Insert supervisor metrics record.

        Args:
            entity: SupervisorCycleMetrics to persist

        Returns:
            Created SupervisorCycleMetrics with ID
        """
        orm = SupervisorMetricsORM(
            id=uuid.uuid4(),
            created_at=datetime.now(UTC),
            workflow_id=entity.workflow_id,
            symbol=entity.symbol,
            timestamp=entity.timestamp,
            required_analyses=entity.required_analyses,
            optional_analyses=entity.optional_analyses,
            skip_analyses=entity.skip_analyses,
            routing_reasoning=entity.routing_reasoning,
            total_workers=entity.total_workers,
            required_workers=entity.required_workers,
            optional_workers=entity.optional_workers,
            successful_workers=entity.successful_workers,
            failed_workers=entity.failed_workers,
            routing_decision_ms=Decimal(str(entity.routing_decision_ms)),
            group1_execution_ms=Decimal(str(entity.group1_execution_ms)),
            research_execution_ms=Decimal(str(entity.research_execution_ms)),
            total_supervisor_overhead_ms=Decimal(str(entity.total_supervisor_overhead_ms)),
            worker_timings=entity.worker_timings,
            worker_errors=entity.worker_errors,
            total_llm_calls=entity.total_llm_calls,
            total_cost_usd=Decimal(str(entity.total_cost_usd)),
            planning_fallback_used=entity.planning_fallback_used,
            synthesis_fallback_used=entity.synthesis_fallback_used,
            confidence_adjustment=Decimal(str(entity.confidence_adjustment)),
            synthesis_reasoning=entity.synthesis_reasoning,
            parallel_efficiency_percent=Decimal(str(entity.parallel_efficiency_percent)),
            timeout_triggered=entity.timeout_triggered,
        )
        self._session.add(orm)
        await self._session.commit()
        await self._session.refresh(orm)
        entity.id = str(orm.id)
        entity.created_at = orm.created_at
        logger.debug(f"Created supervisor metrics: workflow_id={entity.workflow_id}")
        return entity

    async def get_by_id(self, entity_id: str) -> SupervisorCycleMetrics | None:
        """Get supervisor metrics by ID.

        Args:
            entity_id: Supervisor metrics UUID string

        Returns:
            SupervisorCycleMetrics if found, None otherwise
        """
        result = await self._session.execute(
            select(SupervisorMetricsORM).where(SupervisorMetricsORM.id == uuid.UUID(entity_id))
        )
        orm = result.scalar_one_or_none()
        return self._to_metrics(orm) if orm else None

    async def get_recent(self, limit: int = 50, symbol: str | None = None) -> list[SupervisorCycleMetrics]:
        """Get recent supervisor metrics.

        Args:
            limit: Maximum records to return
            symbol: Optional symbol filter

        Returns:
            List of recent SupervisorCycleMetrics (newest first)
        """
        query = select(SupervisorMetricsORM).order_by(SupervisorMetricsORM.timestamp.desc()).limit(limit)
        if symbol:
            query = query.where(SupervisorMetricsORM.symbol == symbol)
        result = await self._session.execute(query)
        return [self._to_metrics(orm) for orm in result.scalars().all()]

    async def get_by_symbol(self, symbol: str, limit: int = 100) -> list[SupervisorCycleMetrics]:
        """Get metrics for specific symbol.

        Args:
            symbol: Stock ticker symbol
            limit: Maximum records to return

        Returns:
            List of SupervisorCycleMetrics for symbol
        """
        result = await self._session.execute(
            select(SupervisorMetricsORM)
            .where(SupervisorMetricsORM.symbol == symbol)
            .order_by(SupervisorMetricsORM.timestamp.desc())
            .limit(limit)
        )
        return [self._to_metrics(orm) for orm in result.scalars().all()]

    async def get_date_range(
        self, start: datetime, end: datetime, limit: int = 1000
    ) -> list[SupervisorCycleMetrics]:
        """Get metrics within date range.

        Args:
            start: Start datetime
            end: End datetime
            limit: Maximum records to return

        Returns:
            List of SupervisorCycleMetrics in range
        """
        result = await self._session.execute(
            select(SupervisorMetricsORM)
            .where(SupervisorMetricsORM.timestamp >= start)
            .where(SupervisorMetricsORM.timestamp <= end)
            .order_by(SupervisorMetricsORM.timestamp.desc())
            .limit(limit)
        )
        return [self._to_metrics(orm) for orm in result.scalars().all()]

    async def get_error_summary(self, hours: int = 24) -> dict[str, int]:
        """Get error counts by worker type for recent period.

        Args:
            hours: Hours to look back

        Returns:
            Dict mapping worker name to error count
        """
        cutoff = datetime.now(UTC) - timedelta(hours=hours)
        result = await self._session.execute(
            select(SupervisorMetricsORM.worker_errors)
            .where(SupervisorMetricsORM.timestamp >= cutoff)
            .where(func.jsonb_typeof(SupervisorMetricsORM.worker_errors) == "object")
        )

        error_counts: dict[str, int] = {}
        for (worker_errors,) in result:
            if isinstance(worker_errors, dict):
                for worker_name in worker_errors:
                    error_counts[worker_name] = error_counts.get(worker_name, 0) + 1

        return error_counts

    async def get_efficiency_stats(self, symbol: str | None = None, days: int = 7) -> dict:
        """Get efficiency statistics for symbol or all symbols.

        Args:
            symbol: Optional symbol filter
            days: Days to look back

        Returns:
            Dict with avg efficiency, durations, timeout rate
        """
        cutoff = datetime.now(UTC) - timedelta(days=days)
        query = select(
            func.avg(SupervisorMetricsORM.parallel_efficiency_percent).label("avg_efficiency"),
            func.avg(SupervisorMetricsORM.routing_decision_ms).label("avg_routing_ms"),
            func.avg(SupervisorMetricsORM.group1_execution_ms).label("avg_group1_ms"),
            func.avg(SupervisorMetricsORM.research_execution_ms).label("avg_research_ms"),
            func.avg(SupervisorMetricsORM.total_supervisor_overhead_ms).label("avg_total_ms"),
            func.sum(func.cast(SupervisorMetricsORM.timeout_triggered, Integer)).label("timeout_count"),
            func.count().label("total_count"),
        ).where(SupervisorMetricsORM.timestamp >= cutoff)

        if symbol:
            query = query.where(SupervisorMetricsORM.symbol == symbol)

        result = await self._session.execute(query)
        row = result.one_or_none()

        if not row:
            return {
                "avg_efficiency_percent": 0.0,
                "avg_routing_ms": 0.0,
                "avg_group1_ms": 0.0,
                "avg_research_ms": 0.0,
                "avg_total_ms": 0.0,
                "timeout_rate_percent": 0.0,
                "sample_size": 0,
            }

        total_count = row.total_count or 0
        timeout_rate = (float(row.timeout_count or 0) / total_count * 100) if total_count > 0 else 0.0

        return {
            "avg_efficiency_percent": float(row.avg_efficiency or 0),
            "avg_routing_ms": float(row.avg_routing_ms or 0),
            "avg_group1_ms": float(row.avg_group1_ms or 0),
            "avg_research_ms": float(row.avg_research_ms or 0),
            "avg_total_ms": float(row.avg_total_ms or 0),
            "timeout_rate_percent": timeout_rate,
            "sample_size": total_count,
        }

    async def delete_older_than(self, days: int = 7) -> int:
        """Delete metrics older than N days.

        Args:
            days: Age threshold in days

        Returns:
            Number of deleted records
        """
        cutoff = datetime.now(UTC) - timedelta(days=days)
        result = await self._session.execute(
            select(func.count())
            .select_from(SupervisorMetricsORM)
            .where(SupervisorMetricsORM.timestamp < cutoff)
        )
        count = result.scalar_one()

        await self._session.execute(
            delete(SupervisorMetricsORM).where(SupervisorMetricsORM.timestamp < cutoff)
        )
        await self._session.commit()
        logger.info(f"Deleted {count} supervisor metrics older than {days} days")
        return count

    def _to_metrics(self, orm: SupervisorMetricsORM) -> SupervisorCycleMetrics:
        """Convert ORM to domain model.

        Args:
            orm: SupervisorMetricsORM instance

        Returns:
            SupervisorCycleMetrics
        """
        return SupervisorCycleMetrics(
            id=str(orm.id),
            created_at=orm.created_at,
            workflow_id=orm.workflow_id,
            symbol=orm.symbol,
            timestamp=orm.timestamp,
            required_analyses=list(orm.required_analyses),
            optional_analyses=list(orm.optional_analyses),
            skip_analyses=dict(orm.skip_analyses),
            routing_reasoning=orm.routing_reasoning,
            total_workers=orm.total_workers,
            required_workers=orm.required_workers,
            optional_workers=orm.optional_workers,
            successful_workers=orm.successful_workers,
            failed_workers=orm.failed_workers,
            routing_decision_ms=float(orm.routing_decision_ms),
            group1_execution_ms=float(orm.group1_execution_ms),
            research_execution_ms=float(orm.research_execution_ms),
            total_supervisor_overhead_ms=float(orm.total_supervisor_overhead_ms),
            worker_timings=dict(orm.worker_timings),
            worker_errors=dict(orm.worker_errors),
            total_llm_calls=orm.total_llm_calls,
            total_cost_usd=float(orm.total_cost_usd),
            planning_fallback_used=orm.planning_fallback_used,
            synthesis_fallback_used=orm.synthesis_fallback_used,
            confidence_adjustment=float(orm.confidence_adjustment),
            synthesis_reasoning=orm.synthesis_reasoning,
            parallel_efficiency_percent=float(orm.parallel_efficiency_percent),
            timeout_triggered=orm.timeout_triggered,
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return "SupervisorMetricsRepository()"
