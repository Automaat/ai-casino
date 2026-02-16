"""Workflow execution metrics repository for database operations."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from loguru import logger
from sqlalchemy import select

from src.database.models import WorkflowExecutionMetricsORM
from src.database.repositories.base import BaseRepository
from src.metrics.execution import (
    AgentTimingMetric,
    LLMCallMetric,
    PipelineStageMetric,
    SubOperationMetric,
    WorkflowExecutionMetrics,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


class WorkflowExecutionMetricsRepository(BaseRepository[WorkflowExecutionMetrics]):
    """Repository for workflow execution metrics persistence."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with database session.

        Args:
            session: SQLAlchemy async session
        """
        super().__init__(session)

    async def create(self, entity: WorkflowExecutionMetrics) -> WorkflowExecutionMetrics:
        """Create new workflow execution metrics record.

        Args:
            entity: WorkflowExecutionMetrics to persist

        Returns:
            Created WorkflowExecutionMetrics
        """
        orm = WorkflowExecutionMetricsORM(
            workflow_id=uuid.UUID(entity.workflow_id),
            symbol=entity.symbol,
            timestamp=entity.timestamp,
            total_latency_ms=Decimal(str(entity.total_latency_ms)),
            provider=entity.provider,
            model=entity.model,
            total_input_tokens=entity.total_input_tokens,
            total_output_tokens=entity.total_output_tokens,
            total_estimated_cost_usd=Decimal(str(entity.total_estimated_cost_usd)),
            llm_calls=[call.model_dump(mode="json") for call in entity.llm_calls],
            sub_operations=[op.model_dump(mode="json") for op in entity.sub_operations],
            agent_timings=[timing.model_dump(mode="json") for timing in entity.agent_timings],
            pipeline_stages=[stage.model_dump(mode="json") for stage in entity.pipeline_stages],
            created_at=datetime.now(UTC),
        )
        self._session.add(orm)
        await self._session.commit()
        logger.info(
            f"Created workflow execution metrics: {entity.workflow_id} ({entity.symbol}, "
            f"${entity.total_estimated_cost_usd:.6f})"
        )
        return entity

    async def get_by_workflow_id(self, workflow_id: str) -> WorkflowExecutionMetrics | None:
        """Get workflow execution metrics by workflow ID.

        Args:
            workflow_id: Workflow UUID string

        Returns:
            WorkflowExecutionMetrics if found, None otherwise
        """
        result = await self._session.execute(
            select(WorkflowExecutionMetricsORM).where(
                WorkflowExecutionMetricsORM.workflow_id == uuid.UUID(workflow_id)
            )
        )
        orm = result.scalar_one_or_none()
        return self._to_entity(orm) if orm else None

    async def list_recent(self, limit: int = 50, symbol: str | None = None) -> list[WorkflowExecutionMetrics]:
        """Get recent workflow execution metrics.

        Args:
            limit: Maximum number of records to return
            symbol: Optional symbol filter

        Returns:
            List of recent WorkflowExecutionMetrics
        """
        stmt = (
            select(WorkflowExecutionMetricsORM)
            .order_by(WorkflowExecutionMetricsORM.timestamp.desc())
            .limit(limit)
        )
        if symbol:
            stmt = stmt.where(WorkflowExecutionMetricsORM.symbol == symbol)

        result = await self._session.execute(stmt)
        return [self._to_entity(orm) for orm in result.scalars().all()]

    async def get_by_id(self, entity_id: str) -> WorkflowExecutionMetrics | None:
        """Get workflow execution metrics by workflow ID (alias for get_by_workflow_id).

        Args:
            entity_id: Workflow UUID string

        Returns:
            WorkflowExecutionMetrics if found, None otherwise
        """
        return await self.get_by_workflow_id(entity_id)

    def _to_entity(self, orm: WorkflowExecutionMetricsORM) -> WorkflowExecutionMetrics:
        """Convert ORM model to WorkflowExecutionMetrics.

        Args:
            orm: WorkflowExecutionMetricsORM instance

        Returns:
            WorkflowExecutionMetrics
        """
        return WorkflowExecutionMetrics(
            workflow_id=str(orm.workflow_id),
            symbol=orm.symbol,
            timestamp=orm.timestamp,
            total_latency_ms=float(orm.total_latency_ms),
            provider=orm.provider,
            model=orm.model,
            total_input_tokens=orm.total_input_tokens,
            total_output_tokens=orm.total_output_tokens,
            total_estimated_cost_usd=float(orm.total_estimated_cost_usd),
            llm_calls=[LLMCallMetric.model_validate(call) for call in orm.llm_calls],
            sub_operations=[SubOperationMetric.model_validate(op) for op in orm.sub_operations],
            agent_timings=[AgentTimingMetric.model_validate(timing) for timing in orm.agent_timings],
            pipeline_stages=[PipelineStageMetric.model_validate(stage) for stage in orm.pipeline_stages],
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return "WorkflowExecutionMetricsRepository()"
