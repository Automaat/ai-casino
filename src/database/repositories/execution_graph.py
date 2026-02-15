"""Execution graph repository for database operations."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from loguru import logger
from sqlalchemy import delete, select

from src.database.models import ExecutionGraphORM
from src.database.repositories.base import BaseRepository
from src.execution_tracking.models import ExecutionGraph

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


class ExecutionGraphRepository(BaseRepository[ExecutionGraph]):
    """Repository for execution graph persistence."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with database session.

        Args:
            session: SQLAlchemy async session
        """
        super().__init__(session)

    async def create(self, entity: ExecutionGraph) -> ExecutionGraph:
        """Create new execution graph.

        Args:
            entity: ExecutionGraph to persist

        Returns:
            Created ExecutionGraph
        """
        orm = ExecutionGraphORM(
            id=uuid.uuid4(),
            workflow_id=str(entity.workflow_id),
            symbol=entity.symbol,
            graph_jsonb=entity.model_dump(mode="json"),
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        self._session.add(orm)
        await self._session.commit()
        logger.info(f"Created execution graph: {entity.workflow_id}")
        return entity

    async def get_by_id(self, entity_id: str) -> ExecutionGraph | None:
        """Get execution graph by UUID.

        Args:
            entity_id: Graph UUID string

        Returns:
            ExecutionGraph if found, None otherwise
        """
        result = await self._session.execute(
            select(ExecutionGraphORM).where(ExecutionGraphORM.id == uuid.UUID(entity_id))
        )
        orm = result.scalar_one_or_none()
        return self._to_graph(orm) if orm else None

    async def get_by_workflow_id(self, workflow_id: str) -> ExecutionGraph | None:
        """Get execution graph by workflow ID.

        Args:
            workflow_id: Workflow ID string

        Returns:
            ExecutionGraph if found, None otherwise
        """
        result = await self._session.execute(
            select(ExecutionGraphORM).where(ExecutionGraphORM.workflow_id == workflow_id)
        )
        orm = result.scalar_one_or_none()
        return self._to_graph(orm) if orm else None

    async def list_recent(self, limit: int = 50, symbol: str | None = None) -> list[ExecutionGraph]:
        """Get recent execution graphs.

        Args:
            limit: Maximum number of graphs to return
            symbol: Optional symbol filter

        Returns:
            List of ExecutionGraphs ordered by created_at desc
        """
        stmt = select(ExecutionGraphORM).order_by(ExecutionGraphORM.created_at.desc()).limit(limit)
        if symbol:
            stmt = stmt.where(ExecutionGraphORM.symbol == symbol)

        result = await self._session.execute(stmt)
        return [self._to_graph(orm) for orm in result.scalars().all()]

    async def get_by_date_range(
        self,
        start: datetime,
        end: datetime,
        symbol: str | None = None,
        limit: int = 500,
    ) -> list[ExecutionGraph]:
        """Get execution graphs within date range.

        Args:
            start: Start datetime (inclusive)
            end: End datetime (inclusive)
            symbol: Optional symbol filter
            limit: Maximum number of graphs to return

        Returns:
            List of ExecutionGraphs in date range
        """
        stmt = (
            select(ExecutionGraphORM)
            .where(
                ExecutionGraphORM.created_at >= start,
                ExecutionGraphORM.created_at <= end,
            )
            .order_by(ExecutionGraphORM.created_at.desc())
            .limit(limit)
        )
        if symbol:
            stmt = stmt.where(ExecutionGraphORM.symbol == symbol)

        result = await self._session.execute(stmt)
        return [self._to_graph(orm) for orm in result.scalars().all()]

    async def delete_before(self, cutoff: datetime) -> int:
        """Delete execution graphs older than cutoff date.

        Args:
            cutoff: Delete graphs with created_at < cutoff

        Returns:
            Number of graphs deleted
        """
        result = await self._session.execute(
            delete(ExecutionGraphORM).where(ExecutionGraphORM.created_at < cutoff)
        )
        await self._session.commit()
        deleted_count = getattr(result, "rowcount", 0)
        logger.info(f"Deleted {deleted_count} execution graphs before {cutoff}")
        return deleted_count

    def _to_graph(self, orm: ExecutionGraphORM) -> ExecutionGraph:
        """Convert ORM model to ExecutionGraph.

        Args:
            orm: ExecutionGraphORM instance

        Returns:
            ExecutionGraph

        Raises:
            TypeError: If JSONB data is invalid
        """
        if not isinstance(orm.graph_jsonb, dict):
            msg = f"Invalid JSONB data for graph {orm.workflow_id}"
            raise TypeError(msg)

        return ExecutionGraph(**orm.graph_jsonb)

    def __repr__(self) -> str:
        """Return string representation."""
        return "ExecutionGraphRepository()"
