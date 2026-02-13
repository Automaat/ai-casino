"""Repository for execution metric persistence."""

import uuid
from decimal import Decimal
from typing import TYPE_CHECKING

from loguru import logger
from sqlalchemy import func, select

from src.database.models import ExecutionMetricORM
from src.database.repositories.base import BaseRepository
from src.metrics.execution_metric import ExecutionMetric

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession
else:
    AsyncSession = object


class ExecutionMetricRepository(BaseRepository[ExecutionMetric]):
    """Repository for execution metric operations."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with database session.

        Args:
            session: SQLAlchemy async session
        """
        super().__init__(session)
        logger.debug("Initialized ExecutionMetricRepository")

    async def create(self, entity: ExecutionMetric) -> ExecutionMetric:
        """Create execution metric record.

        Args:
            entity: ExecutionMetric to persist

        Returns:
            ExecutionMetric with generated ID
        """
        orm = ExecutionMetricORM(
            id=uuid.uuid4(),
            order_id=entity.order_id,
            symbol=entity.symbol,
            side=entity.side,
            quantity=entity.quantity,
            requested_price=entity.requested_price,
            filled_price=entity.filled_price,
            submitted_at=entity.submitted_at,
            filled_at=entity.filled_at,
            execution_time_ms=entity.execution_time_ms,
            slippage_bps=entity.slippage_bps,
            broker=entity.broker,
            venue=entity.venue,
            status=entity.status,
            created_at=entity.created_at,
        )
        self._session.add(orm)
        await self._session.commit()
        logger.info(f"Created execution metric: {entity.order_id} (slippage: {entity.slippage_bps}bps)")
        entity.id = str(orm.id)
        return entity

    async def get_by_id(self, entity_id: str) -> ExecutionMetric | None:
        """Get execution metric by UUID.

        Args:
            entity_id: UUID string

        Returns:
            ExecutionMetric or None if not found
        """
        result = await self._session.execute(
            select(ExecutionMetricORM).where(ExecutionMetricORM.id == uuid.UUID(entity_id))
        )
        orm = result.scalar_one_or_none()
        return self._to_entity(orm) if orm else None

    async def get_by_order_id(self, order_id: str) -> ExecutionMetric | None:
        """Get execution metric by broker order ID.

        Args:
            order_id: Broker order ID

        Returns:
            ExecutionMetric or None if not found
        """
        result = await self._session.execute(
            select(ExecutionMetricORM).where(ExecutionMetricORM.order_id == order_id)
        )
        orm = result.scalar_one_or_none()
        return self._to_entity(orm) if orm else None

    async def get_by_symbol(self, symbol: str, limit: int = 100) -> list[ExecutionMetric]:
        """Get recent executions for symbol.

        Args:
            symbol: Stock ticker
            limit: Max records to return

        Returns:
            List of ExecutionMetrics ordered by submitted_at DESC
        """
        result = await self._session.execute(
            select(ExecutionMetricORM)
            .where(ExecutionMetricORM.symbol == symbol)
            .order_by(ExecutionMetricORM.submitted_at.desc())
            .limit(limit)
        )
        return [self._to_entity(orm) for orm in result.scalars()]

    async def get_avg_slippage_by_broker(self, broker: str | None = None) -> Decimal:
        """Calculate average slippage (in bps) by broker.

        Args:
            broker: Broker name filter (None = all brokers)

        Returns:
            Average slippage in basis points
        """
        query = select(func.avg(ExecutionMetricORM.slippage_bps)).where(
            ExecutionMetricORM.slippage_bps.isnot(None)
        )
        if broker:
            query = query.where(ExecutionMetricORM.broker == broker)

        result = await self._session.execute(query)
        avg_slippage = result.scalar_one_or_none()
        return Decimal(str(avg_slippage)) if avg_slippage else Decimal("0.0")

    def _to_entity(self, orm: ExecutionMetricORM) -> ExecutionMetric:
        """Convert ORM model to domain entity.

        Args:
            orm: ExecutionMetricORM instance

        Returns:
            ExecutionMetric domain model
        """
        return ExecutionMetric(
            id=str(orm.id),
            order_id=orm.order_id,
            symbol=orm.symbol,
            side=orm.side,
            quantity=orm.quantity,
            requested_price=orm.requested_price,
            filled_price=orm.filled_price,
            submitted_at=orm.submitted_at,
            filled_at=orm.filled_at,
            execution_time_ms=orm.execution_time_ms,
            slippage_bps=orm.slippage_bps,
            broker=orm.broker,
            venue=orm.venue,
            status=orm.status,
            created_at=orm.created_at,
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return "ExecutionMetricRepository()"
