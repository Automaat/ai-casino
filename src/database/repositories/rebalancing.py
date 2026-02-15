"""Rebalancing record repository for database operations."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from loguru import logger
from sqlalchemy import select

from src.daemon.state.models import PortfolioAllocationRecord, PortfolioRebalancingRecord
from src.database.models import RebalancingRecordORM
from src.database.repositories.base import BaseRepository

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


class RebalancingRecordRepository(BaseRepository[PortfolioRebalancingRecord]):
    """Repository for rebalancing record persistence."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with database session.

        Args:
            session: SQLAlchemy async session
        """
        super().__init__(session)

    async def create(self, entity: PortfolioRebalancingRecord) -> PortfolioRebalancingRecord:
        """Create new rebalancing record.

        Args:
            entity: PortfolioRebalancingRecord to persist

        Returns:
            Created PortfolioRebalancingRecord
        """
        allocations_json = [
            {
                "symbol": alloc.symbol,
                "weight": alloc.weight,
                "action": alloc.action,
                "delta": alloc.delta,
            }
            for alloc in entity.allocations
        ]

        orm = RebalancingRecordORM(
            id=uuid.uuid4(),
            timestamp=entity.timestamp,
            method=entity.method,
            allocations=allocations_json,
            expected_return=Decimal(str(entity.expected_return)),
            expected_volatility=Decimal(str(entity.expected_volatility)),
            sharpe_ratio=Decimal(str(entity.sharpe_ratio)),
            rebalances_executed=entity.rebalances_executed,
            rebalances_pending=entity.rebalances_pending,
            created_at=datetime.now(UTC),
        )
        self._session.add(orm)
        await self._session.commit()
        logger.info(f"Created rebalancing record: {entity.method}")
        return entity

    async def get_by_id(self, entity_id: str) -> PortfolioRebalancingRecord | None:
        """Get rebalancing record by ID.

        Args:
            entity_id: Rebalancing record UUID string

        Returns:
            PortfolioRebalancingRecord if found, None otherwise
        """
        result = await self._session.execute(
            select(RebalancingRecordORM).where(RebalancingRecordORM.id == uuid.UUID(entity_id))
        )
        orm = result.scalar_one_or_none()
        return self._to_record(orm) if orm else None

    async def get_recent(self, limit: int = 100) -> list[PortfolioRebalancingRecord]:
        """Get recent rebalancing records.

        Args:
            limit: Maximum number of records to return

        Returns:
            List of recent PortfolioRebalancingRecords
        """
        result = await self._session.execute(
            select(RebalancingRecordORM).order_by(RebalancingRecordORM.timestamp.desc()).limit(limit)
        )
        return [self._to_record(orm) for orm in result.scalars().all()]

    def _to_record(self, orm: RebalancingRecordORM) -> PortfolioRebalancingRecord:
        """Convert ORM model to PortfolioRebalancingRecord.

        Args:
            orm: RebalancingRecordORM instance

        Returns:
            PortfolioRebalancingRecord
        """
        allocations = []
        if isinstance(orm.allocations, list):
            for alloc_dict in orm.allocations:
                allocations.append(
                    PortfolioAllocationRecord(
                        symbol=alloc_dict["symbol"],
                        weight=alloc_dict["weight"],
                        action=alloc_dict["action"],
                        delta=alloc_dict["delta"],
                    )
                )

        return PortfolioRebalancingRecord(
            timestamp=orm.timestamp,
            method=orm.method,
            allocations=allocations,
            expected_return=float(orm.expected_return),
            expected_volatility=float(orm.expected_volatility),
            sharpe_ratio=float(orm.sharpe_ratio),
            rebalances_executed=orm.rebalances_executed,
            rebalances_pending=orm.rebalances_pending,
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return "RebalancingRecordRepository()"
