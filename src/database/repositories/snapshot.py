"""Portfolio snapshot repository for database operations."""

import uuid
from datetime import datetime
from decimal import Decimal

from loguru import logger
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import PortfolioSnapshotORM
from src.database.repositories.base import BaseRepository


class PortfolioSnapshot(BaseModel):
    """Portfolio snapshot data model."""

    id: str | None = None
    timestamp: datetime
    balance: float
    available_cash: float
    total_exposure: float
    portfolio_value: float
    positions: dict
    trigger: str


class PortfolioSnapshotRepository(BaseRepository[PortfolioSnapshot]):
    """Repository for portfolio snapshot persistence."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with database session.

        Args:
            session: SQLAlchemy async session
        """
        super().__init__(session)

    async def create(self, entity: PortfolioSnapshot) -> PortfolioSnapshot:
        """Create new portfolio snapshot.

        Args:
            entity: PortfolioSnapshot to persist

        Returns:
            Created PortfolioSnapshot with ID
        """
        snapshot_id = uuid.uuid4()
        orm = PortfolioSnapshotORM(
            id=snapshot_id,
            timestamp=entity.timestamp,
            balance=Decimal(str(entity.balance)),
            available_cash=Decimal(str(entity.available_cash)),
            total_exposure=Decimal(str(entity.total_exposure)),
            portfolio_value=Decimal(str(entity.portfolio_value)),
            positions=entity.positions,
            trigger=entity.trigger,
        )
        self._session.add(orm)
        await self._session.commit()
        logger.info(f"Created portfolio snapshot: {snapshot_id}")
        entity.id = str(snapshot_id)
        return entity

    async def get_by_id(self, entity_id: str) -> PortfolioSnapshot | None:
        """Get snapshot by ID.

        Args:
            entity_id: Snapshot UUID string

        Returns:
            PortfolioSnapshot if found, None otherwise
        """
        result = await self._session.execute(
            select(PortfolioSnapshotORM).where(PortfolioSnapshotORM.id == uuid.UUID(entity_id))
        )
        orm = result.scalar_one_or_none()
        return self._to_snapshot(orm) if orm else None

    async def get_latest(self) -> PortfolioSnapshot | None:
        """Get most recent portfolio snapshot.

        Returns:
            Latest PortfolioSnapshot or None if empty
        """
        result = await self._session.execute(
            select(PortfolioSnapshotORM).order_by(PortfolioSnapshotORM.timestamp.desc()).limit(1)
        )
        orm = result.scalar_one_or_none()
        return self._to_snapshot(orm) if orm else None

    async def get_by_date_range(self, start: datetime, end: datetime) -> list[PortfolioSnapshot]:
        """Get snapshots within date range.

        Args:
            start: Start datetime (inclusive)
            end: End datetime (inclusive)

        Returns:
            List of PortfolioSnapshots in range
        """
        result = await self._session.execute(
            select(PortfolioSnapshotORM)
            .where(PortfolioSnapshotORM.timestamp >= start)
            .where(PortfolioSnapshotORM.timestamp <= end)
            .order_by(PortfolioSnapshotORM.timestamp.asc())
        )
        return [self._to_snapshot(orm) for orm in result.scalars().all()]

    async def get_by_date_range_sampled(
        self, start: datetime, end: datetime, max_points: int = 100
    ) -> list[PortfolioSnapshot]:
        """Get snapshots with downsampling for large ranges.

        Args:
            start: Start datetime (inclusive)
            end: End datetime (inclusive)
            max_points: Maximum number of points to return

        Returns:
            List of PortfolioSnapshots (downsampled if needed)
        """
        count_result = await self._session.execute(
            select(func.count(PortfolioSnapshotORM.id))
            .where(PortfolioSnapshotORM.timestamp >= start)
            .where(PortfolioSnapshotORM.timestamp <= end)
        )
        total_count = count_result.scalar_one()

        if total_count <= max_points:
            return await self.get_by_date_range(start, end)

        result = await self._session.execute(
            select(PortfolioSnapshotORM)
            .where(PortfolioSnapshotORM.timestamp >= start)
            .where(PortfolioSnapshotORM.timestamp <= end)
            .order_by(PortfolioSnapshotORM.timestamp.asc())
        )
        all_snapshots = result.scalars().all()

        # Compute evenly spaced indices including first and last snapshots
        if max_points <= 1:
            indices = [total_count - 1]
        else:
            indices = [round(i * (total_count - 1) / (max_points - 1)) for i in range(max_points)]

        sampled = [all_snapshots[i] for i in indices]

        return [self._to_snapshot(orm) for orm in sampled]

    async def get_by_trigger(self, trigger: str) -> list[PortfolioSnapshot]:
        """Get snapshots by trigger type.

        Args:
            trigger: Trigger type (SCHEDULED/TRADE/MANUAL)

        Returns:
            List of PortfolioSnapshots with trigger
        """
        result = await self._session.execute(
            select(PortfolioSnapshotORM)
            .where(PortfolioSnapshotORM.trigger == trigger)
            .order_by(PortfolioSnapshotORM.timestamp.desc())
        )
        return [self._to_snapshot(orm) for orm in result.scalars().all()]

    def _to_snapshot(self, orm: PortfolioSnapshotORM) -> PortfolioSnapshot:
        """Convert ORM model to PortfolioSnapshot.

        Args:
            orm: PortfolioSnapshotORM instance

        Returns:
            PortfolioSnapshot
        """
        return PortfolioSnapshot(
            id=str(orm.id),
            timestamp=orm.timestamp,
            balance=float(orm.balance),
            available_cash=float(orm.available_cash),
            total_exposure=float(orm.total_exposure),
            portfolio_value=float(orm.portfolio_value),
            positions=orm.positions,
            trigger=orm.trigger,
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return "PortfolioSnapshotRepository()"
