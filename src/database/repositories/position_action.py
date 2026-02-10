"""Position management action repository for database operations."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING

from loguru import logger
from sqlalchemy import delete, select

from src.daemon.positions import PositionManagementAction
from src.database.models import PositionManagementActionORM
from src.database.repositories.base import BaseRepository

if TYPE_CHECKING:
    from sqlalchemy.engine import Result
    from sqlalchemy.ext.asyncio import AsyncSession


class PositionManagementActionRepository(BaseRepository[PositionManagementAction]):
    """Repository for position management action persistence."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with database session.

        Args:
            session: SQLAlchemy async session
        """
        super().__init__(session)
        logger.debug("Initialized PositionManagementActionRepository")

    async def create(self, entity: PositionManagementAction) -> PositionManagementAction:
        """Create new position management action.

        Args:
            entity: PositionManagementAction to persist

        Returns:
            Created PositionManagementAction
        """
        orm = PositionManagementActionORM(
            id=uuid.uuid4(),
            symbol=entity.symbol,
            action_type=entity.action_type,
            timestamp=entity.timestamp,
            old_stop_loss=Decimal(str(entity.old_stop_loss)) if entity.old_stop_loss is not None else None,
            new_stop_loss=Decimal(str(entity.new_stop_loss)) if entity.new_stop_loss is not None else None,
            qty_sold=Decimal(str(entity.qty_sold)) if entity.qty_sold is not None else None,
            price=Decimal(str(entity.price)),
            reason=entity.reason,
            executed=entity.executed,
            order_id=entity.order_id,
            created_at=datetime.now(UTC),
        )
        self._session.add(orm)
        await self._session.commit()
        logger.info(f"Created position action: {entity.symbol} {entity.action_type}")
        return entity

    async def get_by_id(self, entity_id: str) -> PositionManagementAction | None:
        """Get position action by ID.

        Args:
            entity_id: Position action UUID string

        Returns:
            PositionManagementAction if found, None otherwise
        """
        result = await self._session.execute(
            select(PositionManagementActionORM).where(PositionManagementActionORM.id == uuid.UUID(entity_id))
        )
        orm = result.scalar_one_or_none()
        return self._to_action(orm) if orm else None

    async def get_by_symbol(self, symbol: str, limit: int = 100) -> list[PositionManagementAction]:
        """Get position actions for specific symbol.

        Args:
            symbol: Stock ticker symbol
            limit: Maximum number of actions to return

        Returns:
            List of PositionManagementActions for symbol
        """
        result = await self._session.execute(
            select(PositionManagementActionORM)
            .where(PositionManagementActionORM.symbol == symbol)
            .order_by(PositionManagementActionORM.timestamp.desc())
            .limit(limit)
        )
        return [self._to_action(orm) for orm in result.scalars().all()]

    async def get_recent_actions(self, days: int = 30) -> list[PositionManagementAction]:
        """Get recent position actions across all symbols.

        Args:
            days: Number of days to look back

        Returns:
            List of recent PositionManagementActions
        """
        cutoff = datetime.now(UTC) - timedelta(days=days)
        result = await self._session.execute(
            select(PositionManagementActionORM)
            .where(PositionManagementActionORM.created_at >= cutoff)
            .order_by(PositionManagementActionORM.timestamp.desc())
        )
        return [self._to_action(orm) for orm in result.scalars().all()]

    async def delete_before(self, cutoff: datetime) -> int:
        """Delete position actions older than cutoff date.

        Args:
            cutoff: Delete actions with created_at < cutoff

        Returns:
            Number of actions deleted
        """
        result: Result = await self._session.execute(
            delete(PositionManagementActionORM).where(PositionManagementActionORM.created_at < cutoff)
        )
        await self._session.commit()
        deleted_count = result.rowcount or 0  # type: ignore[missing-attribute]
        logger.info(f"Deleted {deleted_count} position actions before {cutoff}")
        return deleted_count

    def _to_action(self, orm: PositionManagementActionORM) -> PositionManagementAction:
        """Convert ORM model to PositionManagementAction.

        Args:
            orm: PositionManagementActionORM instance

        Returns:
            PositionManagementAction
        """
        return PositionManagementAction(
            symbol=orm.symbol,
            action_type=orm.action_type,
            timestamp=orm.timestamp,
            old_stop_loss=float(orm.old_stop_loss) if orm.old_stop_loss is not None else None,
            new_stop_loss=float(orm.new_stop_loss) if orm.new_stop_loss is not None else None,
            qty_sold=float(orm.qty_sold) if orm.qty_sold is not None else None,
            price=float(orm.price),
            reason=orm.reason,
            executed=orm.executed,
            order_id=orm.order_id,
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return "PositionManagementActionRepository()"
