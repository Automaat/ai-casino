"""Position record repository for database operations."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from loguru import logger
from sqlalchemy import delete, select

from src.daemon.positions import PositionRecord
from src.database.models import PositionRecordORM
from src.database.repositories.base import BaseRepository

if TYPE_CHECKING:
    from sqlalchemy.engine import Result
    from sqlalchemy.ext.asyncio import AsyncSession


class PositionRecordRepository(BaseRepository[PositionRecord]):
    """Repository for position record persistence."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with database session.

        Args:
            session: SQLAlchemy async session
        """
        super().__init__(session)

    async def create(self, entity: PositionRecord) -> PositionRecord:
        """Create new position record.

        Args:
            entity: PositionRecord to persist

        Returns:
            Created PositionRecord
        """
        orm = PositionRecordORM(
            id=uuid.uuid4(),
            symbol=entity.symbol,
            entry_timestamp=entity.entry_timestamp,
            entry_price=Decimal(str(entity.entry_price)),
            entry_signal=entity.entry_signal,
            entry_confidence=Decimal(str(entity.entry_confidence)),
            current_qty=Decimal(str(entity.current_qty)),
            current_stop_loss=Decimal(str(entity.current_stop_loss)),
            initial_stop_loss=Decimal(str(entity.initial_stop_loss)),
            stop_loss_order_id=entity.stop_loss_order_id,
            profit_targets=entity.profit_targets,  # list stored as JSONB
            days_held=entity.days_held,
            last_updated=entity.last_updated,
            trailing_stop_activated=entity.trailing_stop_activated,
            breakeven_activated=entity.breakeven_activated,
            high_water_mark=Decimal(str(entity.high_water_mark))
            if entity.high_water_mark is not None
            else None,
            created_at=datetime.now(UTC),
        )
        self._session.add(orm)
        await self._session.commit()
        logger.info(f"Created position record: {entity.symbol} qty={entity.current_qty}")
        return entity

    async def update(self, entity: PositionRecord) -> PositionRecord:
        """Update existing position record.

        Args:
            entity: PositionRecord with updated values

        Returns:
            Updated PositionRecord
        """
        result = await self._session.execute(
            select(PositionRecordORM).where(PositionRecordORM.symbol == entity.symbol)
        )
        orm = result.scalar_one_or_none()
        if not orm:
            msg = f"Position not found: {entity.symbol}"
            raise ValueError(msg)

        orm.current_qty = Decimal(str(entity.current_qty))
        orm.current_stop_loss = Decimal(str(entity.current_stop_loss))
        orm.stop_loss_order_id = entity.stop_loss_order_id
        orm.profit_targets = entity.profit_targets
        orm.days_held = entity.days_held
        orm.last_updated = entity.last_updated
        orm.trailing_stop_activated = entity.trailing_stop_activated
        orm.breakeven_activated = entity.breakeven_activated
        orm.high_water_mark = (
            Decimal(str(entity.high_water_mark)) if entity.high_water_mark is not None else None
        )

        await self._session.commit()
        logger.info(f"Updated position record: {entity.symbol}")
        return entity

    async def get_by_id(self, entity_id: str) -> PositionRecord | None:
        """Get position record by ID.

        Args:
            entity_id: Position record UUID string

        Returns:
            PositionRecord if found, None otherwise
        """
        result = await self._session.execute(
            select(PositionRecordORM).where(PositionRecordORM.id == uuid.UUID(entity_id))
        )
        orm = result.scalar_one_or_none()
        return self._to_record(orm) if orm else None

    async def get_by_symbol(self, symbol: str) -> PositionRecord | None:
        """Get position record by symbol.

        Args:
            symbol: Stock ticker symbol

        Returns:
            PositionRecord if found, None otherwise
        """
        result = await self._session.execute(
            select(PositionRecordORM).where(PositionRecordORM.symbol == symbol)
        )
        orm = result.scalar_one_or_none()
        return self._to_record(orm) if orm else None

    async def get_all_active(self) -> list[PositionRecord]:
        """Get all active position records.

        Returns:
            List of all PositionRecords
        """
        result = await self._session.execute(
            select(PositionRecordORM).order_by(PositionRecordORM.last_updated.desc())
        )
        return [self._to_record(orm) for orm in result.scalars().all()]

    async def delete_by_symbol(self, symbol: str) -> bool:
        """Delete position record by symbol.

        Args:
            symbol: Stock ticker symbol

        Returns:
            True if deleted, False if not found
        """
        result: Result = await self._session.execute(
            delete(PositionRecordORM).where(PositionRecordORM.symbol == symbol)
        )
        await self._session.commit()
        deleted = (result.rowcount or 0) > 0  # type: ignore[missing-attribute]
        if deleted:
            logger.info(f"Deleted position record: {symbol}")
        return deleted

    def _to_record(self, orm: PositionRecordORM) -> PositionRecord:
        """Convert ORM model to PositionRecord.

        Args:
            orm: PositionRecordORM instance

        Returns:
            PositionRecord
        """
        return PositionRecord(
            symbol=orm.symbol,
            entry_timestamp=orm.entry_timestamp,
            entry_price=float(orm.entry_price),
            entry_signal=orm.entry_signal,
            entry_confidence=float(orm.entry_confidence),
            current_qty=float(orm.current_qty),
            current_stop_loss=float(orm.current_stop_loss),
            initial_stop_loss=float(orm.initial_stop_loss),
            stop_loss_order_id=orm.stop_loss_order_id,
            profit_targets=orm.profit_targets if isinstance(orm.profit_targets, list) else [],
            days_held=orm.days_held,
            last_updated=orm.last_updated,
            trailing_stop_activated=orm.trailing_stop_activated,
            breakeven_activated=orm.breakeven_activated,
            high_water_mark=float(orm.high_water_mark) if orm.high_water_mark is not None else None,
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return "PositionRecordRepository()"
