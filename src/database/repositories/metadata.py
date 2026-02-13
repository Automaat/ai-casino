"""Metadata repository for daemon scalar state."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from loguru import logger
from sqlalchemy import select

from src.database.models import DaemonMetadataORM
from src.database.repositories.base import BaseRepository

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


class MetadataRepository(BaseRepository[dict]):
    """Repository for daemon metadata key-value persistence."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with database session.

        Args:
            session: SQLAlchemy async session
        """
        super().__init__(session)
        logger.debug("Initialized MetadataRepository")

    async def create(self, entity: dict) -> dict:
        """Create metadata entry (use set instead)."""
        msg = "Use set() method instead"
        raise NotImplementedError(msg)

    async def get_by_id(self, entity_id: str) -> dict | None:
        """Get metadata by key (use get instead)."""
        return await self.get(entity_id)

    async def get(self, key: str) -> datetime | int | str | float | list | dict | None:
        """Get metadata value by key.

        Args:
            key: Metadata key (e.g., "trading.last_run")

        Returns:
            Metadata value (datetime, int, str, list, dict, etc.) or None if not found
        """
        result = await self._session.execute(select(DaemonMetadataORM).where(DaemonMetadataORM.key == key))
        orm = result.scalar_one_or_none()
        if not orm:
            return None

        # Extract value from JSONB (stored as {"data": <value>})
        value_dict = orm.value
        if not isinstance(value_dict, dict) or "data" not in value_dict:
            return None

        value = value_dict["data"]

        # Parse datetime strings back to datetime objects
        if isinstance(value, str) and value.endswith("Z"):
            try:
                # Replace Z with UTC timezone - fromisoformat handles +00:00
                return datetime.fromisoformat(value.removesuffix("Z")).replace(tzinfo=UTC)
            except ValueError:
                return value

        return value

    async def set(self, key: str, value: datetime | int | str | float | list | dict) -> None:
        """Set metadata value by key.

        Args:
            key: Metadata key (e.g., "trading.last_run")
            value: Value to store (datetime, int, str, list, dict, etc.)
        """
        # Serialize datetime to ISO format string
        serialized = value.isoformat() if isinstance(value, datetime) else value

        # Check if exists
        result = await self._session.execute(select(DaemonMetadataORM).where(DaemonMetadataORM.key == key))
        orm = result.scalar_one_or_none()

        if orm:
            # Update existing
            orm.value = {"data": serialized}
            orm.updated_at = datetime.now(UTC)
        else:
            # Create new
            orm = DaemonMetadataORM(key=key, value={"data": serialized}, updated_at=datetime.now(UTC))
            self._session.add(orm)

        await self._session.commit()
        logger.debug(f"Set metadata: {key} = {value}")

    async def delete(self, key: str) -> bool:
        """Delete metadata by key.

        Args:
            key: Metadata key to delete

        Returns:
            True if deleted, False if not found
        """
        result = await self._session.execute(select(DaemonMetadataORM).where(DaemonMetadataORM.key == key))
        orm = result.scalar_one_or_none()
        if not orm:
            return False

        await self._session.delete(orm)
        await self._session.commit()
        logger.debug(f"Deleted metadata: {key}")
        return True

    def __repr__(self) -> str:
        """Return string representation."""
        return "MetadataRepository()"
