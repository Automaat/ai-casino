"""Position state manager."""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger
from pydantic import PrivateAttr

from src.daemon.state.managers.base import StateManager

if TYPE_CHECKING:
    from src.daemon.positions import PositionManagementAction, PositionRecord
    from src.database.engine import DatabaseEngine


class PositionStateManager(StateManager):
    """Active portfolio position CRUD."""

    _database_engine: DatabaseEngine | None = PrivateAttr(default=None)

    def set_repositories(
        self,
        position_repository: object,
        position_action_repository: object,
    ) -> None:
        """Enable database persistence (deprecated - use set_database_engine).

        Args:
            position_repository: Ignored (for API compatibility)
            position_action_repository: Ignored (for API compatibility)
        """
        # No-op for backward compatibility
        logger.debug("PositionStateManager.set_repositories called (no-op)")

    def set_database_engine(self, engine: DatabaseEngine) -> None:
        """Set database engine for creating fresh sessions.

        Args:
            engine: DatabaseEngine instance
        """
        self._database_engine = engine
        logger.debug("PositionStateManager database engine set")

    async def add_position(self, position: PositionRecord) -> None:
        """Add or update position in state.

        Args:
            position: Position record to add
        """
        if not self._database_engine:
            return

        try:
            from src.database.repositories.position import PositionRecordRepository

            async with self._database_engine.session() as session:
                repo = PositionRecordRepository(session)
                # Check if exists
                existing = await repo.get_by_symbol(position.symbol)
                if existing:
                    await repo.update(position)
                    logger.debug(f"Updated position: {position.symbol}")
                else:
                    await repo.create(position)
                    logger.debug(f"Added position: {position.symbol}")
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to add/update position: {e}")

    async def remove_position(self, symbol: str) -> None:
        """Remove position from state.

        Args:
            symbol: Stock ticker to remove
        """
        if not self._database_engine:
            return

        try:
            from src.database.repositories.position import PositionRecordRepository

            async with self._database_engine.session() as session:
                repo = PositionRecordRepository(session)
                deleted = await repo.delete_by_symbol(symbol)
                if deleted:
                    logger.debug(f"Removed position: {symbol}")
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to remove position: {e}")

    async def update_position(self, position: PositionRecord) -> None:
        """Update existing position in state.

        Args:
            position: Position record to update
        """
        await self.add_position(position)

    async def record_position_action(self, action: PositionManagementAction) -> None:
        """Record position management action.

        Args:
            action: Action to record
        """
        if not self._database_engine:
            return

        try:
            from src.database.repositories.position_action import PositionManagementActionRepository

            async with self._database_engine.session() as session:
                repo = PositionManagementActionRepository(session)
                await repo.create(action)
                logger.debug(f"Recorded position action: {action.symbol} {action.action_type}")
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to record position action: {e}")

    async def get_position(self, symbol: str) -> PositionRecord | None:
        """Get position record by symbol.

        Args:
            symbol: Stock ticker

        Returns:
            PositionRecord or None
        """
        if not self._database_engine:
            return None

        try:
            from src.database.repositories.position import PositionRecordRepository

            async with self._database_engine.session() as session:
                repo = PositionRecordRepository(session)
                return await repo.get_by_symbol(symbol)
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to get position: {e}")
            return None

    async def get_all_positions(self) -> list[PositionRecord]:
        """Get all active positions.

        Returns:
            List of all PositionRecords
        """
        if not self._database_engine:
            return []

        try:
            from src.database.repositories.position import PositionRecordRepository

            async with self._database_engine.session() as session:
                repo = PositionRecordRepository(session)
                return await repo.get_all_active()
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to get all positions: {e}")
            return []

    async def get_recent_actions(
        self, symbol: str | None = None, limit: int = 100
    ) -> list[PositionManagementAction]:
        """Get recent position management actions.

        Args:
            symbol: Optional symbol filter
            limit: Maximum number of actions to return

        Returns:
            List of recent PositionManagementActions
        """
        if not self._database_engine:
            return []

        try:
            from src.database.repositories.position_action import PositionManagementActionRepository

            async with self._database_engine.session() as session:
                repo = PositionManagementActionRepository(session)
                return await repo.get_recent(symbol=symbol, limit=limit)
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to get recent actions: {e}")
            return []

    async def get_active_positions(self) -> dict[str, dict]:
        """Get active positions as dict for backward compatibility.

        Returns:
            Dict mapping symbol to position dict
        """
        positions = await self.get_all_positions()
        return {pos.symbol: pos.model_dump() for pos in positions}

    async def get_position_management_history(self) -> list[dict]:
        """Get position management history as list of dicts.

        Returns:
            List of action dicts
        """
        actions = await self.get_recent_actions(limit=100)
        return [action.model_dump() for action in actions]

    def __repr__(self) -> str:
        """Return string representation."""
        return "PositionStateManager(db_backed)"
