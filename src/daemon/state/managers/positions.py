"""Position state manager."""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger
from pydantic import PrivateAttr

from src.daemon.state.managers.base import StateManager

if TYPE_CHECKING:
    from src.daemon.positions import PositionManagementAction, PositionRecord
    from src.database.repositories.position import PositionRecordRepository
    from src.database.repositories.position_action import PositionManagementActionRepository


class PositionStateManager(StateManager):
    """Active portfolio position CRUD."""

    _position_repository: PositionRecordRepository | None = PrivateAttr(default=None)
    _position_action_repository: PositionManagementActionRepository | None = PrivateAttr(default=None)

    def set_repositories(
        self,
        position_repository: PositionRecordRepository,
        position_action_repository: PositionManagementActionRepository,
    ) -> None:
        """Inject repositories.

        Args:
            position_repository: Position record repository
            position_action_repository: Position action repository
        """
        self._position_repository = position_repository
        self._position_action_repository = position_action_repository
        logger.debug("PositionStateManager repositories injected")

    async def add_position(self, position: PositionRecord) -> None:
        """Add or update position in state.

        Args:
            position: Position record to add
        """
        if not self._position_repository:
            logger.warning("Position repository not available")
            return

        # Check if exists
        existing = await self._position_repository.get_by_symbol(position.symbol)
        if existing:
            await self._position_repository.update(position)
            logger.debug(f"Updated position: {position.symbol}")
        else:
            await self._position_repository.create(position)
            logger.debug(f"Added position: {position.symbol}")

    async def remove_position(self, symbol: str) -> None:
        """Remove position from state.

        Args:
            symbol: Stock ticker to remove
        """
        if not self._position_repository:
            logger.warning("Position repository not available")
            return

        deleted = await self._position_repository.delete_by_symbol(symbol)
        if deleted:
            logger.debug(f"Removed position: {symbol}")

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
        if not self._position_action_repository:
            logger.warning("Position action repository not available")
            return

        await self._position_action_repository.create(action)
        logger.debug(f"Recorded position action: {action.symbol} {action.action_type}")

    async def get_position(self, symbol: str) -> PositionRecord | None:
        """Get position record by symbol.

        Args:
            symbol: Stock ticker

        Returns:
            PositionRecord or None
        """
        if not self._position_repository:
            return None
        return await self._position_repository.get_by_symbol(symbol)

    async def get_all_positions(self) -> list[PositionRecord]:
        """Get all active positions.

        Returns:
            List of all PositionRecords
        """
        if not self._position_repository:
            return []
        return await self._position_repository.get_all_active()

    async def get_recent_actions(self, symbol: str | None = None, limit: int = 100) -> list[PositionManagementAction]:
        """Get recent position management actions.

        Args:
            symbol: Optional symbol filter
            limit: Maximum number of actions to return

        Returns:
            List of recent PositionManagementActions
        """
        if not self._position_action_repository:
            return []
        return await self._position_action_repository.get_recent(symbol=symbol, limit=limit)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"PositionStateManager(positions={len(self.active_positions)})"
