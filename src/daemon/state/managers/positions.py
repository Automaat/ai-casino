"""Position state manager."""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger
from pydantic import Field

from src.daemon.state.managers.base import StateManager

if TYPE_CHECKING:
    from src.daemon.positions import PositionManagementAction, PositionRecord


class PositionStateManager(StateManager):
    """Active portfolio position CRUD."""

    active_positions: dict[str, dict] = Field(default_factory=dict)
    position_management_history: list[dict] = Field(default_factory=list)

    def add_position(self, position: PositionRecord) -> None:
        """Add or update position in state.

        Args:
            position: Position record to add
        """
        self.active_positions[position.symbol] = position.model_dump(mode="json")
        logger.debug(f"Added position: {position.symbol}")

    def remove_position(self, symbol: str) -> None:
        """Remove position from state.

        Args:
            symbol: Stock ticker to remove
        """
        if symbol in self.active_positions:
            self.active_positions.pop(symbol)
            logger.debug(f"Removed position: {symbol}")

    def update_position(self, position: PositionRecord) -> None:
        """Update existing position in state.

        Args:
            position: Position record to update
        """
        self.add_position(position)

    def record_position_action(self, action: PositionManagementAction) -> None:
        """Record position management action.

        Args:
            action: Action to record
        """
        self.position_management_history.append(action.model_dump(mode="json"))
        self.position_management_history = self._cap_history(self.position_management_history, 100, 100)

    def get_position(self, symbol: str) -> PositionRecord | None:
        """Get position record by symbol.

        Args:
            symbol: Stock ticker

        Returns:
            PositionRecord or None
        """
        from src.daemon.positions import PositionRecord

        if symbol not in self.active_positions:
            return None
        return PositionRecord.model_validate(self.active_positions[symbol])

    def __repr__(self) -> str:
        """Return string representation."""
        return f"PositionStateManager(positions={len(self.active_positions)})"
