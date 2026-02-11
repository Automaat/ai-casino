"""Snapshot state manager."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from loguru import logger
from pydantic import PrivateAttr

from src.daemon.state.managers.base import StateManager, _log_task_exception
from src.daemon.state.models import PortfolioSnapshot

if TYPE_CHECKING:
    from src.database.repositories.snapshot import PortfolioSnapshotRepository


class SnapshotStateManager(StateManager):
    """Portfolio snapshot persistence."""

    _snapshot_repository: PortfolioSnapshotRepository | None = PrivateAttr(default=None)

    def set_repository(self, repository: PortfolioSnapshotRepository) -> None:
        """Inject snapshot repository.

        Args:
            repository: Portfolio snapshot repository
        """
        self._snapshot_repository = repository
        logger.debug("Snapshot repository injected")

    def snapshot_portfolio(self, snapshot: PortfolioSnapshot) -> None:
        """Create portfolio snapshot and persist to database.

        Args:
            snapshot: Portfolio snapshot with balance, positions, and trigger info
        """
        if self._snapshot_repository:
            try:
                from src.database.repositories.snapshot import PortfolioSnapshot as DBSnapshot

                db_snapshot = DBSnapshot(
                    timestamp=datetime.now(UTC),
                    balance=snapshot.balance,
                    available_cash=snapshot.available_cash,
                    total_exposure=snapshot.total_exposure,
                    portfolio_value=snapshot.portfolio_value,
                    positions=snapshot.positions,
                    trigger=snapshot.trigger,
                )
                task = asyncio.create_task(self._snapshot_repository.create(db_snapshot))
                task.add_done_callback(_log_task_exception)
                logger.info(
                    f"Scheduled portfolio snapshot persistence: {snapshot.trigger} "
                    f"value={snapshot.portfolio_value}"
                )
            except Exception as e:
                logger.error(f"Failed to persist portfolio snapshot to database: {e}")
                raise

    def __repr__(self) -> str:
        """Return string representation."""
        return "SnapshotStateManager()"
