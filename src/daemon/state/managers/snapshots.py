"""Snapshot state manager."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from loguru import logger

from src.daemon.state.managers.base import StateManager, _make_task_cleanup_callback
from src.daemon.state.models import PortfolioSnapshot


class SnapshotStateManager(StateManager):
    """Portfolio snapshot persistence."""

    def snapshot_portfolio(self, snapshot: PortfolioSnapshot) -> None:
        """Create portfolio snapshot and persist to database.

        Args:
            snapshot: Portfolio snapshot with balance, positions, and trigger info
        """
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

            async def persist_snapshot() -> None:
                from src.database.connection import get_session
                from src.database.repositories.snapshot import PortfolioSnapshotRepository

                async with get_session() as session:
                    await PortfolioSnapshotRepository(session).create(db_snapshot)

            task = asyncio.create_task(persist_snapshot())
            self._pending_tasks.add(task)
            task.add_done_callback(_make_task_cleanup_callback(self._pending_tasks))
            logger.info(
                f"Scheduled portfolio snapshot persistence: {snapshot.trigger} "
                f"value={snapshot.portfolio_value}"
            )
        except Exception as e:
            logger.opt(exception=True).error(f"Failed to persist portfolio snapshot to database: {e}")
            raise

    def __repr__(self) -> str:
        """Return string representation."""
        return "SnapshotStateManager()"
