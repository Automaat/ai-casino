"""Position persistence operations."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from loguru import logger

from src.daemon.positions.models import PositionManagementAction, PositionRecord

if TYPE_CHECKING:
    from src.database.engine import DatabaseEngine


def _make_task_cleanup_callback(task_set: set[asyncio.Task[Any]]) -> Callable[[asyncio.Task[object]], None]:
    """Create callback that removes task from set and logs exceptions."""

    def _cleanup_and_log(task: asyncio.Task[object]) -> None:
        """Log exceptions and remove task from tracking set."""
        task_set.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.opt(exception=exc).error("Background task failed")

    return _cleanup_and_log


class PositionPersistenceManager:
    """Handle position database persistence."""

    def __init__(self, database_engine: DatabaseEngine | None = None) -> None:
        """Initialize persistence manager.

        Args:
            database_engine: Optional database engine for creating per-task sessions
        """
        self._database_engine = database_engine
        self._pending_tasks: set[asyncio.Task[Any]] = set()

    @property
    def database_engine(self) -> DatabaseEngine | None:
        """Get database engine."""
        return self._database_engine

    def set_database(self, database_engine: DatabaseEngine) -> None:
        """Set database engine after initialization.

        Args:
            database_engine: Database engine for creating per-task sessions
        """
        self._database_engine = database_engine

    def persist_position_create(self, position: PositionRecord) -> None:
        """Persist new position to database."""
        if self._database_engine:
            try:
                task = asyncio.create_task(self._async_persist_position_create(position))
                self._pending_tasks.add(task)
                task.add_done_callback(_make_task_cleanup_callback(self._pending_tasks))
            except Exception as e:
                logger.opt(exception=True).error(f"Failed to persist new position to database: {e}")
                raise

    async def _async_persist_position_create(self, position: PositionRecord) -> None:
        """Async helper to persist position with fresh session."""
        from src.database.repositories.position import PositionRecordRepository

        if self._database_engine is None:
            msg = "Database engine not initialized"
            raise RuntimeError(msg)
        async with self._database_engine.session() as session:
            repository = PositionRecordRepository(session)
            await repository.create(position)

    def persist_position_update(self, position: PositionRecord) -> None:
        """Persist position update to database."""
        if self._database_engine:
            try:
                task = asyncio.create_task(self._async_persist_position_update(position))
                self._pending_tasks.add(task)
                task.add_done_callback(_make_task_cleanup_callback(self._pending_tasks))
            except Exception as e:
                logger.opt(exception=True).error(f"Failed to update position in database: {e}")
                raise

    async def _async_persist_position_update(self, position: PositionRecord) -> None:
        """Async helper to persist position update with fresh session."""
        from src.database.repositories.position import PositionRecordRepository

        if self._database_engine is None:
            msg = "Database engine not initialized"
            raise RuntimeError(msg)
        async with self._database_engine.session() as session:
            repository = PositionRecordRepository(session)
            await repository.update(position)

    def persist_position_delete(self, symbol: str) -> None:
        """Delete position from database."""
        if self._database_engine:
            try:
                task = asyncio.create_task(self._async_persist_position_delete(symbol))
                self._pending_tasks.add(task)
                task.add_done_callback(_make_task_cleanup_callback(self._pending_tasks))
            except Exception as e:
                logger.opt(exception=True).error(f"Failed to delete position from database: {e}")
                raise

    async def _async_persist_position_delete(self, symbol: str) -> None:
        """Async helper to delete position with fresh session."""
        from src.database.repositories.position import PositionRecordRepository

        if self._database_engine is None:
            msg = "Database engine not initialized"
            raise RuntimeError(msg)
        async with self._database_engine.session() as session:
            repository = PositionRecordRepository(session)
            await repository.delete_by_symbol(symbol)

    def persist_action(self, action: PositionManagementAction) -> None:
        """Persist action to database."""
        if self._database_engine:
            try:
                task = asyncio.create_task(self._async_persist_action(action))
                self._pending_tasks.add(task)
                task.add_done_callback(_make_task_cleanup_callback(self._pending_tasks))
                logger.debug(f"Persisted position action to database: {action.symbol} {action.action_type}")
            except Exception as e:
                logger.opt(exception=True).error(f"Failed to persist position action to database: {e}")
                raise

    async def _async_persist_action(self, action: PositionManagementAction) -> None:
        """Async helper to persist action with fresh session."""
        from src.database.repositories.position_action import PositionManagementActionRepository

        if self._database_engine is None:
            msg = "Database engine not initialized"
            raise RuntimeError(msg)
        async with self._database_engine.session() as session:
            repository = PositionManagementActionRepository(session)
            await repository.create(action)

    async def wait_for_pending_tasks(self, timeout_seconds: float = 5.0) -> None:
        """Wait for all pending background tasks to complete.

        Args:
            timeout_seconds: Maximum seconds to wait for tasks
        """
        if not self._pending_tasks:
            return

        logger.info(f"Waiting for {len(self._pending_tasks)} pending position persistence tasks...")
        pending = set(self._pending_tasks)
        _done, not_done = await asyncio.wait(pending, timeout=timeout_seconds)
        if not_done:
            logger.opt(exception=True).warning(
                f"Position persistence tasks timed out after {timeout_seconds}s, cancelling..."
            )
            for task in not_done:
                task.cancel()
            await asyncio.sleep(0.1)
            self._pending_tasks.clear()
        else:
            logger.info("All pending position persistence tasks completed")

    def __repr__(self) -> str:
        """Return string representation."""
        return f"PositionPersistenceManager(pending={len(self._pending_tasks)})"
