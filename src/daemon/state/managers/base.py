"""Base state manager infrastructure."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any, TypeVar

from loguru import logger
from pydantic import BaseModel, PrivateAttr

T = TypeVar("T")


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


def _log_task_exception(task: asyncio.Task[object]) -> None:
    """Log exceptions from fire-and-forget tasks."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.opt(exception=exc).error("Background task failed")


class StateManager(BaseModel):
    """Base class for domain state managers."""

    _pending_tasks: set[asyncio.Task[Any]] = PrivateAttr(default_factory=set)
    _event_loop_id: int | None = PrivateAttr(default=None)

    def _is_different_event_loop(self) -> bool:
        """Check if current event loop differs from stored repositories' loop.

        Returns:
            True if repositories were created in different event loop
        """
        try:
            current_loop = asyncio.get_running_loop()
            current_id = id(current_loop)

            if self._event_loop_id is None:
                self._event_loop_id = current_id
                return False

            return self._event_loop_id != current_id
        except RuntimeError:
            # No running event loop
            return False

    def _cap_history(self, history: list[T], max_size: int, keep_size: int) -> list[T]:
        """Cap history when exceeding max_size.

        Args:
            history: History list to cap
            max_size: Maximum size before capping
            keep_size: Number of entries to keep after capping

        Returns:
            Capped history list
        """
        if len(history) > max_size:
            return history[-keep_size:]
        return history

    async def wait_for_pending_tasks(self, timeout_seconds: float = 5.0) -> None:
        """Wait for all pending background tasks to complete.

        Args:
            timeout_seconds: Maximum seconds to wait for tasks

        Called during daemon shutdown to ensure database operations complete cleanly.
        """
        if not self._pending_tasks:
            return

        logger.info(f"Waiting for {len(self._pending_tasks)} pending {self.__class__.__name__} tasks...")
        try:
            await asyncio.wait_for(
                asyncio.gather(*self._pending_tasks, return_exceptions=True),  # type: ignore[bad-argument-type]
                timeout_seconds,
            )
            logger.info(f"All pending {self.__class__.__name__} tasks completed")
        except TimeoutError:
            logger.opt(exception=True).warning(
                f"{self.__class__.__name__} tasks timed out after {timeout_seconds}s, cancelling..."
            )
            for task in self._pending_tasks:
                if not task.done():
                    task.cancel()
            # Wait briefly for cancellations to propagate
            await asyncio.sleep(0.1)
            self._pending_tasks.clear()

    def __repr__(self) -> str:
        """Return string representation."""
        return f"{self.__class__.__name__}()"
