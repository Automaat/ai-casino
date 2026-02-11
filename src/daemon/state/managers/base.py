"""Base state manager infrastructure."""

from __future__ import annotations

import asyncio
from typing import TypeVar

from loguru import logger
from pydantic import BaseModel

T = TypeVar("T")


def _log_task_exception(task: asyncio.Task[object]) -> None:
    """Log exceptions from fire-and-forget tasks."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.opt(exception=exc).error("Background task failed")


class StateManager(BaseModel):
    """Base class for domain state managers."""

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

    def __repr__(self) -> str:
        """Return string representation."""
        return f"{self.__class__.__name__}()"
