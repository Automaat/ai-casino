"""Watcher base classes."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod

from loguru import logger


class Watcher(ABC):
    """Abstract base class for all market watchers."""

    running: bool

    @property
    @abstractmethod
    def name(self) -> str:
        """Watcher display name."""
        ...

    @abstractmethod
    async def run(self) -> None:
        """Run the watcher until stopped."""
        ...

    def stop(self) -> None:
        """Signal the watcher to stop."""
        self.running = False


class PeriodicWatcher(Watcher):
    """Watcher that polls on a fixed interval with 1s-granularity sleep for responsive shutdown."""

    def __init__(self, poll_interval: int) -> None:
        """Initialize periodic watcher.

        Args:
            poll_interval: Seconds between polls
        """
        self.poll_interval = poll_interval
        self.running = False

    @abstractmethod
    async def _tick(self) -> None:
        """Execute one poll cycle."""
        ...

    async def run(self) -> None:
        """Run poll loop until stopped."""
        self.running = True
        logger.info(f"{self.name} started (interval={self.poll_interval}s)")
        while self.running:
            try:
                await self._tick()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.opt(exception=True).error(f"{self.name} cycle failed: {e}")
            remaining = float(self.poll_interval)
            while remaining > 0 and self.running:
                await asyncio.sleep(min(1.0, remaining))
                remaining -= 1.0
        logger.info(f"{self.name} stopped")
