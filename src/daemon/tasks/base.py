"""Task execution framework with boilerplate handling."""

from __future__ import annotations

import time as time_mod
from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING

from loguru import logger
from rich.console import Console

if TYPE_CHECKING:
    from src.daemon.factory import DaemonComponents
    from src.di.container import AppContainer

console = Console()


class TaskExecutor(ABC):
    """Base class for daemon tasks with shared boilerplate.

    Handles:
    - Deduplication check (ran today?)
    - Logging (start/complete/error)
    - Console output (header/footer)
    - State recording + persistence
    - Error handling with error recording

    Subclasses implement:
    - task_name: Human-readable name
    - get_last_run(): State field for dedup
    - execute(): Core business logic
    - record_success(): State update
    """

    def __init__(self, components: DaemonComponents, container: AppContainer) -> None:
        """Initialize task executor.

        Args:
            components: Daemon components
            container: DI container for service access
        """
        self.components = components
        self.container = container

    @property
    @abstractmethod
    def task_name(self) -> str:
        """Task display name."""
        ...

    @abstractmethod
    async def execute(self) -> None:
        """Execute core task logic."""
        ...

    @abstractmethod
    def get_last_run(self) -> datetime | None:
        """Get last run timestamp for dedup."""
        ...

    @abstractmethod
    def record_success(self, duration: float) -> None:
        """Record successful execution in state."""
        ...

    def should_skip_today(self) -> bool:
        """Check if task already ran today.

        Returns:
            True if task already completed today
        """
        last_run = self.get_last_run()
        if not last_run:
            return False

        now = datetime.now(self.components.scheduler.timezone)
        last_date = last_run.astimezone(self.components.scheduler.timezone).date()
        return last_date == now.date()

    async def run(self) -> None:
        """Run task with full boilerplate.

        Handles:
        - Dedup check
        - Logging + console output
        - Error handling
        - State persistence
        """
        if self.should_skip_today():
            logger.debug(f"{self.task_name} already completed today")
            return

        now = datetime.now(self.components.scheduler.timezone)
        logger.info(f"Starting {self.task_name}")
        console.print(f"\n[bold cyan]{self.task_name} ({now:%H:%M})[/bold cyan]")
        console.print("-" * 50)

        try:
            start_time = time_mod.time()
            await self.execute()
            duration = time_mod.time() - start_time

            self.record_success(duration)
            self.components.state.save(self.components.config.state.state_file)

            console.print(f"\n[dim]Complete ({duration:.0f}s)[/dim]\n")
            logger.info(f"{self.task_name} completed in {duration:.1f}s")
        except Exception as e:
            error_msg = f"{self.task_name} failed: {e}"
            logger.opt(exception=True).error(error_msg)
            self.components.state.record_error(error_msg)
            raise
