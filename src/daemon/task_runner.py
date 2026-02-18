"""Scheduled task dispatcher for daemon."""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from loguru import logger

from src.daemon.config import DaemonConfig
from src.daemon.scheduler import MarketScheduler

if TYPE_CHECKING:
    from src.daemon.runner import DaemonRunner
    from src.daemon.task_service import DaemonTaskService

# Tasks that run as background asyncio.Tasks (non-blocking)
_BACKGROUND_TASKS = frozenset({"reddit_scraping"})


@dataclass
class ScheduledTask:
    """Metadata for scheduled task."""

    name: str
    check_method: str  # Scheduler method (e.g., "is_optimization_time")
    runner_method: str  # Runner method (e.g., "_run_optimization")
    enabled_check: str | None = None  # Config path (e.g., "optimization.enabled")
    is_async: bool = False


class ScheduledTaskRunner:
    """Run scheduled after-hours/pre-market tasks."""

    # Task registry (explicit, not dynamic)
    TASKS: ClassVar[list[ScheduledTask]] = [
        ScheduledTask("game_plan", "is_game_plan_time", "_run_game_plan", is_async=True),
        ScheduledTask("prefetch", "is_prefetch_time", "_run_prefetch", "prefetch.enabled", is_async=True),
        ScheduledTask(
            "pre_market_refresh",
            "is_pre_market_refresh_time",
            "_run_pre_market_refresh",
            "prefetch.enabled",
            is_async=True,
        ),
        ScheduledTask("earnings_fetch", "is_earnings_fetch_time", "_run_earnings_fetch", is_async=True),
        ScheduledTask("sector_rotation", "is_sector_rotation_time", "_run_sector_rotation", is_async=True),
        ScheduledTask(
            "portfolio_rebalancing",
            "is_portfolio_rebalancing_time",
            "_run_portfolio_rebalancing",
        ),
        ScheduledTask("peer_analysis", "is_peer_analysis_time", "_run_peer_analysis", is_async=True),
        ScheduledTask("correlation_audit", "is_correlation_audit_time", "_run_correlation_audit"),
        ScheduledTask(
            "portfolio_health",
            "is_portfolio_health_time",
            "_run_portfolio_health",
            "portfolio_health.enabled",
            is_async=True,
        ),
        ScheduledTask(
            "monte_carlo",
            "is_monte_carlo_time",
            "_run_monte_carlo_stress_testing",
            "monte_carlo.enabled",
        ),
        ScheduledTask("optimization", "is_optimization_time", "_run_optimization", "optimization.enabled"),
        ScheduledTask("signal_tracking", "is_signal_tracking_time", "_run_signal_tracking"),
        ScheduledTask(
            "discovery_outcome",
            "is_discovery_outcome_time",
            "_run_discovery_outcome",
            "discovery.enabled",
            is_async=True,
        ),
        ScheduledTask(
            "reddit_scraping",
            "is_reddit_scraping_time",
            "_run_reddit_scraping",
            "reddit_scraper.enabled",
            is_async=True,
        ),
    ]

    def __init__(
        self,
        config: DaemonConfig,
        scheduler: MarketScheduler,
        daemon_runner: DaemonRunner,
    ) -> None:
        """Initialize task runner.

        Args:
            config: Daemon configuration
            scheduler: Market scheduler for time checks
            daemon_runner: DaemonRunner instance for task delegation
        """
        self.config = config
        self.scheduler = scheduler
        self._runner = daemon_runner
        self._task_service: DaemonTaskService | None = None  # Wired later via set_task_service
        self._background_tasks: dict[str, asyncio.Task[None]] = {}
        logger.info("ScheduledTaskRunner initialized")

    def __repr__(self) -> str:
        """Return string representation."""
        return f"ScheduledTaskRunner(tasks={len(self.TASKS)})"

    def set_task_service(self, task_service: DaemonTaskService) -> None:
        """Wire task service after initialization.

        Args:
            task_service: DaemonTaskService instance
        """
        self._task_service = task_service

    async def run_scheduled_tasks(self) -> None:
        """Run all scheduled tasks based on time and config."""
        # Early return if runner not yet wired (during initialization)
        if self._runner is None:
            logger.debug("Task runner not yet wired, skipping scheduled tasks")
            return

        # Prune completed background tasks
        self._background_tasks = {name: t for name, t in self._background_tasks.items() if not t.done()}

        for task in self.TASKS:
            # Check if task is enabled (if it has an enabled check)
            if task.enabled_check and not self._is_task_enabled(task.enabled_check):
                continue

            # Check if it's time to run this task
            if not getattr(self.scheduler, task.check_method)():
                continue

            # Execute the task
            logger.debug(f"Running scheduled task: {task.name}")

            # Map runner method names to task service method names
            service_method_name = task.runner_method.lstrip("_").replace("maybe_run_", "run_")

            # Check if task_service has this method (extracted tasks)
            if self._task_service and hasattr(self._task_service, service_method_name):
                task_method = getattr(self._task_service, service_method_name)
            else:
                task_method = getattr(self._runner, task.runner_method)

            # Background tasks: fire-and-forget with overlap guard
            if task.name in _BACKGROUND_TASKS and task.is_async:
                if task.name in self._background_tasks:
                    logger.debug(f"Background task {task.name} already running, skipping")
                    continue
                bg_task = asyncio.create_task(
                    self._run_background_task(task.name, task_method),
                    name=f"bg-{task.name}",
                )
                self._background_tasks[task.name] = bg_task
                continue

            if task.is_async:
                await task_method()
            else:
                task_method()

        # Special case: daily risk report (runs when market closed)
        if not self.scheduler.is_market_open() and self._task_service:
            await self._task_service.run_daily_risk_report()

    @staticmethod
    async def _run_background_task(name: str, method: Callable[[], Awaitable[None]]) -> None:
        """Run a task in the background with error handling.

        Args:
            name: Task name for logging
            method: Async callable to execute
        """
        try:
            logger.info(f"Background task started: {name}")
            await method()
            logger.info(f"Background task completed: {name}")
        except Exception as e:
            logger.opt(exception=True).error(f"Background task {name} failed: {e}")

    async def stop_background_tasks(self, wait_seconds: float = 30.0) -> None:
        """Cancel and await all running background tasks.

        Args:
            wait_seconds: Max seconds to wait before cancelling
        """
        active = {name: t for name, t in self._background_tasks.items() if not t.done()}
        if not active:
            return

        logger.info(f"Waiting for {len(active)} background task(s) to complete: {list(active)}")
        _done, pending = await asyncio.wait(list(active.values()), timeout=wait_seconds)
        if pending:
            logger.warning(f"{len(pending)} background task(s) did not finish in time, cancelling")
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

        self._background_tasks.clear()

    def _is_task_enabled(self, config_path: str) -> bool:
        """Check if task enabled via config path.

        Args:
            config_path: Dotted path to config field (e.g., "optimization.enabled")

        Returns:
            True if task is enabled
        """
        obj = self.config
        for attr in config_path.split("."):
            obj = getattr(obj, attr)
        return bool(obj)
