"""Scheduled task dispatcher for daemon."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from loguru import logger

if TYPE_CHECKING:
    from src.daemon.config import DaemonConfig
    from src.daemon.runner import DaemonRunner
    from src.daemon.scheduler import MarketScheduler
    from src.daemon.task_service import DaemonTaskService


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
        ScheduledTask("prefetch", "is_prefetch_time", "_run_prefetch", "prefetch.enabled"),
        ScheduledTask(
            "pre_market_refresh",
            "is_pre_market_refresh_time",
            "_run_pre_market_refresh",
            "prefetch.enabled",
        ),
        ScheduledTask("earnings_fetch", "is_earnings_fetch_time", "_run_earnings_fetch"),
        ScheduledTask("sector_rotation", "is_sector_rotation_time", "_run_sector_rotation"),
        ScheduledTask(
            "portfolio_rebalancing",
            "is_portfolio_rebalancing_time",
            "_run_portfolio_rebalancing",
        ),
        ScheduledTask("peer_analysis", "is_peer_analysis_time", "_run_peer_analysis"),
        ScheduledTask("correlation_audit", "is_correlation_audit_time", "_run_correlation_audit"),
        ScheduledTask(
            "monte_carlo",
            "is_monte_carlo_time",
            "_run_monte_carlo_stress_testing",
            "monte_carlo.enabled",
        ),
        ScheduledTask(
            "after_hours_screening",
            "is_after_hours_screening_time",
            "_run_after_hours_screening",
        ),
        ScheduledTask("optimization", "is_optimization_time", "_run_optimization", "optimization.enabled"),
        ScheduledTask("signal_tracking", "is_signal_tracking_time", "_run_signal_tracking"),
    ]

    def __init__(
        self,
        config: "DaemonConfig",
        scheduler: "MarketScheduler",
        daemon_runner: "DaemonRunner",
    ) -> None:
        """Initialize task runner.

        Args:
            config: Daemon configuration
            scheduler: Market scheduler for time checks
            daemon_runner: DaemonRunner instance for task delegation
        task_service: Optional task service for extracted scheduled tasks
        """
        self.config = config
        self.scheduler = scheduler
        self._runner = daemon_runner
        self._task_service: DaemonTaskService | None = None  # Wired later via set_task_service
        logger.info("ScheduledTaskRunner initialized")

    def set_task_service(self, task_service: "DaemonTaskService") -> None:
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
            # Runner: _run_optimization -> Task Service: run_optimization
            # Runner: _maybe_run_discovery -> Task Service: run_discovery
            service_method_name = task.runner_method.lstrip("_").replace("maybe_run_", "run_")

            # Check if task_service has this method (extracted tasks)
            if self._task_service and hasattr(self._task_service, service_method_name):
                task_method = getattr(self._task_service, service_method_name)
            else:
                # Runner method (not yet extracted)
                task_method = getattr(self._runner, task.runner_method)

            if task.is_async:
                await task_method()
            else:
                task_method()

        # Special case: daily risk report (runs when market closed)
        if not self.scheduler.is_market_open() and self._task_service:
            self._task_service.run_daily_risk_report()

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
