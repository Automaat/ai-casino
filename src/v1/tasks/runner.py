"""Task runner — autonomous loop that evaluates schedules and executes due tasks."""

import asyncio
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from loguru import logger

from src.v1.tasks.interface import Task
from src.v1.tasks.models import TaskResult
from src.v1.tasks.scheduling import is_due

_CHECK_INTERVAL_SECONDS = 60


class TaskRunner:
    """Autonomous scheduled task runner — checks every 60s, independent of daemon cycles."""

    def __init__(self, tasks: list[Task], timezone: ZoneInfo) -> None:
        """Initialize runner.

        Args:
            tasks: List of tasks to manage
            timezone: Timezone for schedule evaluation
        """
        self._tasks = {t.name: t for t in tasks}
        self._tz = timezone
        logger.info(f"TaskRunner initialized with {len(tasks)} tasks: {list(self._tasks)}")

    async def run(self) -> None:
        """Run autonomous task loop. Checks every 60s."""
        logger.info(f"TaskRunner started: {list(self._tasks)}")
        try:
            while True:
                await self._check_and_run()
                await asyncio.sleep(_CHECK_INTERVAL_SECONDS)
        except asyncio.CancelledError:
            logger.info("TaskRunner cancelled, shutting down task loop")
            raise

    async def _check_and_run(self) -> list[TaskResult]:
        """Check all tasks and run those that are due.

        Returns:
            List of results from executed tasks
        """
        now = datetime.now(self._tz)
        results: list[TaskResult] = []

        for task in self._tasks.values():
            last = await task.last_run_at()
            due = is_due(task.schedule, last, now)
            last_str = f"{last:%Y-%m-%d %H:%M}" if last else "never"
            logger.debug(
                f"Task {task.name}: due={due}, scheduled={task.schedule.time}, "
                f"last_run={last_str}, now={now:%H:%M}"
            )
            if due:
                result = await self._run_task(task)
                results.append(result)

        if results:
            ok = sum(1 for r in results if r.success)
            fail = len(results) - ok
            logger.info(f"TaskRunner tick: {len(results)} tasks ran ({ok} OK, {fail} failed)")

        return results

    async def _run_task(self, task: Task) -> TaskResult:
        """Execute a single task with error handling.

        Args:
            task: Task to execute

        Returns:
            TaskResult (success or failure)
        """
        logger.info(f"Task {task.name} starting (schedule: {task.schedule.time} {task.schedule.days})")
        start = time.monotonic()

        try:
            result = await task.execute()
            status = "OK" if result.success else "FAIL"
            msg = f" — {result.message}" if result.message else ""
            logger.info(f"Task {task.name} finished: {status} in {result.duration_seconds:.1f}s{msg}")
            return result
        except Exception as e:
            duration = time.monotonic() - start
            logger.opt(exception=True).error(f"Task {task.name} crashed after {duration:.1f}s: {e}")
            return TaskResult(
                task_name=task.name,
                success=False,
                duration_seconds=duration,
                message=f"Error: {e!s}",
            )

    def __repr__(self) -> str:
        """String representation."""
        return f"TaskRunner(tasks={list(self._tasks)})"
