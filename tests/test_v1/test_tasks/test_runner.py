"""Tests for v1 TaskRunner."""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, PropertyMock, patch
from zoneinfo import ZoneInfo

import pytest

from src.v1.tasks.interface import Task
from src.v1.tasks.models import WEEKDAYS, DedupStrategy, TaskResult, TaskSchedule
from src.v1.tasks.runner import TaskRunner

ET = ZoneInfo("America/New_York")


def _make_task(
    name: str = "test_task",
    time: str = "04:00",
    enabled: bool = True,
    dedup: DedupStrategy = DedupStrategy.DAILY,
    last_run: datetime | None = None,
    execute_result: TaskResult | None = None,
) -> Task:
    """Create a mock task."""
    task = AsyncMock(spec=Task)
    type(task).name = PropertyMock(return_value=name)
    type(task).schedule = PropertyMock(
        return_value=TaskSchedule(time=time, days=WEEKDAYS, enabled=enabled, dedup=dedup)
    )
    task.last_run_at.return_value = last_run
    if execute_result is None:
        execute_result = TaskResult(task_name=name, success=True, duration_seconds=1.0)
    task.execute.return_value = execute_result
    return task


class TestCheckAndRun:
    """Tests for TaskRunner._check_and_run()."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_runs_due_task(self) -> None:
        """Runs a task that is due."""
        task = _make_task(time="04:00")
        runner = TaskRunner([task], ET)

        with patch("src.v1.tasks.runner.is_due", return_value=True):
            results = await runner._check_and_run()

        assert len(results) == 1
        assert results[0].success is True
        task.execute.assert_awaited_once()

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_skips_not_due_task(self) -> None:
        """Skips tasks that are not due."""
        task = _make_task(enabled=False)
        runner = TaskRunner([task], ET)

        with patch("src.v1.tasks.runner.is_due", return_value=False):
            results = await runner._check_and_run()

        assert len(results) == 0
        task.execute.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_skips_already_ran_today(self) -> None:
        """Skips task that already ran today (DAILY dedup)."""
        last = datetime(2026, 2, 18, 4, 0, tzinfo=ET)
        task = _make_task(time="04:00", last_run=last)
        runner = TaskRunner([task], ET)

        with patch("src.v1.tasks.runner.is_due", return_value=False):
            results = await runner._check_and_run()

        assert len(results) == 0

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_handles_task_exception(self) -> None:
        """Catches exceptions and returns failure result."""
        task = _make_task()
        task.execute.side_effect = RuntimeError("boom")
        runner = TaskRunner([task], ET)

        with patch("src.v1.tasks.runner.is_due", return_value=True):
            results = await runner._check_and_run()

        assert len(results) == 1
        assert results[0].success is False
        assert "boom" in results[0].message

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_multiple_tasks(self) -> None:
        """Runs multiple due tasks."""
        t1 = _make_task(name="task_a")
        t2 = _make_task(name="task_b")
        runner = TaskRunner([t1, t2], ET)

        with patch("src.v1.tasks.runner.is_due", return_value=True):
            results = await runner._check_and_run()

        assert len(results) == 2


class TestRunLoop:
    """Tests for TaskRunner.run() autonomous loop."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_run_loops_and_cancels(self) -> None:
        """run() loops, executes tasks, and stops on cancellation."""
        task = _make_task()
        runner = TaskRunner([task], ET)
        call_count = 0

        async def mock_check_and_run() -> list[TaskResult]:
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                raise asyncio.CancelledError
            return [TaskResult(task_name="test_task", success=True, duration_seconds=0.1)]

        with patch.object(runner, "_check_and_run", side_effect=mock_check_and_run):
            with patch("src.v1.tasks.runner.asyncio.sleep", new_callable=AsyncMock):
                with pytest.raises(asyncio.CancelledError):
                    await runner.run()

        assert call_count >= 2
