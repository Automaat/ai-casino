"""Tests for SignalTrackingTask."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.daemon.config.reporting import SignalTrackingConfig
from src.v1.tasks.implementations.signal_tracking import SignalTrackingTask
from src.v1.tasks.models import DedupStrategy


def _make_task(
    enabled: bool = True,
    db_last_run: object = None,
    tracker_stats: dict | None = None,
    tracker_raises: Exception | None = None,
) -> tuple[SignalTrackingTask, AsyncMock, AsyncMock]:
    """Create task with mock dependencies.

    Returns:
        Tuple of (task, mock_state, mock_set_last_signal_tracking)
    """
    cache = MagicMock()
    market_fetcher = MagicMock()

    state = MagicMock()
    state.get_last_signal_tracking = AsyncMock(return_value=db_last_run)
    state.set_last_signal_tracking = AsyncMock()

    config = SignalTrackingConfig(enabled=enabled, tracking_time="17:00")

    task = SignalTrackingTask(
        historical_cache=cache,
        market_fetcher=market_fetcher,
        state=state,
        config=config,
    )

    mock_tracker = MagicMock()
    if tracker_raises:
        mock_tracker.update_outcomes.side_effect = tracker_raises
    else:
        mock_tracker.update_outcomes.return_value = tracker_stats or {"updated_1d": 2, "updated_5d": 1}

    return task, state, mock_tracker


class TestTaskMetadata:
    """Tests for task name and schedule."""

    @pytest.mark.unit
    def test_name(self) -> None:
        task, _, _ = _make_task()
        assert task.name == "signal_tracking"

    @pytest.mark.unit
    def test_schedule_uses_daily_dedup(self) -> None:
        task, _, _ = _make_task()
        schedule = task.schedule
        assert schedule.dedup == DedupStrategy.DAILY
        assert schedule.time == "17:00"
        assert schedule.enabled is True

    @pytest.mark.unit
    def test_schedule_disabled(self) -> None:
        task, _, _ = _make_task(enabled=False)
        assert task.schedule.enabled is False

    @pytest.mark.unit
    def test_repr(self) -> None:
        task, _, _ = _make_task()
        r = repr(task)
        assert "SignalTrackingTask" in r
        assert "enabled=True" in r


class TestExecuteSuccess:
    """Tests for successful execution."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_success_returns_task_result(self) -> None:
        task, _state, mock_tracker = _make_task(tracker_stats={"updated_1d": 3})

        with patch("src.daemon.signal_tracker.SignalOutcomeTracker", return_value=mock_tracker):
            result = await task.execute()

        assert result.success is True
        assert result.task_name == "signal_tracking"
        assert result.duration_seconds >= 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_success_updates_in_memory_last_run(self) -> None:
        task, _state, mock_tracker = _make_task()

        assert task._last_run is None

        with patch("src.daemon.signal_tracker.SignalOutcomeTracker", return_value=mock_tracker):
            await task.execute()

        assert task._last_run is not None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_success_persists_to_state(self) -> None:
        task, state, mock_tracker = _make_task()

        with patch("src.daemon.signal_tracker.SignalOutcomeTracker", return_value=mock_tracker):
            await task.execute()

        state.set_last_signal_tracking.assert_called_once()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_success_message_contains_stats(self) -> None:
        task, _state, mock_tracker = _make_task(tracker_stats={"updated_1d": 5, "updated_5d": 2})

        with patch("src.daemon.signal_tracker.SignalOutcomeTracker", return_value=mock_tracker):
            result = await task.execute()

        assert "updated=7" in (result.message or "")


class TestExecuteFailure:
    """Tests for failed execution."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_failure_returns_failed_task_result(self) -> None:
        task, _state, mock_tracker = _make_task(tracker_raises=RuntimeError("fetch failed"))

        with patch("src.daemon.signal_tracker.SignalOutcomeTracker", return_value=mock_tracker):
            result = await task.execute()

        assert result.success is False
        assert "fetch failed" in (result.message or "")

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_failure_does_not_update_last_run(self) -> None:
        task, state, mock_tracker = _make_task(tracker_raises=RuntimeError("error"))

        assert task._last_run is None

        with patch("src.daemon.signal_tracker.SignalOutcomeTracker", return_value=mock_tracker):
            await task.execute()

        assert task._last_run is None
        state.set_last_signal_tracking.assert_not_called()


class TestLastRunFallback:
    """Tests for in-memory fallback when DB is disabled."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_last_run_at_uses_db_when_available(self) -> None:
        from datetime import UTC, datetime

        db_ts = datetime(2025, 1, 1, 17, 0, tzinfo=UTC)
        task, _state, _ = _make_task(db_last_run=db_ts)

        result = await task.last_run_at()
        assert result == db_ts

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_last_run_at_falls_back_to_in_memory(self) -> None:
        from datetime import UTC, datetime

        task, state, mock_tracker = _make_task(db_last_run=None)

        assert await task.last_run_at() is None

        with patch("src.daemon.signal_tracker.SignalOutcomeTracker", return_value=mock_tracker):
            await task.execute()

        # DB still returns None (disabled), but in-memory fallback is used
        state.get_last_signal_tracking.return_value = None
        result = await task.last_run_at()
        assert result is not None
        assert isinstance(result, datetime)
        assert result.tzinfo == UTC
