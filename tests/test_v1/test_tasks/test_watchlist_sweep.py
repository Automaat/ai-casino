"""Tests for WatchlistSweepTask."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.daemon.events import WatchlistStaleEvent
from src.v1.coordinator.models import SweepPassConfig
from src.v1.tasks.implementations.watchlist_sweep import WatchlistSweepTask
from src.v1.tasks.models import DedupStrategy


def _make_task(
    config: SweepPassConfig | None = None,
    watchlist: list[str] | None = None,
    timestamps: dict | None = None,
) -> tuple[WatchlistSweepTask, MagicMock, MagicMock]:
    """Create task with mock dependencies.

    Returns:
        Tuple of (task, mock_broker_manager, mock_queue)
    """
    queue = MagicMock()
    queue.enqueue = AsyncMock()

    broker_manager = MagicMock()
    broker_manager.get_merged_watchlist = AsyncMock(return_value=watchlist or [])

    db_engine = MagicMock()
    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)
    db_engine.session.return_value = mock_session

    task = WatchlistSweepTask(
        queue=queue,
        broker_manager=broker_manager,
        database_engine=db_engine,
        config=config or SweepPassConfig(enabled=True),
    )

    # Patch repository used inside _fetch_last_analysis_timestamps
    repo_mock = AsyncMock()
    repo_mock.get_last_analysis_timestamps = AsyncMock(return_value=timestamps or {})
    task._repo_mock = repo_mock

    return task, broker_manager, queue


class TestTaskMetadata:
    """Tests for task name and schedule."""

    @pytest.mark.unit
    def test_name(self) -> None:
        task, _, _ = _make_task()
        assert task.name == "watchlist_sweep"

    @pytest.mark.unit
    def test_schedule_uses_interval_dedup(self) -> None:
        config = SweepPassConfig(enabled=True, interval_minutes=90)
        task, _, _ = _make_task(config=config)
        schedule = task.schedule
        assert schedule.dedup == DedupStrategy.INTERVAL
        assert schedule.dedup_interval_minutes == 90
        assert schedule.enabled is True

    @pytest.mark.unit
    def test_schedule_disabled(self) -> None:
        config = SweepPassConfig(enabled=False)
        task, _, _ = _make_task(config=config)
        assert task.schedule.enabled is False

    @pytest.mark.unit
    def test_repr(self) -> None:
        task, _, _ = _make_task()
        r = repr(task)
        assert "WatchlistSweepTask" in r
        assert "enabled=True" in r


class TestExecuteEmptyWatchlist:
    """Tests for empty watchlist scenario."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_empty_watchlist_returns_success(self) -> None:
        task, _broker_manager, queue = _make_task(watchlist=[])
        result = await task.execute()
        assert result.success is True
        assert "Empty watchlist" in (result.message or "")
        queue.enqueue.assert_not_called()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_empty_watchlist_updates_last_run(self) -> None:
        task, _, _ = _make_task(watchlist=[])
        assert await task.last_run_at() is None
        await task.execute()
        assert await task.last_run_at() is not None


class TestExecuteNoStaleSymbols:
    """Tests for scenario where all symbols are fresh."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_fresh_symbols_returns_no_stale(self) -> None:
        now = datetime.now(UTC)
        fresh_ts = now - timedelta(hours=1)  # 1h ago, stale_hours=4 → fresh
        config = SweepPassConfig(enabled=True, stale_hours=4)
        timestamps = {"AAPL": fresh_ts.replace(tzinfo=None), "MSFT": fresh_ts.replace(tzinfo=None)}

        task, _, queue = _make_task(config=config, watchlist=["AAPL", "MSFT"], timestamps=timestamps)

        with patch(
            "src.database.repositories.analysis.AnalysisRecordRepository",
            return_value=MagicMock(get_last_analysis_timestamps=AsyncMock(return_value=timestamps)),
        ):
            result = await task.execute()

        assert result.success is True
        assert "No stale symbols" in (result.message or "")
        queue.enqueue.assert_not_called()


class TestExecuteNeverAnalyzedPrioritization:
    """Tests for never-analyzed symbol prioritization."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_never_analyzed_prioritized_first(self) -> None:
        now = datetime.now(UTC)
        stale_ts = now - timedelta(hours=6)  # stale (>4h)
        config = SweepPassConfig(enabled=True, stale_hours=4, max_symbols=10)
        # AAPL analyzed recently-stale, TSLA never analyzed
        timestamps = {"AAPL": stale_ts.replace(tzinfo=None)}

        task, _, queue = _make_task(config=config, watchlist=["AAPL", "TSLA"], timestamps=timestamps)

        with patch(
            "src.database.repositories.analysis.AnalysisRecordRepository",
            return_value=MagicMock(get_last_analysis_timestamps=AsyncMock(return_value=timestamps)),
        ):
            result = await task.execute()

        assert result.success is True
        queue.enqueue.assert_called_once()
        event: WatchlistStaleEvent = queue.enqueue.call_args.args[0]
        assert isinstance(event, WatchlistStaleEvent)
        symbols_in_order = [s.symbol for s in event.stale_symbols]
        # TSLA (never analyzed) must come before AAPL (stale)
        assert symbols_in_order.index("TSLA") < symbols_in_order.index("AAPL")

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_never_analyzed_has_none_age(self) -> None:
        config = SweepPassConfig(enabled=True, stale_hours=4)
        timestamps: dict = {}  # AAPL never analyzed

        task, _, queue = _make_task(config=config, watchlist=["AAPL"], timestamps=timestamps)

        with patch(
            "src.database.repositories.analysis.AnalysisRecordRepository",
            return_value=MagicMock(get_last_analysis_timestamps=AsyncMock(return_value=timestamps)),
        ):
            await task.execute()

        event: WatchlistStaleEvent = queue.enqueue.call_args.args[0]
        aapl_info = next(s for s in event.stale_symbols if s.symbol == "AAPL")
        assert aapl_info.last_analysis_age_hours is None


class TestExecuteStaleOldestFirst:
    """Tests for oldest-first ordering of stale symbols."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_stale_symbols_ordered_oldest_first(self) -> None:
        now = datetime.now(UTC)
        config = SweepPassConfig(enabled=True, stale_hours=4, max_symbols=10)
        timestamps = {
            "AAPL": (now - timedelta(hours=8)).replace(tzinfo=None),  # older
            "MSFT": (now - timedelta(hours=5)).replace(tzinfo=None),  # newer stale
            "TSLA": (now - timedelta(hours=12)).replace(tzinfo=None),  # oldest
        }

        task, _, queue = _make_task(config=config, watchlist=["AAPL", "MSFT", "TSLA"], timestamps=timestamps)

        with patch(
            "src.database.repositories.analysis.AnalysisRecordRepository",
            return_value=MagicMock(get_last_analysis_timestamps=AsyncMock(return_value=timestamps)),
        ):
            await task.execute()

        event: WatchlistStaleEvent = queue.enqueue.call_args.args[0]
        symbols_in_order = [s.symbol for s in event.stale_symbols]
        assert symbols_in_order == ["TSLA", "AAPL", "MSFT"]


class TestExecuteMaxSymbolsLimit:
    """Tests for max_symbols cap."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_max_symbols_limits_enqueued_count(self) -> None:
        now = datetime.now(UTC)
        config = SweepPassConfig(enabled=True, stale_hours=4, max_symbols=2)
        timestamps = {
            "AAPL": (now - timedelta(hours=6)).replace(tzinfo=None),
            "MSFT": (now - timedelta(hours=7)).replace(tzinfo=None),
            "TSLA": (now - timedelta(hours=8)).replace(tzinfo=None),
        }

        task, _, queue = _make_task(config=config, watchlist=["AAPL", "MSFT", "TSLA"], timestamps=timestamps)

        with patch(
            "src.database.repositories.analysis.AnalysisRecordRepository",
            return_value=MagicMock(get_last_analysis_timestamps=AsyncMock(return_value=timestamps)),
        ):
            await task.execute()

        event: WatchlistStaleEvent = queue.enqueue.call_args.args[0]
        assert len(event.stale_symbols) == 2


class TestExecuteDbFailure:
    """Tests for database failure scenarios."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_db_failure_propagates_as_failed_task_result(self) -> None:
        task, _, queue = _make_task(watchlist=["AAPL"])

        with patch(
            "src.database.repositories.analysis.AnalysisRecordRepository",
            side_effect=RuntimeError("DB unavailable"),
        ):
            result = await task.execute()

        assert result.success is False
        assert "DB unavailable" in (result.message or "")
        queue.enqueue.assert_not_called()
