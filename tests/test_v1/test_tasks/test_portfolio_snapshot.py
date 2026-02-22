"""Tests for PortfolioSnapshotTask."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest
from result import Ok

from src.daemon.config.portfolio import PortfolioSnapshotConfig
from src.v1.tasks.implementations.portfolio_snapshot import PortfolioSnapshotTask
from src.v1.tasks.models import DedupStrategy


def _make_config(enabled: bool = True, interval_minutes: int = 10) -> PortfolioSnapshotConfig:
    return PortfolioSnapshotConfig(enabled=enabled, interval_minutes=interval_minutes)


def _make_account_info() -> Mock:
    account = Mock()
    account.balance = 100_000.0
    account.available_cash = 50_000.0
    account.total_exposure = 50_000.0
    account.portfolio_value = 100_000.0
    account.positions = {}
    return account


def _make_db_engine() -> AsyncMock:
    session = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    engine = AsyncMock()
    engine.session = Mock(return_value=session)
    return engine


class TestSchedule:
    """Tests for PortfolioSnapshotTask.schedule property."""

    @pytest.mark.unit
    def test_schedule_uses_interval_dedup(self) -> None:
        """Schedule uses INTERVAL strategy with no time field."""
        broker = Mock()
        engine = AsyncMock()
        config = _make_config(interval_minutes=15)
        task = PortfolioSnapshotTask(broker=broker, database_engine=engine, config=config)

        schedule = task.schedule

        assert schedule.dedup == DedupStrategy.INTERVAL
        assert schedule.dedup_interval_minutes == 15
        assert schedule.time is None

    @pytest.mark.unit
    def test_schedule_reflects_enabled_config(self) -> None:
        """Schedule enabled flag mirrors config."""
        broker = Mock()
        engine = AsyncMock()

        task_on = PortfolioSnapshotTask(
            broker=broker, database_engine=engine, config=_make_config(enabled=True)
        )
        task_off = PortfolioSnapshotTask(
            broker=broker, database_engine=engine, config=_make_config(enabled=False)
        )

        assert task_on.schedule.enabled is True
        assert task_off.schedule.enabled is False


class TestExecute:
    """Tests for PortfolioSnapshotTask.execute()."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_execute_creates_snapshot(self) -> None:
        """Snapshot created with correct trigger and account fields."""
        account = _make_account_info()
        broker = Mock()
        broker.get_account_info = Mock(return_value=Ok(account))
        engine = _make_db_engine()
        config = _make_config()
        task = PortfolioSnapshotTask(broker=broker, database_engine=engine, config=config)

        with patch("src.v1.tasks.implementations.portfolio_snapshot.PortfolioSnapshotRepository") as repo_cls:
            repo = AsyncMock()
            repo_cls.return_value = repo

            result = await task.execute()

        assert result.success is True
        assert result.task_name == "portfolio_snapshot"
        repo.create.assert_called_once()
        snapshot_arg = repo.create.call_args[0][0]
        assert snapshot_arg.trigger == "SCHEDULED"
        assert snapshot_arg.balance == 100_000.0
        assert snapshot_arg.portfolio_value == 100_000.0

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_execute_returns_duration(self) -> None:
        """TaskResult includes a non-negative duration."""
        account = _make_account_info()
        broker = Mock()
        broker.get_account_info = Mock(return_value=Ok(account))
        engine = _make_db_engine()
        task = PortfolioSnapshotTask(broker=broker, database_engine=engine, config=_make_config())

        with patch("src.v1.tasks.implementations.portfolio_snapshot.PortfolioSnapshotRepository") as repo_cls:
            repo_cls.return_value = AsyncMock()
            result = await task.execute()

        assert result.duration_seconds >= 0.0

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_execute_broker_error_propagates(self) -> None:
        """Broker failure propagates (critical operation)."""
        broker = Mock()
        broker.get_account_info = Mock(side_effect=RuntimeError("broker down"))
        engine = _make_db_engine()
        task = PortfolioSnapshotTask(broker=broker, database_engine=engine, config=_make_config())

        with pytest.raises(RuntimeError, match="broker down"):
            await task.execute()


class TestLastRunAt:
    """Tests for PortfolioSnapshotTask.last_run_at()."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_returns_none_when_no_snapshots(self) -> None:
        """Returns None when no snapshots exist."""
        broker = Mock()
        engine = _make_db_engine()
        task = PortfolioSnapshotTask(broker=broker, database_engine=engine, config=_make_config())

        with patch("src.v1.tasks.implementations.portfolio_snapshot.PortfolioSnapshotRepository") as repo_cls:
            repo = AsyncMock()
            repo.get_latest = AsyncMock(return_value=None)
            repo_cls.return_value = repo

            result = await task.last_run_at()

        assert result is None

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_returns_latest_timestamp(self) -> None:
        """Returns timestamp from latest snapshot."""
        broker = Mock()
        engine = _make_db_engine()
        task = PortfolioSnapshotTask(broker=broker, database_engine=engine, config=_make_config())
        ts = datetime(2026, 2, 18, 10, 0, tzinfo=UTC)

        with patch("src.v1.tasks.implementations.portfolio_snapshot.PortfolioSnapshotRepository") as repo_cls:
            snapshot = Mock()
            snapshot.timestamp = ts
            repo = AsyncMock()
            repo.get_latest = AsyncMock(return_value=snapshot)
            repo_cls.return_value = repo

            result = await task.last_run_at()

        assert result == ts
