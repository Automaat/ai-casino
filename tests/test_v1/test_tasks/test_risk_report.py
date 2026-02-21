"""Tests for RiskReportTask."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.risk.models import PortfolioRiskReport
from src.daemon.config.risk import RiskLimitsConfig
from src.daemon.events import RiskReportEvent
from src.daemon.state.models import RiskReportRecord
from src.v1.tasks.implementations.risk_report import RiskReportTask
from src.v1.tasks.models import DedupStrategy


def _make_risk_report(risk_status: str = "BREACH") -> PortfolioRiskReport:
    """Create PortfolioRiskReport for testing."""
    return PortfolioRiskReport(
        date="2026-02-21",
        risk_status=risk_status,
        var_95=0.04,
        var_99=0.06,
        cvar_95=0.05,
        cvar_99=0.07,
        cdar_95=0.08,
        max_drawdown=0.12,
        portfolio_volatility=0.18,
        current_exposure_percent=75.0,
        num_positions=5,
        var_limit_breached=risk_status == "BREACH",
        cvar_limit_breached=False,
    )


def _make_account() -> MagicMock:
    """Create mock BrokerAccountInfo."""
    account = MagicMock()
    account.positions = {}
    account.portfolio_value = 100000.0
    account.total_exposure = 75000.0
    return account


def _make_task(
    config: RiskLimitsConfig | None = None,
    include_queue: bool = True,
) -> tuple[RiskReportTask, MagicMock, MagicMock | None]:
    """Create task with mock dependencies.

    Returns:
        Tuple of (task, mock_state, mock_queue)
    """
    broker = MagicMock()
    state = MagicMock()
    state.record_risk_report = AsyncMock()
    state.get_last_risk_report = AsyncMock(return_value=None)

    scheduler = MagicMock()
    next_open = datetime.now(UTC) + timedelta(hours=2)
    scheduler.next_regular_open.return_value = next_open

    risk_manager = MagicMock()

    queue: MagicMock | None = None
    if include_queue:
        queue = MagicMock()
        queue.enqueue = AsyncMock()

    cfg = config or RiskLimitsConfig(enabled=True)

    task = RiskReportTask(
        risk_manager=risk_manager,
        broker=broker,
        queue=queue,
        state=state,
        scheduler=scheduler,
        config=cfg,
    )
    return task, state, queue


class TestTaskMetadata:
    """Tests for task name and schedule."""

    @pytest.mark.unit
    def test_name(self) -> None:
        task, _, _ = _make_task()
        assert task.name == "risk_report"

    @pytest.mark.unit
    def test_schedule_daily_dedup(self) -> None:
        task, _, _ = _make_task()
        schedule = task.schedule
        assert schedule.dedup == DedupStrategy.DAILY
        assert schedule.enabled is True
        assert schedule.time == "16:30"

    @pytest.mark.unit
    def test_schedule_disabled(self) -> None:
        cfg = RiskLimitsConfig(enabled=False)
        task, _, _ = _make_task(config=cfg)
        assert task.schedule.enabled is False

    @pytest.mark.unit
    def test_repr(self) -> None:
        task, _, _ = _make_task()
        r = repr(task)
        assert "RiskReportTask" in r
        assert "enabled=True" in r


class TestExecute:
    """Tests for execute method."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_records_risk_report(self) -> None:
        """Records RiskReportRecord regardless of status."""
        report = _make_risk_report("OK")
        account = _make_account()
        task, state, _ = _make_task()

        with patch("src.v1.tasks.implementations.risk_report.asyncio") as mock_asyncio:
            mock_asyncio.to_thread = AsyncMock(side_effect=[account, report])
            mock_asyncio.create_task = MagicMock()
            await task.execute()

        state.record_risk_report.assert_called_once()
        call_arg = state.record_risk_report.call_args.args[0]
        assert isinstance(call_arg, RiskReportRecord)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_enqueues_on_breach(self) -> None:
        """Enqueues event when risk_status is BREACH."""
        report = _make_risk_report("BREACH")
        account = _make_account()
        task, _, queue = _make_task()

        with patch("src.v1.tasks.implementations.risk_report.asyncio") as mock_asyncio:
            mock_asyncio.to_thread = AsyncMock(side_effect=[account, report])
            mock_asyncio.create_task = MagicMock()
            await task.execute()

        assert queue is not None
        queue.enqueue.assert_called_once()
        event = queue.enqueue.call_args.args[0]
        assert isinstance(event, RiskReportEvent)
        assert event.risk_status == "BREACH"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_enqueues_on_warning(self) -> None:
        """Enqueues event when risk_status is WARNING."""
        report = _make_risk_report("WARNING")
        account = _make_account()
        task, _, queue = _make_task()

        with patch("src.v1.tasks.implementations.risk_report.asyncio") as mock_asyncio:
            mock_asyncio.to_thread = AsyncMock(side_effect=[account, report])
            mock_asyncio.create_task = MagicMock()
            await task.execute()

        assert queue is not None
        queue.enqueue.assert_called_once()
        event = queue.enqueue.call_args.args[0]
        assert event.risk_status == "WARNING"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_no_enqueue_for_ok_status(self) -> None:
        """Does not enqueue event when risk_status is OK."""
        report = _make_risk_report("OK")
        account = _make_account()
        task, _, queue = _make_task()

        with patch("src.v1.tasks.implementations.risk_report.asyncio") as mock_asyncio:
            mock_asyncio.to_thread = AsyncMock(side_effect=[account, report])
            mock_asyncio.create_task = MagicMock()
            await task.execute()

        assert queue is not None
        queue.enqueue.assert_not_called()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_no_enqueue_when_queue_is_none(self) -> None:
        """Does not enqueue when queue is None, even on BREACH."""
        report = _make_risk_report("BREACH")
        account = _make_account()
        task, state, _ = _make_task(include_queue=False)

        with patch("src.v1.tasks.implementations.risk_report.asyncio") as mock_asyncio:
            mock_asyncio.to_thread = AsyncMock(side_effect=[account, report])
            mock_asyncio.create_task = MagicMock()
            result = await task.execute()

        assert result.success is True
        state.record_risk_report.assert_called_once()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_enqueue_sets_process_after_to_next_open(self) -> None:
        """process_after in enqueue call matches scheduler.next_regular_open()."""
        report = _make_risk_report("BREACH")
        account = _make_account()
        task, _, queue = _make_task()
        expected_open = task._scheduler.next_regular_open()

        with patch("src.v1.tasks.implementations.risk_report.asyncio") as mock_asyncio:
            mock_asyncio.to_thread = AsyncMock(side_effect=[account, report])
            mock_asyncio.create_task = MagicMock()
            await task.execute()

        assert queue is not None
        call_kwargs = queue.enqueue.call_args.kwargs
        assert call_kwargs["process_after"] == expected_open

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_enqueue_ttl_covers_weekend(self) -> None:
        """TTL spans a Friday→Monday gap (~64h)."""
        report = _make_risk_report("BREACH")
        account = _make_account()
        task, _, queue = _make_task()

        # Simulate Friday close → Monday open (~64h gap)
        far_future = datetime.now(UTC) + timedelta(hours=64)
        task._scheduler.next_regular_open.return_value = far_future

        with patch("src.v1.tasks.implementations.risk_report.asyncio") as mock_asyncio:
            mock_asyncio.to_thread = AsyncMock(side_effect=[account, report])
            mock_asyncio.create_task = MagicMock()
            await task.execute()

        assert queue is not None
        ttl_hours = queue.enqueue.call_args.kwargs["ttl_hours"]
        assert ttl_hours > 64

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_result_success(self) -> None:
        """execute() returns successful TaskResult."""
        report = _make_risk_report("OK")
        account = _make_account()
        task, _, _ = _make_task()

        with patch("src.v1.tasks.implementations.risk_report.asyncio") as mock_asyncio:
            mock_asyncio.to_thread = AsyncMock(side_effect=[account, report])
            mock_asyncio.create_task = MagicMock()
            result = await task.execute()

        assert result.success is True
        assert result.task_name == "risk_report"
        assert "OK" in (result.message or "")

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_last_run_at_delegates_to_state(self) -> None:
        """last_run_at() returns value from state."""
        task, state, _ = _make_task()
        ts = datetime.now(UTC)
        state.get_last_risk_report.return_value = ts

        result = await task.last_run_at()

        assert result == ts
        state.get_last_risk_report.assert_called_once()
