"""Tests for PositionReviewTask."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.daemon.config.position_review import PositionReviewConfig
from src.daemon.events import PositionReviewEvent
from src.strategies.session import TradingSession
from src.v1.tasks.implementations.position_review import PositionReviewTask
from src.v1.tasks.models import DedupStrategy


def _make_broker_position(
    symbol: str = "AAPL",
    qty: float = 10.0,
    avg_entry_price: float = 150.0,
    unrealized_pnl: float = 50.0,
    unrealized_pnl_percent: float = 3.3,
    market_value: float = 1550.0,
) -> MagicMock:
    """Create mock BrokerPosition."""
    pos = MagicMock()
    pos.symbol = symbol
    pos.qty = qty
    pos.avg_entry_price = avg_entry_price
    pos.unrealized_pnl = unrealized_pnl
    pos.unrealized_pnl_percent = unrealized_pnl_percent
    pos.market_value = market_value
    return pos


def _make_account(
    positions: dict | None = None,
    portfolio_value: float = 100000.0,
    total_exposure: float = 50000.0,
) -> MagicMock:
    """Create mock BrokerAccountInfo."""
    account = MagicMock()
    account.positions = positions or {}
    account.portfolio_value = portfolio_value
    account.total_exposure = total_exposure
    return account


def _make_task(
    config: PositionReviewConfig | None = None,
    session: TradingSession = TradingSession.REGULAR,
    db_engine: MagicMock | None = None,
) -> tuple[PositionReviewTask, MagicMock, MagicMock]:
    """Create task with mock dependencies.

    Returns:
        Tuple of (task, mock_broker, mock_queue)
    """
    broker = MagicMock()
    queue = MagicMock()
    queue.enqueue = AsyncMock()
    scheduler = MagicMock()
    scheduler.get_trading_session.return_value = session

    task = PositionReviewTask(
        broker=broker,
        queue=queue,
        config=config or PositionReviewConfig(enabled=True),
        scheduler=scheduler,
        database_engine=db_engine,
    )
    return task, broker, queue


class TestTaskMetadata:
    """Tests for task name and schedule."""

    @pytest.mark.unit
    def test_name(self) -> None:
        task, _, _ = _make_task()
        assert task.name == "position_review"

    @pytest.mark.unit
    def test_schedule_uses_interval_dedup(self) -> None:
        config = PositionReviewConfig(enabled=True, interval_minutes=45)
        task, _, _ = _make_task(config=config)
        schedule = task.schedule
        assert schedule.dedup == DedupStrategy.INTERVAL
        assert schedule.dedup_interval_minutes == 45
        assert schedule.time is None
        assert schedule.enabled is True

    @pytest.mark.unit
    def test_schedule_disabled(self) -> None:
        config = PositionReviewConfig(enabled=False)
        task, _, _ = _make_task(config=config)
        assert task.schedule.enabled is False

    @pytest.mark.unit
    def test_repr(self) -> None:
        task, _, _ = _make_task()
        r = repr(task)
        assert "PositionReviewTask" in r
        assert "enabled=True" in r


class TestExecute:
    """Tests for execute method."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_skip_outside_regular_session(self) -> None:
        task, _broker, queue = _make_task(session=TradingSession.PRE_MARKET)
        result = await task.execute()
        assert result.success is True
        assert "Skipped" in (result.message or "")
        queue.enqueue.assert_not_called()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_no_positions_skips(self) -> None:
        task, _broker, queue = _make_task()

        with patch("src.v1.tasks.implementations.position_review.asyncio") as mock_asyncio:
            mock_asyncio.to_thread = AsyncMock(return_value=_make_account(positions={}))
            result = await task.execute()

        assert result.success is True
        assert result.message == "No positions"
        queue.enqueue.assert_not_called()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_enqueues_with_positions(self) -> None:
        pos = _make_broker_position("AAPL", qty=10.0, unrealized_pnl=50.0, unrealized_pnl_percent=3.3)
        account = _make_account(positions={"AAPL": pos})
        task, _broker, queue = _make_task()

        with patch("src.v1.tasks.implementations.position_review.asyncio") as mock_asyncio:
            mock_asyncio.to_thread = AsyncMock(return_value=account)
            result = await task.execute()

        assert result.success is True
        assert "1 positions reviewed" in (result.message or "")
        queue.enqueue.assert_called_once()

        call_args = queue.enqueue.call_args
        event = call_args.args[0] if call_args.args else call_args.kwargs.get("event")
        triage = call_args.args[1] if len(call_args.args) > 1 else call_args.kwargs.get("triage")

        assert isinstance(event, PositionReviewEvent)
        assert len(event.positions) == 1
        assert event.positions[0].symbol == "AAPL"
        assert triage.urgency.value == "IMMEDIATE"
        assert triage.symbols == ["AAPL"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_updates_last_run(self) -> None:
        task, _broker, _queue = _make_task()
        assert await task.last_run_at() is None

        with patch("src.v1.tasks.implementations.position_review.asyncio") as mock_asyncio:
            mock_asyncio.to_thread = AsyncMock(return_value=_make_account(positions={}))
            await task.execute()

        last = await task.last_run_at()
        assert last is not None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_graceful_without_db(self) -> None:
        """Task works without database engine (no entry metadata)."""
        pos = _make_broker_position("TSLA", qty=5.0, unrealized_pnl=-100.0, unrealized_pnl_percent=-6.0)
        account = _make_account(positions={"TSLA": pos})
        task, _broker, queue = _make_task(db_engine=None)

        with patch("src.v1.tasks.implementations.position_review.asyncio") as mock_asyncio:
            mock_asyncio.to_thread = AsyncMock(return_value=account)
            result = await task.execute()

        assert result.success is True
        event = queue.enqueue.call_args.args[0]
        assert event.positions[0].days_held is None
        assert event.positions[0].entry_confidence is None
        assert "SIGNIFICANT_LOSS" in event.positions[0].flags


class TestComputeHealthFlags:
    """Tests for static health flag computation."""

    @pytest.mark.unit
    def test_significant_loss(self) -> None:
        flags = PositionReviewTask.compute_health_flags(-5.0, None, None)
        assert "SIGNIFICANT_LOSS" in flags

    @pytest.mark.unit
    def test_deteriorating(self) -> None:
        flags = PositionReviewTask.compute_health_flags(-2.5, None, None)
        assert "DETERIORATING" in flags
        assert "SIGNIFICANT_LOSS" not in flags

    @pytest.mark.unit
    def test_extended_hold(self) -> None:
        flags = PositionReviewTask.compute_health_flags(0.0, 25, None)
        assert "EXTENDED_HOLD" in flags

    @pytest.mark.unit
    def test_aging(self) -> None:
        flags = PositionReviewTask.compute_health_flags(0.0, 12, None)
        assert "AGING" in flags
        assert "EXTENDED_HOLD" not in flags

    @pytest.mark.unit
    def test_low_entry_confidence(self) -> None:
        flags = PositionReviewTask.compute_health_flags(0.0, None, 0.5)
        assert "LOW_ENTRY_CONFIDENCE" in flags

    @pytest.mark.unit
    def test_consider_profit_taking(self) -> None:
        flags = PositionReviewTask.compute_health_flags(12.0, None, None)
        assert "CONSIDER_PROFIT_TAKING" in flags

    @pytest.mark.unit
    def test_no_flags_healthy(self) -> None:
        flags = PositionReviewTask.compute_health_flags(3.0, 5, 0.8)
        assert flags == []

    @pytest.mark.unit
    def test_multiple_flags(self) -> None:
        flags = PositionReviewTask.compute_health_flags(-7.0, 25, 0.4)
        assert "SIGNIFICANT_LOSS" in flags
        assert "EXTENDED_HOLD" in flags
        assert "LOW_ENTRY_CONFIDENCE" in flags
