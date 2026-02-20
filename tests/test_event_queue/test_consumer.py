"""Tests for EventQueueConsumer."""

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock

import pytest

from src.coordinator.models import CoordinatorConfig, CoordinatorCycleResult
from src.v1.event_queue.consumer import EventQueueConsumer, _group_by_symbol_overlap
from src.v1.event_queue.models import QueuedMarketEvent


def _make_event(
    event_id: str = "evt-1",
    event_type: str = "news",
    symbols: list[str] | None = None,
    symbol: str | None = None,
) -> QueuedMarketEvent:
    """Create test QueuedMarketEvent."""
    triage = {
        "urgency": "IMMEDIATE",
        "sentiment": "BULLISH",
        "confidence": 0.8,
        "reasoning": "Test",
        "symbols": symbols or [],
    }
    event_data = {"event_type": event_type, "source": "test"}
    if symbol:
        event_data["symbol"] = symbol
    return QueuedMarketEvent(
        event_id=event_id,
        event_type=event_type,
        payload={"event": event_data, "triage": triage},
        enqueued_at=datetime.now(UTC),
    )


@pytest.fixture
def mock_queue() -> AsyncMock:
    """Create mock MarketEventQueue."""
    mock = AsyncMock()
    mock.dequeue = AsyncMock(return_value=[])
    mock.purge_expired = AsyncMock(return_value=0)
    mock.size = AsyncMock(return_value=0)
    return mock


@pytest.fixture
def mock_coordinator() -> AsyncMock:
    """Create mock TradingCoordinator."""
    mock = AsyncMock()
    mock.run_event_cycle = AsyncMock(
        return_value=CoordinatorCycleResult(
            summary="Event cycle complete",
            cycle_type="event_driven",
        )
    )
    return mock


@pytest.fixture
def mock_market_service() -> Mock:
    """Create mock MarketService."""
    mock = Mock()
    mock.is_open = Mock(return_value=True)
    mock.current_session = Mock(return_value=None)
    return mock


@pytest.fixture
def config() -> CoordinatorConfig:
    """Create test config."""
    return CoordinatorConfig(
        enabled=True,
        event_max_dequeue=5,
        event_max_tool_calls=15,
        event_poll_interval_seconds=5,
    )


@pytest.fixture
def consumer(
    mock_queue: AsyncMock,
    mock_coordinator: AsyncMock,
    mock_market_service: Mock,
    config: CoordinatorConfig,
) -> EventQueueConsumer:
    """Create EventQueueConsumer."""
    return EventQueueConsumer(
        queue=mock_queue,
        coordinator=mock_coordinator,
        market_service=mock_market_service,
        config=config,
    )


class TestGroupBySymbolOverlap:
    """Tests for _group_by_symbol_overlap."""

    def test_single_event(self) -> None:
        events = [_make_event(symbols=["AAPL"])]
        groups = _group_by_symbol_overlap(events)
        assert len(groups) == 1
        assert len(groups[0]) == 1

    def test_overlapping_symbols_grouped(self) -> None:
        events = [
            _make_event(event_id="e1", symbols=["AAPL", "MSFT"]),
            _make_event(event_id="e2", symbols=["MSFT", "GOOGL"]),
        ]
        groups = _group_by_symbol_overlap(events)
        assert len(groups) == 1
        assert len(groups[0]) == 2

    def test_non_overlapping_separate_groups(self) -> None:
        events = [
            _make_event(event_id="e1", symbols=["AAPL"]),
            _make_event(event_id="e2", symbols=["TSLA"]),
        ]
        groups = _group_by_symbol_overlap(events)
        assert len(groups) == 2

    def test_no_symbols_individual_groups(self) -> None:
        events = [
            _make_event(event_id="e1"),
            _make_event(event_id="e2"),
        ]
        groups = _group_by_symbol_overlap(events)
        assert len(groups) == 2

    def test_empty_events(self) -> None:
        assert _group_by_symbol_overlap([]) == []

    def test_mixed_with_and_without_symbols(self) -> None:
        events = [
            _make_event(event_id="e1", symbols=["AAPL"]),
            _make_event(event_id="e2"),  # no symbols
            _make_event(event_id="e3", symbols=["AAPL"]),
        ]
        groups = _group_by_symbol_overlap(events)
        assert len(groups) == 2  # AAPL group + no-symbol group


class TestEventQueueConsumer:
    """Tests for EventQueueConsumer."""

    @pytest.mark.asyncio
    async def test_poll_once_empty_queue(
        self, consumer: EventQueueConsumer, mock_queue: AsyncMock, mock_coordinator: AsyncMock
    ) -> None:
        """Empty queue skips coordinator call."""
        await consumer._poll_once()
        mock_queue.dequeue.assert_awaited_once()
        mock_coordinator.run_event_cycle.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_poll_once_with_events(
        self, consumer: EventQueueConsumer, mock_queue: AsyncMock, mock_coordinator: AsyncMock
    ) -> None:
        """Events trigger coordinator event cycle."""
        mock_queue.dequeue.return_value = [_make_event(symbols=["AAPL"])]
        await consumer._poll_once()
        mock_coordinator.run_event_cycle.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_poll_once_grouped_events(
        self, consumer: EventQueueConsumer, mock_queue: AsyncMock, mock_coordinator: AsyncMock
    ) -> None:
        """Events with different symbols create separate cycles."""
        mock_queue.dequeue.return_value = [
            _make_event(event_id="e1", symbols=["AAPL"]),
            _make_event(event_id="e2", symbols=["TSLA"]),
        ]
        await consumer._poll_once()
        assert mock_coordinator.run_event_cycle.await_count == 2

    @pytest.mark.asyncio
    async def test_poll_once_error_handling(
        self, consumer: EventQueueConsumer, mock_queue: AsyncMock, mock_coordinator: AsyncMock
    ) -> None:
        """Coordinator failure doesn't crash the consumer."""
        mock_queue.dequeue.return_value = [_make_event(symbols=["AAPL"])]
        mock_coordinator.run_event_cycle.side_effect = RuntimeError("boom")
        # Should not raise
        await consumer._poll_once()

    @pytest.mark.asyncio
    async def test_poll_once_passes_market_open(
        self,
        consumer: EventQueueConsumer,
        mock_queue: AsyncMock,
        mock_coordinator: AsyncMock,
        mock_market_service: Mock,
    ) -> None:
        """Market open flag is passed to coordinator."""
        mock_market_service.is_open.return_value = False
        mock_queue.dequeue.return_value = [_make_event(symbols=["AAPL"])]
        await consumer._poll_once()

        call_kwargs = mock_coordinator.run_event_cycle.call_args[1]
        assert call_kwargs["market_open"] is False

    @pytest.mark.asyncio
    async def test_run_loop_cancellation(self, consumer: EventQueueConsumer) -> None:
        """Consumer loop exits cleanly on cancellation."""
        task = asyncio.create_task(consumer.run())
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    @pytest.mark.asyncio
    async def test_purge_called_periodically(
        self, consumer: EventQueueConsumer, mock_queue: AsyncMock
    ) -> None:
        """Purge is triggered after ~20 poll iterations."""
        consumer._purge_counter = 19
        await consumer._poll_once()
        # Purge task should have been created (counter reset)
        assert consumer._purge_counter == 0
        # Give the background task a moment to run
        await asyncio.sleep(0.05)
        mock_queue.purge_expired.assert_awaited()

    def test_repr(self, consumer: EventQueueConsumer) -> None:
        repr_str = repr(consumer)
        assert "EventQueueConsumer" in repr_str
        assert "poll=5s" in repr_str
