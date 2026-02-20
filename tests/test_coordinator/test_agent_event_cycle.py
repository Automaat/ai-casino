"""Tests for TradingCoordinator.run_event_cycle."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock

import pytest

from src.strategies.session import TradingSession
from src.v1.coordinator.agent import TradingCoordinator
from src.v1.coordinator.models import CoordinatorConfig
from src.v1.event_queue.models import QueuedMarketEvent


def _make_event(
    event_id: str = "evt-1",
    event_type: str = "news",
    symbols: list[str] | None = None,
) -> QueuedMarketEvent:
    """Create test QueuedMarketEvent."""
    return QueuedMarketEvent(
        event_id=event_id,
        event_type=event_type,
        payload={
            "event": {"event_type": event_type, "source": "test"},
            "triage": {
                "urgency": "IMMEDIATE",
                "sentiment": "BULLISH",
                "confidence": 0.8,
                "reasoning": "Test event",
                "symbols": symbols or ["AAPL"],
            },
        },
        enqueued_at=datetime.now(UTC),
    )


@pytest.fixture
def coordinator_config() -> CoordinatorConfig:
    """Create test coordinator config."""
    return CoordinatorConfig(
        enabled=True,
        max_tool_calls=25,
        event_max_tool_calls=10,
        temperature=0.5,
        cycle_timeout_seconds=60,
        max_daily_trades=5,
        max_position_pct=10.0,
        min_confidence_to_trade=0.6,
    )


@pytest.fixture
def mock_llm() -> AsyncMock:
    """Create mock LLM client."""
    mock = AsyncMock()
    mock.acomplete_with_tools = AsyncMock(
        return_value="Event cycle complete. Analyzed AAPL based on news catalyst."
    )
    return mock


@pytest.fixture
def mock_tool_registry() -> Mock:
    """Create mock tool registry."""
    mock = Mock()
    mock.get_definitions = Mock(return_value=[])
    mock.aexecute = AsyncMock(return_value="Tool executed")
    mock.requires_confirmation = Mock(return_value=False)
    mock.__len__ = Mock(return_value=9)
    return mock


@pytest.fixture
def mock_memory() -> AsyncMock:
    """Create mock coordinator memory."""
    mock = AsyncMock()
    mock.retrieve_recent = AsyncMock(return_value=[])
    mock.get_today_summary = AsyncMock(return_value="No analyses today")
    mock.get_today_game_plan = AsyncMock(return_value="Game plan unavailable")
    mock.get_portfolio_summary = AsyncMock(return_value="No portfolio data")
    mock.query_decisions = AsyncMock(return_value=[])
    return mock


@pytest.fixture
def mock_broker() -> Mock:
    """Create mock broker."""
    from src.data.broker import BrokerAccountInfo

    mock = Mock()
    mock.get_account_info = Mock(
        return_value=BrokerAccountInfo(
            balance=10000.0,
            portfolio_value=10000.0,
            available_cash=10000.0,
            total_exposure=0.0,
            positions={},
        )
    )
    return mock


@pytest.fixture
def coordinator(
    mock_llm: AsyncMock,
    mock_tool_registry: Mock,
    mock_memory: AsyncMock,
    coordinator_config: CoordinatorConfig,
    mock_broker: Mock,
) -> TradingCoordinator:
    """Create TradingCoordinator instance."""
    return TradingCoordinator(
        llm_client=mock_llm,
        tool_registry=mock_tool_registry,
        memory=mock_memory,
        config=coordinator_config,
        broker=mock_broker,
        critic_agent=AsyncMock(),
    )


@pytest.mark.asyncio
async def test_event_cycle_success(coordinator: TradingCoordinator, mock_llm: AsyncMock) -> None:
    """Event cycle returns result with event metadata."""
    events = [_make_event()]
    result = await coordinator.run_event_cycle(events)

    assert result.cycle_type == "event_driven"
    assert result.event_ids == ["evt-1"]
    assert "Event cycle complete" in result.summary
    mock_llm.acomplete_with_tools.assert_awaited_once()


@pytest.mark.asyncio
async def test_event_cycle_uses_event_max_tool_calls(
    coordinator: TradingCoordinator, mock_llm: AsyncMock
) -> None:
    """Event cycle uses event_max_tool_calls instead of regular max_tool_calls."""
    events = [_make_event()]
    await coordinator.run_event_cycle(events)

    call_args = mock_llm.acomplete_with_tools.call_args
    params = call_args[0][0]
    assert params.max_tool_calls == 10  # event_max_tool_calls


@pytest.mark.asyncio
async def test_event_cycle_multiple_events(coordinator: TradingCoordinator) -> None:
    """Event cycle handles multiple events."""
    events = [
        _make_event(event_id="e1", symbols=["AAPL"]),
        _make_event(event_id="e2", event_type="anomaly", symbols=["TSLA"]),
    ]
    result = await coordinator.run_event_cycle(events)

    assert result.cycle_type == "event_driven"
    assert set(result.event_ids) == {"e1", "e2"}


@pytest.mark.asyncio
async def test_event_cycle_market_closed(coordinator: TradingCoordinator, mock_llm: AsyncMock) -> None:
    """Event cycle passes market_open=False to prompt builder."""
    events = [_make_event()]
    result = await coordinator.run_event_cycle(events, market_open=False)

    assert result.cycle_type == "event_driven"
    call_args = mock_llm.acomplete_with_tools.call_args
    params = call_args[0][0]
    assert "skip trade execution" in params.prompt


@pytest.mark.asyncio
async def test_event_cycle_error_handling(coordinator: TradingCoordinator, mock_llm: AsyncMock) -> None:
    """Event cycle returns error result on failure."""
    mock_llm.acomplete_with_tools.side_effect = RuntimeError("LLM failed")
    events = [_make_event()]

    result = await coordinator.run_event_cycle(events)

    assert "Error" in result.summary
    assert result.cycle_type == "event_driven"
    assert result.event_ids == ["evt-1"]


@pytest.mark.asyncio
async def test_event_cycle_timeout(coordinator: TradingCoordinator, mock_llm: AsyncMock) -> None:
    """Event cycle returns timeout result when cycle exceeds limit."""
    import asyncio

    async def slow_response(*args, **kwargs):
        await asyncio.sleep(999)

    mock_llm.acomplete_with_tools.side_effect = slow_response
    coordinator._config.cycle_timeout_seconds = 1

    events = [_make_event()]
    result = await coordinator.run_event_cycle(events)

    assert "timeout" in result.summary.lower()
    assert result.cycle_type == "event_driven"


@pytest.mark.asyncio
async def test_event_cycle_trading_session(coordinator: TradingCoordinator, mock_llm: AsyncMock) -> None:
    """Event cycle passes trading session to prompt."""
    events = [_make_event()]
    await coordinator.run_event_cycle(events, trading_session=TradingSession.PRE_MARKET)

    call_args = mock_llm.acomplete_with_tools.call_args
    params = call_args[0][0]
    assert "PRE_MARKET" in params.prompt
