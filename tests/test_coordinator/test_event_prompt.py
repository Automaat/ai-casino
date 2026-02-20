"""Tests for EventCyclePromptBuilder."""

from datetime import UTC, datetime

import pytest

from src.strategies.session import TradingSession
from src.v1.coordinator.event_prompt import EventCycleContext, EventCyclePromptBuilder, extract_symbols
from src.v1.coordinator.models import CoordinatorConfig
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
        "reasoning": "Strong catalyst detected",
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
def config() -> CoordinatorConfig:
    """Create test coordinator config."""
    return CoordinatorConfig(
        enabled=True,
        max_tool_calls=25,
        event_max_tool_calls=15,
        event_max_dequeue=5,
        max_daily_trades=10,
        max_position_pct=10.0,
        min_confidence_to_trade=0.6,
    )


@pytest.fixture
def builder() -> EventCyclePromptBuilder:
    """Create prompt builder."""
    return EventCyclePromptBuilder()


def _ctx(
    positions_summary: str = "",
    session: TradingSession = TradingSession.REGULAR,
    market_open: bool = True,
    game_plan: str = "",
) -> EventCycleContext:
    """Build EventCycleContext with defaults."""
    return EventCycleContext(
        positions_summary=positions_summary,
        session=session,
        market_open=market_open,
        game_plan=game_plan,
    )


class TestExtractSymbols:
    """Tests for extract_symbols."""

    def test_extracts_from_triage(self) -> None:
        event = _make_event(symbols=["AAPL", "MSFT"])
        assert extract_symbols([event]) == {"AAPL", "MSFT"}

    def test_extracts_from_event_data(self) -> None:
        event = _make_event(symbol="TSLA")
        assert extract_symbols([event]) == {"TSLA"}

    def test_merges_triage_and_event(self) -> None:
        event = _make_event(symbols=["AAPL"], symbol="TSLA")
        assert extract_symbols([event]) == {"AAPL", "TSLA"}

    def test_empty_events(self) -> None:
        assert extract_symbols([]) == set()

    def test_no_symbols(self) -> None:
        event = _make_event()
        assert extract_symbols([event]) == set()

    def test_deduplicates_across_events(self) -> None:
        e1 = _make_event(event_id="e1", symbols=["AAPL"])
        e2 = _make_event(event_id="e2", symbols=["AAPL", "MSFT"])
        assert extract_symbols([e1, e2]) == {"AAPL", "MSFT"}


class TestEventCyclePromptBuilder:
    """Tests for EventCyclePromptBuilder."""

    def test_builds_prompt_with_header(
        self, builder: EventCyclePromptBuilder, config: CoordinatorConfig
    ) -> None:
        events = [_make_event(symbols=["AAPL"])]
        prompt = builder.build(
            events=events,
            context=_ctx(positions_summary="No open positions", market_open=True),
            config=config,
        )
        assert "Event-Driven Cycle" in prompt
        assert "AAPL" in prompt
        assert "No open positions" in prompt
        assert "1 event(s)" in prompt

    def test_market_closed_flag(self, builder: EventCyclePromptBuilder, config: CoordinatorConfig) -> None:
        events = [_make_event()]
        prompt = builder.build(events=events, context=_ctx(market_open=False), config=config)
        assert "skip trade execution" in prompt

    def test_multiple_event_types(self, builder: EventCyclePromptBuilder, config: CoordinatorConfig) -> None:
        events = [
            _make_event(event_id="e1", event_type="news", symbols=["AAPL"]),
            _make_event(event_id="e2", event_type="anomaly", symbols=["TSLA"]),
        ]
        prompt = builder.build(events=events, context=_ctx(market_open=True), config=config)
        assert "News Event" in prompt
        assert "Market Anomaly Event" in prompt
        assert "2 event(s)" in prompt

    def test_unknown_event_type_fallback_to_news(
        self, builder: EventCyclePromptBuilder, config: CoordinatorConfig
    ) -> None:
        events = [_make_event(event_type="unknown_type")]
        prompt = builder.build(events=events, context=_ctx(market_open=True), config=config)
        assert "News Event" in prompt

    def test_risk_limits_in_header(self, builder: EventCyclePromptBuilder, config: CoordinatorConfig) -> None:
        events = [_make_event()]
        prompt = builder.build(events=events, context=_ctx(market_open=True), config=config)
        assert "10.0%" in prompt
        assert "60%" in prompt

    def test_each_event_type_template(
        self, builder: EventCyclePromptBuilder, config: CoordinatorConfig
    ) -> None:
        for event_type in ["news", "social", "filing", "trump", "anomaly", "news_trending"]:
            events = [_make_event(event_type=event_type)]
            prompt = builder.build(events=events, context=_ctx(market_open=True), config=config)
            assert len(prompt) > 100

    def test_signal_event_includes_game_plan(
        self, builder: EventCyclePromptBuilder, config: CoordinatorConfig
    ) -> None:
        events = [_make_event(event_type="signal", symbol="AAPL")]
        game_plan = "Priority: AAPL, MSFT | Risk: LOW | Sector: Tech"
        prompt = builder.build(
            events=events,
            context=_ctx(market_open=True, game_plan=game_plan),
            config=config,
        )
        assert game_plan in prompt
        assert "Pre-Market Signal Event" in prompt

    def test_game_plan_in_header_for_all_events(
        self, builder: EventCyclePromptBuilder, config: CoordinatorConfig
    ) -> None:
        events = [_make_event(event_type="news")]
        game_plan = "Priority: AAPL | Risk: NEUTRAL | Sector: Tech"
        prompt = builder.build(
            events=events,
            context=_ctx(market_open=True, game_plan=game_plan),
            config=config,
        )
        assert game_plan in prompt
        assert "Today's Game Plan" in prompt

    def test_empty_game_plan_omitted_from_header(
        self, builder: EventCyclePromptBuilder, config: CoordinatorConfig
    ) -> None:
        events = [_make_event(event_type="news")]
        prompt = builder.build(
            events=events,
            context=_ctx(market_open=True, game_plan=""),
            config=config,
        )
        assert "Today's Game Plan" not in prompt
