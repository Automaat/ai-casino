"""Tests for EventCyclePromptBuilder."""

from datetime import UTC, datetime

import pytest

from src.coordinator.event_prompt import EventCyclePromptBuilder, extract_symbols
from src.coordinator.models import CoordinatorConfig
from src.event_queue.models import QueuedMarketEvent
from src.strategies.session import TradingSession


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
            positions_summary="No open positions",
            session=TradingSession.REGULAR,
            config=config,
            market_open=True,
        )
        assert "Event-Driven Cycle" in prompt
        assert "AAPL" in prompt
        assert "No open positions" in prompt
        assert "1 event(s)" in prompt

    def test_market_closed_flag(self, builder: EventCyclePromptBuilder, config: CoordinatorConfig) -> None:
        events = [_make_event()]
        prompt = builder.build(
            events=events,
            positions_summary="",
            session=TradingSession.REGULAR,
            config=config,
            market_open=False,
        )
        assert "skip trade execution" in prompt

    def test_multiple_event_types(self, builder: EventCyclePromptBuilder, config: CoordinatorConfig) -> None:
        events = [
            _make_event(event_id="e1", event_type="news", symbols=["AAPL"]),
            _make_event(event_id="e2", event_type="anomaly", symbols=["TSLA"]),
        ]
        prompt = builder.build(
            events=events,
            positions_summary="",
            session=TradingSession.REGULAR,
            config=config,
            market_open=True,
        )
        assert "News Event" in prompt
        assert "Market Anomaly Event" in prompt
        assert "2 event(s)" in prompt

    def test_unknown_event_type_fallback_to_news(
        self, builder: EventCyclePromptBuilder, config: CoordinatorConfig
    ) -> None:
        events = [_make_event(event_type="unknown_type")]
        prompt = builder.build(
            events=events,
            positions_summary="",
            session=TradingSession.REGULAR,
            config=config,
            market_open=True,
        )
        # Falls back to news template
        assert "News Event" in prompt

    def test_risk_limits_in_header(self, builder: EventCyclePromptBuilder, config: CoordinatorConfig) -> None:
        events = [_make_event()]
        prompt = builder.build(
            events=events,
            positions_summary="",
            session=TradingSession.REGULAR,
            config=config,
            market_open=True,
        )
        assert "10.0%" in prompt
        assert "60%" in prompt

    def test_each_event_type_template(
        self, builder: EventCyclePromptBuilder, config: CoordinatorConfig
    ) -> None:
        for event_type in ["news", "social", "filing", "trump", "anomaly", "news_trending"]:
            events = [_make_event(event_type=event_type)]
            prompt = builder.build(
                events=events,
                positions_summary="",
                session=TradingSession.REGULAR,
                config=config,
                market_open=True,
            )
            assert len(prompt) > 100
