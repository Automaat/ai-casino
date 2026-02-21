"""Tests for PositionReviewEvent model and event prompt rendering."""

from datetime import UTC, datetime

import pytest

from src.daemon.events import EnrichedPosition, PositionReviewEvent
from src.strategies.session import TradingSession
from src.v1.coordinator.event_prompt import (
    EventCycleContext,
    EventCyclePromptBuilder,
    _format_event_details,
    extract_symbols,
)
from src.v1.coordinator.models import CoordinatorConfig
from src.v1.event_queue.models import QueuedMarketEvent


def _make_position(
    symbol: str = "AAPL",
    pnl_pct: float = 3.3,
    days_held: int | None = 5,
    flags: list[str] | None = None,
) -> EnrichedPosition:
    """Create test EnrichedPosition."""
    return EnrichedPosition(
        symbol=symbol,
        qty=10.0,
        avg_entry_price=150.0,
        current_price=155.0,
        unrealized_pnl=50.0,
        unrealized_pnl_percent=pnl_pct,
        days_held=days_held,
        entry_confidence=0.75,
        entry_signal="BUY",
        stop_loss_price=145.0,
        flags=flags or [],
    )


class TestEnrichedPosition:
    """Tests for EnrichedPosition model."""

    @pytest.mark.unit
    def test_create(self) -> None:
        pos = _make_position()
        assert pos.symbol == "AAPL"
        assert pos.qty == 10.0

    @pytest.mark.unit
    def test_repr_with_flags(self) -> None:
        pos = _make_position(flags=["AGING", "LOW_ENTRY_CONFIDENCE"])
        r = repr(pos)
        assert "AAPL" in r
        assert "AGING" in r

    @pytest.mark.unit
    def test_repr_no_flags(self) -> None:
        pos = _make_position(flags=[])
        assert "none" in repr(pos)


class TestPositionReviewEvent:
    """Tests for PositionReviewEvent model."""

    @pytest.mark.unit
    def test_create_with_defaults(self) -> None:
        event = PositionReviewEvent(
            positions=[_make_position()],
            portfolio_value=100000.0,
            total_exposure=50000.0,
        )
        assert event.event_type == "position_review"
        assert event.source == "position_review_task"
        assert event.event_id.startswith("pos-review-")

    @pytest.mark.unit
    def test_to_prompt_text(self) -> None:
        positions = [
            _make_position("AAPL", pnl_pct=3.3, days_held=5, flags=[]),
            _make_position("TSLA", pnl_pct=-6.0, days_held=25, flags=["SIGNIFICANT_LOSS", "EXTENDED_HOLD"]),
        ]
        event = PositionReviewEvent(
            positions=positions,
            portfolio_value=100000.0,
            total_exposure=50000.0,
        )
        text = event.to_prompt_text()
        assert "POSITION REVIEW (2 positions)" in text
        assert "AAPL" in text
        assert "TSLA" in text
        assert "SIGNIFICANT_LOSS" in text
        assert "$100,000" in text

    @pytest.mark.unit
    def test_repr(self) -> None:
        event = PositionReviewEvent(
            positions=[_make_position("AAPL"), _make_position("MSFT")],
            portfolio_value=100000.0,
            total_exposure=50000.0,
        )
        r = repr(event)
        assert "AAPL" in r
        assert "MSFT" in r


class TestEventPromptRendering:
    """Tests for position_review event prompt rendering."""

    @pytest.mark.unit
    def test_format_event_details(self) -> None:
        event_data = {
            "event_type": "position_review",
            "source": "position_review_task",
            "portfolio_value": 100000,
            "total_exposure": 50000,
            "positions": [
                {
                    "symbol": "AAPL",
                    "qty": 10,
                    "avg_entry_price": 150.0,
                    "current_price": 155.0,
                    "unrealized_pnl_percent": 3.3,
                    "days_held": 5,
                    "entry_confidence": 0.75,
                    "stop_loss_price": 145.0,
                    "flags": ["CONSIDER_PROFIT_TAKING"],
                }
            ],
        }
        details = _format_event_details(event_data)
        assert "position_review" in details
        assert "AAPL" in details
        assert "CONSIDER_PROFIT_TAKING" in details
        assert "$100,000" in details

    @pytest.mark.unit
    def test_prompt_builder_renders_position_review(self) -> None:
        builder = EventCyclePromptBuilder()
        config = CoordinatorConfig(
            enabled=True,
            max_tool_calls=25,
            event_max_tool_calls=15,
            event_max_dequeue=5,
            max_daily_trades=10,
            max_position_pct=10.0,
            min_confidence_to_trade=0.6,
        )
        event = QueuedMarketEvent(
            event_id="pos-review-20260221-1000",
            event_type="position_review",
            payload={
                "event": {
                    "event_type": "position_review",
                    "source": "position_review_task",
                    "portfolio_value": 100000,
                    "total_exposure": 50000,
                    "positions": [
                        {
                            "symbol": "AAPL",
                            "qty": 10,
                            "avg_entry_price": 150.0,
                            "current_price": 155.0,
                            "unrealized_pnl_percent": 3.3,
                            "days_held": 5,
                            "entry_confidence": 0.75,
                            "stop_loss_price": 145.0,
                            "flags": [],
                        }
                    ],
                },
                "triage": {
                    "urgency": "IMMEDIATE",
                    "sentiment": "NEUTRAL",
                    "confidence": 1.0,
                    "reasoning": "Scheduled position review",
                    "symbols": ["AAPL"],
                },
            },
            enqueued_at=datetime.now(UTC),
        )
        ctx = EventCycleContext(
            positions_summary="1 position",
            session=TradingSession.REGULAR,
            market_open=True,
        )
        prompt = builder.build(events=[event], context=ctx, config=config)
        assert "Position Review" in prompt
        assert "Portfolio Health Check" in prompt

    @pytest.mark.unit
    def test_extract_symbols_from_position_review(self) -> None:
        event = QueuedMarketEvent(
            event_id="pos-review-test",
            event_type="position_review",
            payload={
                "event": {"event_type": "position_review"},
                "triage": {"symbols": ["AAPL", "TSLA"], "urgency": "IMMEDIATE"},
            },
            enqueued_at=datetime.now(UTC),
        )
        symbols = extract_symbols([event])
        assert symbols == {"AAPL", "TSLA"}
