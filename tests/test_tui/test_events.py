"""Tests for TUI events."""

from datetime import datetime

from src.tui.events import AnalysisEvent, EventType, MessageEvent, TUIEvent


class TestEventType:
    def test_event_types(self):
        assert EventType.USER_MESSAGE == "user_message"
        assert EventType.ASSISTANT_MESSAGE == "assistant_message"
        assert EventType.ANALYSIS_START == "analysis_start"
        assert EventType.ANALYSIS_COMPLETE == "analysis_complete"
        assert EventType.ANALYSIS_ERROR == "analysis_error"
        assert str(EventType.TOKEN) == "token"
        assert EventType.STREAM_END == "stream_end"


class TestTUIEvent:
    def test_default_timestamp(self):
        event = TUIEvent(type=EventType.USER_MESSAGE)

        assert event.timestamp is not None
        assert isinstance(event.timestamp, datetime)

    def test_default_data(self):
        event = TUIEvent(type=EventType.USER_MESSAGE)

        assert event.data == {}


class TestMessageEvent:
    def test_default_values(self):
        event = MessageEvent(type=EventType.USER_MESSAGE)

        assert event.content == ""
        assert event.role == "user"

    def test_custom_values(self):
        event = MessageEvent(
            type=EventType.ASSISTANT_MESSAGE,
            content="Hello",
            role="assistant",
        )

        assert event.content == "Hello"
        assert event.role == "assistant"


class TestAnalysisEvent:
    def test_default_values(self):
        event = AnalysisEvent(type=EventType.ANALYSIS_START)

        assert event.symbol == ""
        assert event.signal is None
        assert event.confidence is None
        assert event.error is None

    def test_custom_values(self):
        event = AnalysisEvent(
            type=EventType.ANALYSIS_COMPLETE,
            symbol="AAPL",
            signal="BUY",
            confidence=0.85,
        )

        assert event.symbol == "AAPL"
        assert event.signal == "BUY"
        assert event.confidence == 0.85

    def test_error_event(self):
        event = AnalysisEvent(
            type=EventType.ANALYSIS_ERROR,
            symbol="INVALID",
            error="Symbol not found",
        )

        assert event.error == "Symbol not found"
