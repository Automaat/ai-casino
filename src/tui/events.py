"""Event types for TUI communication."""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now()  # noqa: DTZ005


class EventType(StrEnum):
    """Types of events in the TUI."""

    USER_MESSAGE = "user_message"
    ASSISTANT_MESSAGE = "assistant_message"
    ANALYSIS_START = "analysis_start"
    ANALYSIS_COMPLETE = "analysis_complete"
    ANALYSIS_ERROR = "analysis_error"
    TOKEN = "token"
    STREAM_END = "stream_end"


class TUIEvent(BaseModel):
    """Base event for TUI communication."""

    type: EventType
    timestamp: datetime = Field(default_factory=_now)
    data: dict = Field(default_factory=dict)


class MessageEvent(TUIEvent):
    """Chat message event."""

    content: str = ""
    role: str = "user"


class AnalysisEvent(TUIEvent):
    """Analysis-related event."""

    symbol: str = ""
    signal: str | None = None
    confidence: float | None = None
    error: str | None = None
