"""Event types for TUI communication."""

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field
from textual.message import Message

from src.tui.commands import CommandResult


def _now() -> datetime:
    return datetime.now(tz=UTC)


class EventType(StrEnum):
    """Types of events in the TUI."""

    USER_MESSAGE = "user_message"
    ASSISTANT_MESSAGE = "assistant_message"
    ANALYSIS_START = "analysis_start"
    ANALYSIS_COMPLETE = "analysis_complete"
    ANALYSIS_ERROR = "analysis_error"
    TOKEN = "token"
    STREAM_END = "stream_end"
    TASK_START = "task_start"
    TASK_STEP = "task_step"
    TASK_COMPLETE = "task_complete"


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


class TaskEvent(TUIEvent):
    """Task progress event."""

    task_id: str = ""
    step_id: str = ""
    step_label: str = ""
    status: str = "pending"


class StreamingEvent(TUIEvent):
    """Token streaming event."""

    token: str = ""
    is_complete: bool = False


# Textual messages for thread-safe UI updates


class AnalysisProgress(Message):
    """Progress update from analysis worker."""

    def __init__(self, step_id: str, status: str, detail: str) -> None:
        """Initialize progress message.

        Args:
            step_id: Analysis step identifier (fetch_data, technical, decision)
            status: Step status (active, complete, error)
            detail: Progress detail message
        """
        self.step_id = step_id
        self.status = status
        self.detail = detail
        super().__init__()

    def __repr__(self) -> str:
        """Return string representation."""
        return f"AnalysisProgress(step={self.step_id}, status={self.status})"


class AnalysisComplete(Message):
    """Analysis completed message."""

    def __init__(self, result: CommandResult, symbol: str, command_type: str = "analyze") -> None:
        """Initialize completion message.

        Args:
            result: Command execution result
            symbol: Stock symbol analyzed
            command_type: Type of command (analyze, technical, sentiment, news, screen)
        """
        self.result = result
        self.symbol = symbol
        self.command_type = command_type
        super().__init__()

    def __repr__(self) -> str:
        """Return string representation."""
        return f"AnalysisComplete(symbol={self.symbol}, command={self.command_type})"
