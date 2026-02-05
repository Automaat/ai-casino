"""Log capture utilities for TUI progress display.

Captures log messages in worker threads and displays them under active progress steps.
"""

import contextlib
import threading
from collections.abc import Callable

from loguru import logger

# Thread-local storage for current active step
_current_step = threading.local()

# Thread-local storage for log buffer
_log_buffer = threading.local()

# Type alias for progress callback
ProgressCallback = Callable[[str, str, str], None]


class LogCaptureSink:
    """Custom loguru sink that captures logs and sends them to progress display."""

    def __init__(self, progress_callback: ProgressCallback) -> None:
        """Initialize log capture sink.

        Args:
            progress_callback: Callback to update progress display (step_id, status, detail)
        """
        self.progress_callback = progress_callback

    def __call__(self, message: object) -> None:
        """Handle log message from loguru.

        Args:
            message: Log message object from loguru
        """
        # Get current step from thread-local storage
        step_id = getattr(_current_step, "value", None)
        if not step_id:
            return  # No active step, skip

        # Extract log level and message content from record
        record = message.record
        level = record["level"].name
        text = record["message"]

        # Format message based on level
        if level == "ERROR":
            formatted = f"✗ {text}"
        elif level == "WARNING":
            formatted = f"⚠ {text}"
        else:  # INFO and below
            formatted = text

        # Truncate intelligently to ~50 chars
        formatted = self._truncate_message(formatted)

        # Update progress detail via callback
        self.progress_callback(step_id, "active", formatted)

    def _truncate_message(self, msg: str) -> str:
        """Truncate message intelligently, preserving important parts.

        Args:
            msg: Message to truncate

        Returns:
            Truncated message (~50 chars)
        """
        max_len = 50
        if len(msg) <= max_len:
            return msg

        # Try to preserve start and end
        if "..." not in msg:
            # Simple truncation with ellipsis
            return msg[: max_len - 3] + "..."

        # Already has ellipsis, just truncate
        return msg[:max_len]


def setup_log_capture(progress_callback: ProgressCallback) -> int:
    """Set up log capture for current thread.

    Args:
        progress_callback: Callback to update progress display

    Returns:
        Handler ID for later teardown
    """
    sink = LogCaptureSink(progress_callback)
    worker_thread = threading.current_thread()

    # Add sink with INFO+ level (no DEBUG spam)
    # Filter to only capture logs from this worker thread
    return logger.add(
        sink,
        level="INFO",
        format="{message}",  # Sink handles formatting
        filter=lambda record: record["thread"].id == worker_thread.ident,
    )


def teardown_log_capture(handler_id: int) -> None:
    """Remove log capture sink.

    Args:
        handler_id: Handler ID from setup_log_capture()
    """
    with contextlib.suppress(ValueError):
        logger.remove(handler_id)


def set_active_step(step_id: str) -> None:
    """Set current active step for log capture.

    Args:
        step_id: Step identifier to associate logs with
    """
    _current_step.value = step_id


def clear_active_step() -> None:
    """Clear current active step."""
    if hasattr(_current_step, "value"):
        delattr(_current_step, "value")
