"""Log capture utilities using loguru contextvars.

Uses logger.contextualize() for thread/async-safe log scoping.
The step context is stored in a threading.local() since contextualize()
can't be dynamically updated from outside the context manager.
"""

import contextlib
import threading
from collections.abc import Callable, Generator
from dataclasses import dataclass, field

from loguru import logger

# Type alias for progress callback
ProgressCallback = Callable[[str, str, str], None]

# Thread-local storage for current step (updated dynamically within context)
_step_context = threading.local()


@dataclass
class _LogCaptureState:
    """Mutable container for log capture state."""

    progress_callback: ProgressCallback | None = field(default=None)


_state = _LogCaptureState()


class LogCaptureSink:
    """Sink that forwards logs to progress display."""

    def __call__(self, message: object) -> None:
        """Handle log message from loguru."""
        if not _state.progress_callback:
            return

        record = message.record
        extra = record.get("extra", {})

        # Only capture logs from TUI worker context
        if not extra.get("tui_worker"):
            return

        # Get step from thread-local storage
        step_id = getattr(_step_context, "step", None)
        if not step_id:
            return

        level = record["level"].name
        text = record["message"]

        # Format message based on level
        if level == "ERROR":
            formatted = f"✗ {text}"
        elif level == "WARNING":
            formatted = f"⚠ {text}"
        else:
            formatted = text

        # Truncate to ~50 chars
        if len(formatted) > 50:
            formatted = formatted[:47] + "..."

        _state.progress_callback(step_id, "active", formatted)


def setup_log_capture(progress_callback: ProgressCallback) -> int:
    """Set up log capture sink.

    Args:
        progress_callback: Callback to update progress display

    Returns:
        Handler ID for later teardown
    """
    _state.progress_callback = progress_callback

    return logger.add(
        LogCaptureSink(),
        level="INFO",
        format="{message}",
        filter=lambda r: r["extra"].get("tui_worker", False),
    )


def teardown_log_capture(handler_id: int) -> None:
    """Remove log capture sink."""
    with contextlib.suppress(ValueError):
        logger.remove(handler_id)
    _state.progress_callback = None


@contextlib.contextmanager
def worker_log_context() -> Generator[None, None, None]:
    """Context manager that marks all logs as coming from TUI worker.

    Usage:
        with worker_log_context():
            # All logger calls here will have tui_worker=True in extra
            logger.info("This gets captured")
    """
    with logger.contextualize(tui_worker=True):
        yield


def set_active_step(step_id: str) -> None:
    """Set current active step for log capture.

    Args:
        step_id: Step identifier to associate logs with
    """
    _step_context.step = step_id


def clear_active_step() -> None:
    """Clear current active step."""
    if hasattr(_step_context, "step"):
        delattr(_step_context, "step")
