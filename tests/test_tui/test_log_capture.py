"""Tests for TUI log capture utilities."""

import threading

import pytest
from loguru import logger

from src.tui.log_capture import clear_active_step, set_active_step, setup_log_capture, teardown_log_capture


@pytest.fixture(autouse=True)
def cleanup_log_capture():
    """Cleanup log capture state before each test."""
    clear_active_step()
    yield
    clear_active_step()


class TestLogCapture:
    """Test log capture functionality."""

    def test_setup_and_teardown(self) -> None:
        """Test setup and teardown of log capture."""
        captured_logs = []

        def progress_callback(step_id: str, status: str, detail: str) -> None:
            captured_logs.append((step_id, status, detail))

        handler_id = setup_log_capture(progress_callback)
        assert handler_id is not None

        teardown_log_capture(handler_id)

    def test_capture_info_log(self) -> None:
        """Test capturing INFO level log."""
        captured_logs = []

        def progress_callback(step_id: str, status: str, detail: str) -> None:
            captured_logs.append((step_id, status, detail))

        handler_id = setup_log_capture(progress_callback)

        set_active_step("test_step")
        logger.info("Test message")

        teardown_log_capture(handler_id)

        assert len(captured_logs) == 1
        assert captured_logs[0][0] == "test_step"
        assert captured_logs[0][1] == "active"
        assert "Test message" in captured_logs[0][2]

    def test_capture_warning_log(self) -> None:
        """Test capturing WARNING level log with prefix."""
        captured_logs = []

        def progress_callback(step_id: str, status: str, detail: str) -> None:
            captured_logs.append((step_id, status, detail))

        handler_id = setup_log_capture(progress_callback)

        set_active_step("test_step")
        logger.warning("Warning message")

        teardown_log_capture(handler_id)

        assert len(captured_logs) == 1
        assert "⚠" in captured_logs[0][2]
        assert "Warning message" in captured_logs[0][2]

    def test_capture_error_log(self) -> None:
        """Test capturing ERROR level log with prefix."""
        captured_logs = []

        def progress_callback(step_id: str, status: str, detail: str) -> None:
            captured_logs.append((step_id, status, detail))

        handler_id = setup_log_capture(progress_callback)

        set_active_step("test_step")
        logger.error("Error message")

        teardown_log_capture(handler_id)

        assert len(captured_logs) == 1
        assert "✗" in captured_logs[0][2]
        assert "Error message" in captured_logs[0][2]

    def test_no_capture_without_active_step(self) -> None:
        """Test that logs are not captured when no step is active."""
        captured_logs = []

        def progress_callback(step_id: str, status: str, detail: str) -> None:
            captured_logs.append((step_id, status, detail))

        handler_id = setup_log_capture(progress_callback)

        logger.info("Test message")

        teardown_log_capture(handler_id)

        assert len(captured_logs) == 0

    def test_clear_active_step(self) -> None:
        """Test clearing active step."""
        captured_logs = []

        def progress_callback(step_id: str, status: str, detail: str) -> None:
            captured_logs.append((step_id, status, detail))

        handler_id = setup_log_capture(progress_callback)

        set_active_step("test_step")
        logger.info("Message 1")

        clear_active_step()
        logger.info("Message 2")

        teardown_log_capture(handler_id)

        assert len(captured_logs) == 1  # Only first message captured
        assert "Message 1" in captured_logs[0][2]

    def test_truncate_long_message(self) -> None:
        """Test that long messages are truncated."""
        captured_logs = []

        def progress_callback(step_id: str, status: str, detail: str) -> None:
            captured_logs.append((step_id, status, detail))

        handler_id = setup_log_capture(progress_callback)

        set_active_step("test_step")
        long_message = "A" * 100
        logger.info(long_message)

        teardown_log_capture(handler_id)

        assert len(captured_logs) == 1
        assert len(captured_logs[0][2]) <= 50
        assert "..." in captured_logs[0][2]

    def test_thread_isolation(self) -> None:
        """Test that log capture is isolated per thread."""
        captured_logs = []

        def progress_callback(step_id: str, status: str, detail: str) -> None:
            captured_logs.append((step_id, status, detail))

        handler_id = setup_log_capture(progress_callback)

        # Set step in main thread
        set_active_step("main_step")

        # Log from different thread (should not capture)
        def thread_func() -> None:
            logger.info("Thread message")

        thread = threading.Thread(target=thread_func)
        thread.start()
        thread.join()

        # Log from main thread (should capture)
        logger.info("Main message")

        teardown_log_capture(handler_id)

        # Only main thread message should be captured
        assert len(captured_logs) == 1
        assert "Main message" in captured_logs[0][2]
