"""Integration tests for Daemon lifecycle with watchers."""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.daemon.lifecycle import DaemonLifecycle


@pytest.fixture
def mock_news_watcher():
    """Mock NewsWatcher for testing."""
    watcher = Mock()
    watcher.running = False

    # Mock run() as async method
    async def run_mock():
        watcher.running = True
        while watcher.running:
            await asyncio.sleep(0.1)

    watcher.run = AsyncMock(side_effect=run_mock)
    return watcher


@pytest.fixture
def minimal_daemon_components(mock_news_watcher):
    """Minimal DaemonComponents for lifecycle testing."""
    from src.daemon.config import DaemonConfig
    from src.daemon.factory import DaemonComponents
    from src.daemon.state import DaemonState

    # Create minimal config
    config = DaemonConfig()
    config.database.enable_persistence = False
    config.api.enabled = False

    # Create minimal components
    components = Mock(spec=DaemonComponents)
    components.config = config
    components.state = DaemonState()
    components.news_watcher = mock_news_watcher
    components.social_watcher = None
    components.trump_watcher = None
    components.position_manager = None

    return components


@pytest.mark.integration
@pytest.mark.asyncio
async def test_lifecycle_starts_watchers_on_startup(minimal_daemon_components):
    """Test watcher tasks created on startup."""
    lifecycle = DaemonLifecycle(minimal_daemon_components)

    with patch("signal.signal"):  # Prevent global signal handler mutation
        await lifecycle.startup()

    # Verify watcher task was created
    assert len(lifecycle._watcher_tasks) == 1
    assert not lifecycle._watcher_tasks[0].done()

    # Cleanup
    await lifecycle.shutdown()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_lifecycle_stops_watchers_on_shutdown(minimal_daemon_components):
    """Test watchers stopped on shutdown."""
    lifecycle = DaemonLifecycle(minimal_daemon_components)

    with patch("signal.signal"):  # Prevent global signal handler mutation
        await lifecycle.startup()
    assert len(lifecycle._watcher_tasks) == 1
    task = lifecycle._watcher_tasks[0]

    # Shutdown
    await lifecycle.shutdown()

    # Verify tasks cleared and completed/cancelled
    assert len(lifecycle._watcher_tasks) == 0
    assert task.done()  # Either completed or cancelled


@pytest.mark.integration
@pytest.mark.asyncio
async def test_watcher_crash_logged_not_propagated(minimal_daemon_components):
    """Test watcher crash logged via done_callback."""
    # Create watcher that raises exception
    mock_watcher = Mock()
    mock_watcher.running = False

    async def crash_run():
        raise ValueError("Test watcher crash")

    mock_watcher.run = AsyncMock(side_effect=crash_run)

    minimal_daemon_components.news_watcher = mock_watcher

    lifecycle = DaemonLifecycle(minimal_daemon_components)

    # Patch logger to verify error logged
    with patch("src.daemon.lifecycle.logger") as mock_logger, patch("signal.signal"):
        await lifecycle.startup()

        # Wait for watcher to crash
        await asyncio.sleep(0.1)

        # Verify task completed with exception
        assert lifecycle._watcher_tasks[0].done()

        # Trigger done callback
        exc = lifecycle._watcher_tasks[0].exception()
        assert isinstance(exc, ValueError)

        # Verify the crash was logged via logger.opt(...).error(...)
        mock_logger.opt.assert_called()
        mock_logger.opt.return_value.error.assert_called()
        error_args, _ = mock_logger.opt.return_value.error.call_args
        if error_args:
            assert "Test watcher crash" in str(error_args[0])

    # Cleanup
    await lifecycle.shutdown()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_multiple_watchers_run_independently():
    """Test multiple watchers run concurrently without blocking."""
    from src.daemon.config import DaemonConfig
    from src.daemon.factory import DaemonComponents
    from src.daemon.state import DaemonState

    # Create two mock watchers
    news_watcher = Mock()
    social_watcher = Mock()

    news_cycles = []
    social_cycles = []

    async def news_run():
        news_watcher.running = True
        while news_watcher.running:
            news_cycles.append(1)
            await asyncio.sleep(0.1)

    async def social_run():
        social_watcher.running = True
        while social_watcher.running:
            social_cycles.append(1)
            await asyncio.sleep(0.05)  # Faster cycle

    news_watcher.run = AsyncMock(side_effect=news_run)
    social_watcher.run = AsyncMock(side_effect=social_run)

    # Create components with both watchers
    config = DaemonConfig()
    config.database.enable_persistence = False
    config.api.enabled = False

    components = Mock(spec=DaemonComponents)
    components.config = config
    components.state = DaemonState()
    components.news_watcher = news_watcher
    components.social_watcher = social_watcher
    components.trump_watcher = None
    components.position_manager = None

    lifecycle = DaemonLifecycle(components)

    with patch("signal.signal"):  # Prevent global signal handler mutation
        await lifecycle.startup()

    # Let both run for a bit
    await asyncio.sleep(0.3)

    # Stop
    await lifecycle.shutdown()

    # Verify both ran independently
    assert len(news_cycles) > 0
    assert len(social_cycles) > 0
    # Social should have more cycles (faster interval)
    assert len(social_cycles) > len(news_cycles)


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.slow
async def test_shutdown_waits_then_cancels():
    """Test shutdown waits 5s then cancels slow watchers."""
    from src.daemon.config import DaemonConfig
    from src.daemon.factory import DaemonComponents
    from src.daemon.state import DaemonState

    # Create watcher with slow run cycle
    mock_watcher = Mock()

    async def slow_run():
        mock_watcher.running = True
        while mock_watcher.running:
            # Sleep longer than shutdown timeout
            await asyncio.sleep(10)

    mock_watcher.run = AsyncMock(side_effect=slow_run)

    config = DaemonConfig()
    config.database.enable_persistence = False
    config.api.enabled = False

    components = Mock(spec=DaemonComponents)
    components.config = config
    components.state = DaemonState()
    components.news_watcher = mock_watcher
    components.social_watcher = None
    components.trump_watcher = None
    components.position_manager = None

    lifecycle = DaemonLifecycle(components)

    with patch("signal.signal"):  # Prevent global signal handler mutation
        await lifecycle.startup()
    await asyncio.sleep(0.1)  # Let watcher start

    # Shutdown - should timeout and cancel
    import time

    start = time.time()
    await lifecycle.shutdown()
    elapsed = time.time() - start

    # Should take ~5s (timeout), not 10s
    assert 4.0 < elapsed < 6.0

    # Verify task was cancelled
    assert len(lifecycle._watcher_tasks) == 0
