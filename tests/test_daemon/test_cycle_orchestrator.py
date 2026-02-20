"""Tests for daemon cycle orchestrator coordinator routing."""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from src.daemon.config import DaemonConfig
from src.daemon.cycle_orchestrator import CycleResult, DaemonCycleOrchestrator
from src.daemon.degradation import DegradationContext, DegradationTier
from src.daemon.factory import DaemonFactory
from src.strategies.session import TradingSession
from src.v1.coordinator.models import CoordinatorCycleResult

pytestmark = pytest.mark.skip(reason="Cycle orchestrator tests need rewrite for async state")


@pytest.fixture
def sample_config_coordinator_enabled(tmp_path: Path) -> DaemonConfig:
    """Create sample daemon config with coordinator enabled."""
    config_dict = {
        "watchlist": ["TSLA", "MSFT"],
        "interval_minutes": 30,
        "market_hours_only": True,
        "auto_trade": False,
        "max_concurrent_analyses": 5,
        "schedule": {
            "start_time": "09:30",
            "end_time": "16:00",
            "timezone": "America/New_York",
            "enable_pre_market": False,
        },
        "state": {
            "state_file": str(tmp_path / "daemon_state.json"),
        },
        "database": {
            "enable_persistence": False,
        },
        "coordinator": {
            "enabled": True,
            "max_tool_calls": 25,
            "temperature": 0.5,
            "confirmation_mode": "auto",
            "cycle_timeout_seconds": 600,
            "max_daily_trades": 10,
            "max_position_pct": 10.0,
            "min_confidence_to_trade": 0.6,
        },
    }
    return DaemonConfig.model_validate(config_dict)


@pytest.fixture
def sample_config_coordinator_disabled(tmp_path: Path) -> DaemonConfig:
    """Create sample daemon config with coordinator disabled."""
    config_dict = {
        "watchlist": ["TSLA", "MSFT"],
        "interval_minutes": 30,
        "market_hours_only": True,
        "auto_trade": False,
        "max_concurrent_analyses": 5,
        "schedule": {
            "start_time": "09:30",
            "end_time": "16:00",
            "timezone": "America/New_York",
            "enable_pre_market": False,
        },
        "state": {
            "state_file": str(tmp_path / "daemon_state.json"),
        },
        "database": {
            "enable_persistence": False,
        },
        "coordinator": {
            "enabled": False,
        },
    }
    return DaemonConfig.model_validate(config_dict)


@pytest.fixture
def mock_degradation_context() -> DegradationContext:
    """Create mock degradation context."""
    return DegradationContext(
        tier=DegradationTier.NONE,
        available_agents=set(),
        unavailable_services=[],
        confidence_adjustment=1.0,
    )


async def test_coordinator_routing_when_enabled(
    sample_config_coordinator_enabled: DaemonConfig,
    mock_degradation_context: DegradationContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that cycle orchestrator routes to coordinator when enabled."""
    factory = DaemonFactory(sample_config_coordinator_enabled)
    components = factory.create_components()

    # Mock market hours check
    monkeypatch.setattr(components.scheduler, "is_market_open", lambda: True)
    monkeypatch.setattr(components.task_runner, "run_scheduled_tasks", AsyncMock())

    # Mock coordinator cycle to return successful result
    mock_coordinator_result = CoordinatorCycleResult(
        summary="Analyzed 2 symbols, executed 1 trade",
        symbols_analyzed=["TSLA", "MSFT"],
        trades_proposed=1,
        trades_executed=1,
        tool_calls_made=5,
        game_plan_generated=True,
    )

    # Create orchestrator and mock its methods
    orchestrator = DaemonCycleOrchestrator(
        components=components,
        task_runner=components.task_runner,
        factory=factory,
        profiler=None,
    )

    monkeypatch.setattr(orchestrator, "_evaluate_degradation", lambda: mock_degradation_context)
    mock_run_coordinator = AsyncMock(return_value=mock_coordinator_result)
    monkeypatch.setattr(orchestrator, "_run_coordinator_cycle_impl", mock_run_coordinator)

    result = await orchestrator.run_cycle()

    # Verify coordinator was called
    mock_run_coordinator.assert_called_once()
    call_args = mock_run_coordinator.call_args
    assert call_args[0][0] == ["TSLA", "MSFT"]  # watchlist
    assert call_args[0][1] == mock_degradation_context
    assert isinstance(call_args[0][2], TradingSession)

    # Verify result
    assert isinstance(result, CycleResult)
    assert result.analysis_performed is True
    assert result.halted is False
    assert result.results_count == 2  # symbols_analyzed count


async def test_coordinator_routing_when_disabled(
    sample_config_coordinator_disabled: DaemonConfig,
    mock_degradation_context: DegradationContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that cycle orchestrator uses legacy cycle when coordinator disabled."""
    factory = DaemonFactory(sample_config_coordinator_disabled)
    components = factory.create_components()

    # Mock market hours check
    monkeypatch.setattr(components.scheduler, "is_market_open", lambda: True)
    monkeypatch.setattr(components.task_runner, "run_scheduled_tasks", AsyncMock())

    # Create orchestrator and mock its methods
    orchestrator = DaemonCycleOrchestrator(
        components=components,
        task_runner=components.task_runner,
        factory=factory,
        profiler=None,
    )

    monkeypatch.setattr(orchestrator, "_evaluate_degradation", lambda: mock_degradation_context)

    # Mock legacy cycle
    mock_analyze_watchlist = AsyncMock(return_value=[])
    monkeypatch.setattr(orchestrator, "_analyze_watchlist", mock_analyze_watchlist)

    # Ensure coordinator is NOT called
    mock_run_coordinator = AsyncMock()
    monkeypatch.setattr(orchestrator, "_run_coordinator_cycle_impl", mock_run_coordinator)

    result = await orchestrator.run_cycle()

    # Verify coordinator was NOT called
    mock_run_coordinator.assert_not_called()

    # Verify legacy cycle was used
    mock_analyze_watchlist.assert_called_once()

    # Verify result
    assert isinstance(result, CycleResult)
    assert result.analysis_performed is True


async def test_coordinator_fallback_on_exception(
    sample_config_coordinator_enabled: DaemonConfig,
    mock_degradation_context: DegradationContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that coordinator failures fall back to legacy cycle."""
    factory = DaemonFactory(sample_config_coordinator_enabled)
    components = factory.create_components()

    # Mock market hours check
    monkeypatch.setattr(components.scheduler, "is_market_open", lambda: True)
    monkeypatch.setattr(components.task_runner, "run_scheduled_tasks", AsyncMock())

    # Create orchestrator and mock its methods
    orchestrator = DaemonCycleOrchestrator(
        components=components,
        task_runner=components.task_runner,
        factory=factory,
        profiler=None,
    )

    monkeypatch.setattr(orchestrator, "_evaluate_degradation", lambda: mock_degradation_context)

    # Mock coordinator to raise exception
    mock_run_coordinator = AsyncMock(side_effect=RuntimeError("Coordinator init failed"))
    monkeypatch.setattr(orchestrator, "_run_coordinator_cycle_impl", mock_run_coordinator)

    # Mock legacy cycle
    mock_analyze_watchlist = AsyncMock(return_value=[])
    monkeypatch.setattr(orchestrator, "_analyze_watchlist", mock_analyze_watchlist)

    result = await orchestrator.run_cycle()

    # Verify coordinator was attempted
    mock_run_coordinator.assert_called_once()

    # Verify fallback to legacy cycle occurred
    mock_analyze_watchlist.assert_called_once()

    # Verify result is successful despite coordinator failure
    assert isinstance(result, CycleResult)
    assert result.analysis_performed is True
    assert result.halted is False


async def test_coordinator_fallback_on_various_exceptions(
    sample_config_coordinator_enabled: DaemonConfig,
    mock_degradation_context: DegradationContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that all exception types trigger fallback, not just ValueError."""
    factory = DaemonFactory(sample_config_coordinator_enabled)
    components = factory.create_components()

    # Mock market hours check
    monkeypatch.setattr(components.scheduler, "is_market_open", lambda: True)
    monkeypatch.setattr(components.task_runner, "run_scheduled_tasks", AsyncMock())

    # Test different exception types
    exception_types = [
        ValueError("Value error"),
        RuntimeError("Runtime error"),
        KeyError("Key error"),
        AttributeError("Attribute error"),
    ]

    for exc in exception_types:
        # Create orchestrator and mock its methods
        orchestrator = DaemonCycleOrchestrator(
            components=components,
            task_runner=components.task_runner,
            factory=factory,
            profiler=None,
        )

        monkeypatch.setattr(orchestrator, "_evaluate_degradation", lambda: mock_degradation_context)

        # Mock coordinator to raise specific exception
        mock_run_coordinator = AsyncMock(side_effect=exc)
        monkeypatch.setattr(orchestrator, "_run_coordinator_cycle_impl", mock_run_coordinator)

        # Mock legacy cycle
        mock_analyze_watchlist = AsyncMock(return_value=[])
        monkeypatch.setattr(orchestrator, "_analyze_watchlist", mock_analyze_watchlist)

        result = await orchestrator.run_cycle()

        # Verify fallback occurred for this exception type
        mock_analyze_watchlist.assert_called_once()
        assert result.analysis_performed is True
