"""Tests for analysis orchestrator."""

from unittest.mock import AsyncMock, Mock

import pytest

from src.daemon.analysis_orchestrator import (
    AnalysisOrchestrationResult,
    AnalysisOrchestrator,
    AnalysisOrchestratorConfig,
)
from src.daemon.factory import DaemonComponents
from src.daemon.state import DaemonState
from src.strategies.session import TradingSession
from src.workflows.types import TradingDecision, TradingWorkflowResult

pytestmark = pytest.mark.skip(reason="Analysis orchestrator tests need rewrite for async state")


@pytest.fixture
def mock_workflow():
    """Create mock workflow."""
    workflow = Mock()
    workflow.analyze = AsyncMock()
    return workflow


@pytest.fixture
def mock_state(tmp_path):
    """Create mock daemon state."""
    return DaemonState.load(str(tmp_path / "state.json"))


@pytest.fixture
def mock_scheduler():
    """Create mock scheduler."""
    scheduler = Mock()
    scheduler.get_trading_session.return_value = TradingSession.REGULAR
    return scheduler


@pytest.fixture
def mock_position_manager():
    """Create mock position manager."""
    manager = Mock()
    manager.sync_with_broker.return_value = ([], [], [])
    manager.review_position.return_value = []
    return manager


@pytest.fixture
def mock_broker():
    """Create mock broker."""
    broker = Mock()
    account_info = Mock()
    account_info.positions = {}
    broker.get_account_info.return_value = account_info
    return broker


@pytest.fixture
def mock_components(mock_workflow, mock_state, mock_scheduler):
    """Create mock DaemonComponents."""
    components = Mock(spec=DaemonComponents)
    components.workflow = mock_workflow
    components.state = mock_state
    components.scheduler = mock_scheduler
    components.broker = None
    components.position_manager = None
    components.event_bus = None
    components.historical_cache = None
    components.notification_service = None
    return components


async def test_orchestrator_basic_flow(mock_components):
    """Test basic orchestration flow."""
    config = AnalysisOrchestratorConfig()
    orchestrator = AnalysisOrchestrator(
        config=config,
        components=mock_components,
    )

    # Mock workflow result
    mock_result = Mock(spec=TradingWorkflowResult)
    mock_result.symbol = "AAPL"
    mock_result.decision = Mock(spec=TradingDecision)
    mock_result.decision.action = Mock(value="BUY")
    mock_result.decision.confidence = 0.8
    mock_result.decision.reasoning = ["Strong technical signals"]
    mock_result.order = None
    mock_result.trading_session = TradingSession.REGULAR
    mock_result.technical = None
    mock_result.sentiment = None
    mock_result.news = None
    mock_result.risk = Mock(current_price=150.0)
    mock_result.regime = None
    mock_result.strategy_used = "momentum"

    mock_components.workflow.analyze.return_value = mock_result

    result = await orchestrator.orchestrate(["AAPL"])

    assert isinstance(result, AnalysisOrchestrationResult)
    assert result.total_symbols == 1
    assert result.successful == 1
    assert result.failed == 0
    assert len(result.results) == 1
    assert result.results[0].symbol == "AAPL"


async def test_orchestrator_position_sync(mock_components, mock_position_manager, mock_broker):
    """Test position syncing is called when enabled."""
    config = AnalysisOrchestratorConfig(enable_position_sync=True)
    mock_components.broker = mock_broker
    mock_components.position_manager = mock_position_manager
    orchestrator = AnalysisOrchestrator(
        config=config,
        components=mock_components,
    )

    # Mock workflow result
    mock_result = Mock(spec=TradingWorkflowResult)
    mock_result.symbol = "AAPL"
    mock_result.decision = Mock(spec=TradingDecision)
    mock_result.decision.action = Mock(value="HOLD")
    mock_result.decision.confidence = 0.7
    mock_result.decision.reasoning = ["Neutral signals"]
    mock_result.order = None
    mock_result.trading_session = TradingSession.REGULAR
    mock_result.technical = None
    mock_result.sentiment = None
    mock_result.news = None
    mock_result.risk = Mock(current_price=105.0)
    mock_result.regime = None
    mock_result.strategy_used = "momentum"

    mock_components.workflow.analyze.return_value = mock_result

    result = await orchestrator.orchestrate(["AAPL"])

    assert result.position_sync_performed is True
    mock_position_manager.sync_with_broker.assert_called_once()


async def test_orchestrator_concurrent_limit(mock_components):
    """Test concurrent analysis respects semaphore limit."""
    config = AnalysisOrchestratorConfig(max_concurrent_analyses=2)
    orchestrator = AnalysisOrchestrator(
        config=config,
        components=mock_components,
    )

    # Track concurrent executions
    concurrent_count = 0
    max_concurrent = 0

    async def track_concurrent(*args, **kwargs):
        nonlocal concurrent_count, max_concurrent
        concurrent_count += 1
        max_concurrent = max(max_concurrent, concurrent_count)
        import asyncio

        await asyncio.sleep(0.01)
        concurrent_count -= 1

        result = Mock(spec=TradingWorkflowResult)
        result.symbol = args[0]
        result.decision = Mock(spec=TradingDecision)
        result.decision.action = Mock(value="HOLD")
        result.decision.confidence = 0.7
        result.decision.reasoning = ["Neutral signals"]
        result.order = None
        result.trading_session = TradingSession.REGULAR
        result.technical = None
        result.sentiment = None
        result.news = None
        result.risk = Mock(current_price=100.0)
        result.regime = None
        result.strategy_used = "momentum"
        return result

    mock_components.workflow.analyze.side_effect = track_concurrent

    result = await orchestrator.orchestrate(["AAPL", "TSLA", "MSFT", "GOOGL"])

    assert result.successful == 4
    assert max_concurrent <= 2


async def test_orchestrator_handles_failures(mock_components):
    """Test orchestrator handles analysis failures."""
    config = AnalysisOrchestratorConfig()
    orchestrator = AnalysisOrchestrator(
        config=config,
        components=mock_components,
    )

    # Mock workflow to fail for one symbol
    async def mock_analyze(symbol, *args, **kwargs):
        if symbol == "FAIL":
            msg = "Intentional failure"
            raise ValueError(msg)

        result = Mock(spec=TradingWorkflowResult)
        result.symbol = symbol
        result.decision = Mock(spec=TradingDecision)
        result.decision.action = Mock(value="HOLD")
        result.decision.confidence = 0.7
        result.decision.reasoning = ["Neutral signals"]
        result.order = None
        result.trading_session = TradingSession.REGULAR
        result.technical = None
        result.sentiment = None
        result.news = None
        result.risk = Mock(current_price=100.0)
        result.regime = None
        result.strategy_used = "momentum"
        return result

    mock_components.workflow.analyze.side_effect = mock_analyze

    result = await orchestrator.orchestrate(["AAPL", "FAIL", "TSLA"])

    assert result.total_symbols == 3
    assert result.successful == 2
    assert result.failed == 1
    assert "FAIL" in result.failed_symbols


async def test_orchestrator_target_allocations(mock_components):
    """Test orchestrator sets target allocations via workflow method."""
    config = AnalysisOrchestratorConfig()
    orchestrator = AnalysisOrchestrator(
        config=config,
        components=mock_components,
    )

    target_allocations = {"AAPL": 0.3, "TSLA": 0.2}

    # Mock workflow result
    mock_result = Mock(spec=TradingWorkflowResult)
    mock_result.symbol = "AAPL"
    mock_result.decision = Mock(spec=TradingDecision)
    mock_result.decision.action = Mock(value="BUY")
    mock_result.decision.confidence = 0.8
    mock_result.decision.reasoning = ["Strong technical signals"]
    mock_result.order = None
    mock_result.trading_session = TradingSession.REGULAR
    mock_result.technical = None
    mock_result.sentiment = None
    mock_result.news = None
    mock_result.risk = Mock(current_price=150.0)
    mock_result.regime = None
    mock_result.strategy_used = "momentum"

    mock_components.workflow.analyze.return_value = mock_result
    mock_components.workflow.set_target_allocations = Mock()

    await orchestrator.orchestrate(["AAPL"], target_allocations=target_allocations)

    # Verify set_target_allocations was called before analyze
    assert mock_components.workflow.set_target_allocations.call_count == 2
    mock_components.workflow.set_target_allocations.assert_any_call(target_allocations)
    mock_components.workflow.set_target_allocations.assert_any_call(None)


async def test_orchestrator_removes_invalid_discovery_candidate_on_no_data(mock_components):
    """Test ValueError('No data returned') triggers removal from active_discovery_candidates."""
    config = AnalysisOrchestratorConfig()
    mock_components.container = None  # skips DB removal path, tests broker_manager removal
    mock_components.broker_manager = Mock()
    mock_components.broker_manager.config = Mock()
    mock_components.economic_calendar_watcher = None
    mock_components.options_flow_watcher = None
    mock_components.social_sentiment_watcher = None

    orchestrator = AnalysisOrchestrator(config=config, components=mock_components)

    async def mock_analyze(symbol, *args, **kwargs):
        raise ValueError(f"No data returned for {symbol}")

    mock_components.workflow.analyze.side_effect = mock_analyze

    result = await orchestrator.orchestrate(["BADTICKER"])

    assert result.failed == 1
    assert "BADTICKER" in result.failed_symbols
    mock_components.broker_manager.config.remove_watchlist_symbol.assert_called_once_with("BADTICKER")
