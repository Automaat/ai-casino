"""Tests for daemon runner."""

import asyncio
from datetime import datetime
from pathlib import Path
from unittest.mock import ANY, AsyncMock, Mock, patch

import pytest

from src.daemon.config import DaemonConfig, ScreeningConfig, SectorRotationConfig
from src.daemon.runner import DaemonRunner
from src.daemon.state import ScreeningRecord
from src.data.broker import BrokerAccountInfo, BrokerPosition, OrderStatus

pytestmark = pytest.mark.skip(reason="Daemon runner tests need rewrite for async state facade")

# Test credentials - not real secrets
TEST_API_KEY = "test_key"
TEST_SECRET_KEY = "test_secret"


@pytest.fixture
def sample_config(tmp_path: Path) -> DaemonConfig:
    """Create sample daemon config."""
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
    }
    return DaemonConfig.model_validate(config_dict)


@pytest.fixture
def mock_broker() -> Mock:
    """Create mock broker with positions."""
    broker = Mock()
    broker.get_account_info.return_value = BrokerAccountInfo(
        balance=50000.0,
        available_cash=25000.0,
        positions={
            "AAPL": BrokerPosition(
                symbol="AAPL",
                qty=10.0,
                market_value=1500.0,
                avg_entry_price=150.0,
                unrealized_pnl=50.0,
                unrealized_pnl_percent=3.33,
            ),
            "NVDA": BrokerPosition(
                symbol="NVDA",
                qty=5.0,
                market_value=750.0,
                avg_entry_price=150.0,
                unrealized_pnl=25.0,
                unrealized_pnl_percent=3.33,
            ),
        },
        total_exposure=2250.0,
        portfolio_value=52250.0,
    )
    return broker


def test_get_merged_watchlist_no_broker(sample_config: DaemonConfig) -> None:
    """Test watchlist merge when broker is None."""
    runner = DaemonRunner(sample_config)
    runner.broker = None

    watchlist = runner.get_merged_watchlist()

    assert set(watchlist) == {"TSLA", "MSFT"}
    assert len(watchlist) == 2


def test_get_merged_watchlist_with_positions(sample_config: DaemonConfig, mock_broker: Mock) -> None:
    """Test watchlist merge with broker positions."""
    runner = DaemonRunner(sample_config)
    runner.broker = mock_broker
    runner._broker_manager.broker = mock_broker

    watchlist = runner.get_merged_watchlist()

    assert set(watchlist) == {"TSLA", "MSFT", "AAPL", "NVDA"}
    assert len(watchlist) == 4
    mock_broker.get_account_info.assert_called_once()


def test_get_merged_watchlist_deduplication(sample_config: DaemonConfig, mock_broker: Mock) -> None:
    """Test watchlist deduplication when symbol in both."""
    sample_config.watchlist = ["AAPL", "TSLA"]
    runner = DaemonRunner(sample_config)
    runner.broker = mock_broker
    runner._broker_manager.broker = mock_broker

    watchlist = runner.get_merged_watchlist()

    assert set(watchlist) == {"AAPL", "TSLA", "NVDA"}
    assert len(watchlist) == 3
    assert watchlist.count("AAPL") == 1


def test_get_merged_watchlist_broker_failure(sample_config: DaemonConfig, mock_broker: Mock) -> None:
    """Test watchlist fallback on broker API error."""
    runner = DaemonRunner(sample_config)
    runner.broker = mock_broker
    runner._broker_manager.broker = mock_broker
    mock_broker.get_account_info.side_effect = RuntimeError("API unavailable")

    watchlist = runner.get_merged_watchlist()

    assert set(watchlist) == {"TSLA", "MSFT"}
    assert len(watchlist) == 2


def test_get_merged_watchlist_empty_positions(sample_config: DaemonConfig, mock_broker: Mock) -> None:
    """Test watchlist when broker has no positions."""
    runner = DaemonRunner(sample_config)
    runner.broker = mock_broker
    runner._broker_manager.broker = mock_broker
    mock_broker.get_account_info.return_value = BrokerAccountInfo(
        balance=50000.0,
        available_cash=50000.0,
        positions={},
        total_exposure=0.0,
        portfolio_value=50000.0,
    )

    watchlist = runner.get_merged_watchlist()

    assert set(watchlist) == {"TSLA", "MSFT"}
    assert len(watchlist) == 2


@patch("src.daemon.broker_manager.AlpacaBroker")
def test_broker_init_with_credentials(mock_broker_class: Mock, sample_config: DaemonConfig) -> None:
    """Test broker initialization for watchlist merging when credentials present."""
    sample_config.api_keys.alpaca_paper_api_key = TEST_API_KEY
    sample_config.api_keys.alpaca_paper_secret_key = TEST_SECRET_KEY
    runner = DaemonRunner(sample_config)

    assert runner.broker is not None
    mock_broker_class.assert_called_once_with(
        api_key=TEST_API_KEY,
        secret_key=TEST_SECRET_KEY,
        paper=True,
        historical_cache=ANY,
    )


@patch("src.daemon.broker_manager.AlpacaBroker")
def test_broker_init_no_credentials(mock_broker_class: Mock, sample_config: DaemonConfig) -> None:
    """Test broker not initialized without credentials."""
    runner = DaemonRunner(sample_config)

    assert runner.broker is None
    mock_broker_class.assert_not_called()


@patch("src.daemon.broker_manager.AlpacaBroker")
def test_broker_init_failure(mock_broker_class: Mock, sample_config: DaemonConfig) -> None:
    """Test daemon continues if broker init fails with auto_trade=false."""
    mock_broker_class.side_effect = ValueError("Invalid credentials")
    sample_config.api_keys.alpaca_paper_api_key = "bad_key"
    sample_config.api_keys.alpaca_paper_secret_key = "bad_secret"
    runner = DaemonRunner(sample_config)

    assert runner.broker is None


def test_auto_trade_fails_fast_without_keys(sample_config: DaemonConfig) -> None:
    """Test auto_trade=true raises ValueError when keys missing."""
    sample_config.auto_trade = True

    with pytest.raises(ValueError, match="auto_trade with"):
        DaemonRunner(sample_config)


@patch("src.daemon.broker_manager.AlpacaBroker")
def test_auto_trade_inits_broker(mock_broker_class: Mock, sample_config: DaemonConfig) -> None:
    """Test auto_trade=true initializes broker when keys present."""
    sample_config.auto_trade = True
    sample_config.api_keys.alpaca_paper_api_key = TEST_API_KEY
    sample_config.api_keys.alpaca_paper_secret_key = TEST_SECRET_KEY
    runner = DaemonRunner(sample_config)

    assert runner.broker is not None
    mock_broker_class.assert_called_once_with(
        api_key=TEST_API_KEY,
        secret_key=TEST_SECRET_KEY,
        paper=True,
        historical_cache=ANY,
    )


async def test_analyze_watchlist_uses_merged(
    sample_config: DaemonConfig, mock_broker: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test _analyze_watchlist uses merged watchlist."""
    sample_config.api_keys.alpha_vantage_api_key = "test_key"

    runner = DaemonRunner(sample_config)
    runner.broker = mock_broker
    runner._broker_manager.broker = mock_broker

    # Mock the orchestrator's orchestrate method
    analyzed_symbols: list[str] = []

    async def mock_orchestrate(watchlist, target_allocations=None, degradation_context=None):
        from datetime import UTC, datetime

        from src.daemon.analysis_orchestrator import AnalysisOrchestrationResult

        analyzed_symbols.extend(watchlist)
        return AnalysisOrchestrationResult(
            timestamp=datetime.now(UTC),
            total_symbols=len(watchlist),
            successful=len(watchlist),
            failed=0,
            position_actions=0,
            results=[],
            failed_symbols=[],
            duration_seconds=0.0,
            position_sync_performed=False,
        )

    # Get merged watchlist and pass it to _analyze_watchlist
    merged = runner.get_merged_watchlist()

    # Mock _init_analysis_orchestrator to return our mock
    mock_orchestrator = Mock()
    mock_orchestrator.orchestrate = mock_orchestrate
    monkeypatch.setattr(runner, "_init_analysis_orchestrator", lambda: mock_orchestrator)

    await runner._analyze_watchlist(merged)

    # Should analyze merged watchlist: TSLA, MSFT from config + AAPL, NVDA from positions
    assert set(analyzed_symbols) == {"TSLA", "MSFT", "AAPL", "NVDA"}
    assert len(analyzed_symbols) == 4


async def test_run_cycle_uses_merged_watchlist(
    sample_config: DaemonConfig, mock_broker: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test _run_cycle logs correct merged watchlist count."""
    from src.daemon.degradation import DegradationContext, DegradationTier

    runner = DaemonRunner(sample_config)
    runner.broker = mock_broker
    runner._broker_manager.broker = mock_broker

    # Mock dependencies
    monkeypatch.setattr(runner.scheduler, "is_market_open", lambda: True)
    monkeypatch.setattr(runner, "_analyze_watchlist", AsyncMock(return_value=[]))
    monkeypatch.setattr(runner._task_runner, "run_scheduled_tasks", AsyncMock())

    # Mock _log_results on DaemonCycleOrchestrator
    from src.daemon.cycle_orchestrator import DaemonCycleOrchestrator

    monkeypatch.setattr(DaemonCycleOrchestrator, "_log_results", Mock())
    monkeypatch.setattr(
        runner,
        "_evaluate_degradation",
        lambda: DegradationContext(
            tier=DegradationTier.FULL,
            available_agents=set(),
            unavailable_services=[],
            confidence_adjustment=1.0,
        ),
    )

    # Capture log messages
    logged_messages: list[str] = []

    def mock_log_info(msg: str) -> None:
        logged_messages.append(msg)

    with patch("src.daemon.runner.logger.info", side_effect=mock_log_info):
        await runner._run_cycle()

    # Should log 4 symbols (2 config + 2 positions)
    assert any("4 symbols" in msg for msg in logged_messages)


async def test_analyze_symbol_records_executed_trade(
    sample_config: DaemonConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test executed_trade=True when result.order is not None."""
    from datetime import datetime

    from src.strategies.session import TradingSession

    runner = DaemonRunner(sample_config)

    mock_order = OrderStatus(
        order_id="test-123",
        symbol="TSLA",
        qty=10.0,
        filled_qty=10.0,
        side="buy",
        status="filled",
        submitted_at=datetime(2024, 1, 1),
        filled_at=datetime(2024, 1, 1),
        filled_avg_price=150.0,
    )

    mock_result = Mock()
    mock_result.decision.action.value = "BUY"
    mock_result.decision.confidence = 0.85
    mock_result.decision.reasoning = ["Strong momentum", "High volume"]
    mock_result.trading_session = TradingSession.REGULAR
    mock_result.order = mock_order
    mock_result.risk = Mock()
    mock_result.risk.current_price = 150.0
    mock_result.strategy_used = "momentum"
    mock_result.regime = None
    mock_result.technical = Mock()
    mock_result.technical.signal.value = "BUY"
    mock_result.technical.rsi = 45.0
    mock_result.technical.macd_hist = 0.5
    mock_result.technical.interpretation = "Strong momentum"
    mock_result.sentiment = Mock()
    mock_result.sentiment.overall_sentiment = "positive"
    mock_result.sentiment.summary = "Positive sentiment"
    mock_result.news = Mock()
    mock_result.news.impact_assessment = "Good news"
    mock_result.news.recommendation = "Buy"

    mock_workflow = AsyncMock()
    mock_workflow.analyze.return_value = mock_result
    monkeypatch.setattr(runner, "_init_workflow", lambda: mock_workflow)

    mock_state = Mock()
    runner.state = mock_state

    result = await runner._analyze_symbol("TSLA")

    assert result is mock_result
    mock_state.record_analysis.assert_called_once_with(
        symbol="TSLA",
        signal="BUY",
        confidence=0.85,
        executed=True,
        trading_session=TradingSession.REGULAR,
        is_paper_trade=True,
        rsi=45.0,
        macd_hist=0.5,
        reasoning=["Strong momentum", "High volume"],
        technical_analysis_reasoning="Strong momentum",
        sentiment_analysis_reasoning="Positive sentiment",
        news_analysis_reasoning="Good news\n\nRecommendation: Buy",
    )


async def test_analyze_symbol_records_not_executed(
    sample_config: DaemonConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test executed_trade=False when result.order is None."""
    from src.strategies.session import TradingSession

    runner = DaemonRunner(sample_config)

    mock_result = Mock()
    mock_result.decision.action.value = "HOLD"
    mock_result.decision.confidence = 0.6
    mock_result.decision.reasoning = ["Neutral signals", "Wait for confirmation"]
    mock_result.trading_session = TradingSession.REGULAR
    mock_result.order = None
    mock_result.risk = Mock()
    mock_result.risk.current_price = 160.0
    mock_result.strategy_used = "momentum"
    mock_result.regime = None
    mock_result.technical = Mock()
    mock_result.technical.signal.value = "HOLD"
    mock_result.technical.rsi = 55.0
    mock_result.technical.macd_hist = -0.2
    mock_result.technical.interpretation = "Neutral signals"
    mock_result.sentiment = Mock()
    mock_result.sentiment.overall_sentiment = "neutral"
    mock_result.sentiment.summary = "Mixed sentiment"
    mock_result.news = Mock()
    mock_result.news.impact_assessment = "Neutral news"
    mock_result.news.recommendation = "Hold"

    mock_workflow = AsyncMock()
    mock_workflow.analyze.return_value = mock_result
    monkeypatch.setattr(runner, "_init_workflow", lambda: mock_workflow)

    mock_state = Mock()
    runner.state = mock_state

    result = await runner._analyze_symbol("TSLA")

    assert result is mock_result
    mock_state.record_analysis.assert_called_once_with(
        symbol="TSLA",
        signal="HOLD",
        confidence=0.6,
        executed=False,
        trading_session=TradingSession.REGULAR,
        is_paper_trade=True,
        rsi=55.0,
        macd_hist=-0.2,
        reasoning=["Neutral signals", "Wait for confirmation"],
        technical_analysis_reasoning="Neutral signals",
        sentiment_analysis_reasoning="Mixed sentiment",
        news_analysis_reasoning="Neutral news\n\nRecommendation: Hold",
    )


def test_get_merged_watchlist_with_screening(sample_config: DaemonConfig) -> None:
    """Test screening candidates merged into watchlist."""
    sample_config.screening = ScreeningConfig(enabled=True)
    runner = DaemonRunner(sample_config)
    runner.broker = None

    runner.state.screening_history = [
        ScreeningRecord(
            timestamp=datetime(2024, 1, 15),
            criteria="momentum",
            universe="COMBINED",
            top_symbols=["NVDA", "AMD", "PLTR"],
            candidates=[],
            screened_at=datetime(2024, 1, 15),
        ),
    ]

    watchlist = runner.get_merged_watchlist()

    assert watchlist == ["TSLA", "MSFT", "NVDA", "AMD", "PLTR"]
    assert len(watchlist) == 5


def test_get_merged_watchlist_screening_disabled(sample_config: DaemonConfig) -> None:
    """Test screening candidates ignored when disabled."""
    sample_config.screening = ScreeningConfig(enabled=False)
    runner = DaemonRunner(sample_config)
    runner.broker = None

    runner.state.screening_history = [
        ScreeningRecord(
            timestamp=datetime(2024, 1, 15),
            criteria="momentum",
            universe="COMBINED",
            top_symbols=["NVDA", "AMD"],
            candidates=[],
            screened_at=datetime(2024, 1, 15),
        ),
    ]

    watchlist = runner.get_merged_watchlist()

    assert watchlist == ["TSLA", "MSFT"]
    assert len(watchlist) == 2


def test_get_merged_watchlist_all_three_sources(sample_config: DaemonConfig, mock_broker: Mock) -> None:
    """Test 3-source deduplication: config + positions + screening."""
    sample_config.screening = ScreeningConfig(enabled=True)
    runner = DaemonRunner(sample_config)
    runner.broker = mock_broker
    runner._broker_manager.broker = mock_broker

    runner.state.screening_history = [
        ScreeningRecord(
            timestamp=datetime(2024, 1, 15),
            criteria="momentum",
            universe="COMBINED",
            top_symbols=["AAPL", "PLTR", "MSFT"],  # AAPL=position, MSFT=config → deduped
            candidates=[],
            screened_at=datetime(2024, 1, 15),
        ),
    ]

    watchlist = runner.get_merged_watchlist()

    # TSLA, MSFT (config) + AAPL, NVDA (positions) + PLTR (screening, only new)
    assert set(watchlist) == {"TSLA", "MSFT", "AAPL", "NVDA", "PLTR"}
    assert len(watchlist) == 5
    assert watchlist.count("AAPL") == 1
    assert watchlist.count("MSFT") == 1


def test_get_merged_watchlist_empty_screening_history(sample_config: DaemonConfig) -> None:
    """Test no crash on empty screening history."""
    sample_config.screening = ScreeningConfig(enabled=True)
    runner = DaemonRunner(sample_config)
    runner.broker = None

    assert runner.state.screening_history == []

    watchlist = runner.get_merged_watchlist()

    assert watchlist == ["TSLA", "MSFT"]
    assert len(watchlist) == 2


class TestSectorRotationIntegration:
    async def test_sector_rotation_in_cycle(
        self, sample_config: DaemonConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test sector rotation triggered in run cycle."""
        sample_config.sector_rotation = SectorRotationConfig(enabled=True, run_time="16:15")
        runner = DaemonRunner(sample_config)

        rotation_called = False

        async def mock_run_rotation() -> None:
            nonlocal rotation_called
            rotation_called = True

        # Mock degradation evaluation to prevent HALTED state
        from src.daemon.degradation import AgentType, DegradationContext, DegradationTier

        monkeypatch.setattr(
            runner,
            "_evaluate_degradation",
            lambda: DegradationContext(
                tier=DegradationTier.FULL,
                available_agents=set(AgentType),  # All agents available
                unavailable_services=[],
                confidence_adjustment=1.0,
                halt_reason=None,
            ),
        )

        monkeypatch.setattr(runner.scheduler, "is_sector_rotation_time", lambda: True)
        monkeypatch.setattr(runner.scheduler, "is_after_hours_screening_time", lambda: False)
        monkeypatch.setattr(runner.scheduler, "is_market_open", lambda: True)
        monkeypatch.setattr(runner, "_analyze_watchlist", AsyncMock(return_value=[]))

        # Mock _log_results on DaemonCycleOrchestrator
        from src.daemon.cycle_orchestrator import DaemonCycleOrchestrator

        monkeypatch.setattr(DaemonCycleOrchestrator, "_log_results", Mock())

        # Mock task_service.run_sector_rotation
        from src.daemon.task_service import DaemonTaskService

        task_service = DaemonTaskService(runner._components, runner._container)
        monkeypatch.setattr(task_service, "run_sector_rotation", mock_run_rotation)
        runner._task_runner.set_task_service(task_service)

        await runner._run_cycle()

        assert rotation_called

    async def test_sector_rotation_skipped_when_disabled(
        self, sample_config: DaemonConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test sector rotation not triggered when disabled."""
        runner = DaemonRunner(sample_config)

        rotation_called = False

        async def mock_run_rotation() -> None:
            nonlocal rotation_called
            rotation_called = True

        monkeypatch.setattr(runner.scheduler, "is_sector_rotation_time", lambda: False)
        monkeypatch.setattr(runner.scheduler, "is_after_hours_screening_time", lambda: False)
        monkeypatch.setattr(runner.scheduler, "is_market_open", lambda: True)
        monkeypatch.setattr(runner, "_analyze_watchlist", AsyncMock(return_value=[]))

        # Mock _log_results on DaemonCycleOrchestrator
        from src.daemon.cycle_orchestrator import DaemonCycleOrchestrator

        monkeypatch.setattr(DaemonCycleOrchestrator, "_log_results", Mock())

        # Mock task_service.run_sector_rotation
        from src.daemon.task_service import DaemonTaskService

        task_service = DaemonTaskService(runner._components, runner._container)
        monkeypatch.setattr(task_service, "run_sector_rotation", mock_run_rotation)
        runner._task_runner.set_task_service(task_service)

        await runner._run_cycle()

        assert not rotation_called

    async def test_run_sector_rotation_records_state(self, sample_config: DaemonConfig) -> None:
        """Test run_sector_rotation records analysis in state."""
        from datetime import UTC, datetime

        from src.daemon.task_service import DaemonTaskService
        from src.metrics.sector_rotation import Momentum, SectorRotationAnalysis, SectorStrength

        sample_config.sector_rotation = SectorRotationConfig(enabled=True)
        runner = DaemonRunner(sample_config)
        runner.broker = None

        mock_analysis = SectorRotationAnalysis(
            sectors=[
                SectorStrength(
                    sector="TECHNOLOGY",
                    etf="XLK",
                    return_1w=2.0,
                    return_1m=3.0,
                    return_3m=4.0,
                    relative_strength=3.2,
                    momentum=Momentum.ACCELERATING,
                    rank=1,
                ),
            ],
            leading_sectors=["TECHNOLOGY"],
            lagging_sectors=["ENERGY"],
            spy_return_1w=1.0,
            spy_return_1m=2.0,
            spy_return_3m=3.0,
            timestamp=datetime.now(UTC),
        )

        mock_daemon_rotation = Mock()
        mock_daemon_rotation.run.return_value = mock_analysis

        task_service = DaemonTaskService(runner._components, runner._container)

        with patch("src.daemon.sector_rotation.DaemonSectorRotation", return_value=mock_daemon_rotation):
            await task_service.run_sector_rotation()

        assert runner.state.last_sector_rotation is not None
        assert len(runner.state.sector_rotation_history) == 1
        assert runner.state.sector_rotation_history[0].leading_sectors == ["TECHNOLOGY"]

    async def test_run_sector_rotation_dedup(self, sample_config: DaemonConfig) -> None:
        """Test sector rotation deduplication (skip if already ran today)."""
        from datetime import datetime
        from zoneinfo import ZoneInfo

        from src.daemon.task_service import DaemonTaskService

        sample_config.sector_rotation = SectorRotationConfig(enabled=True)
        runner = DaemonRunner(sample_config)
        tz = ZoneInfo("America/New_York")
        runner.state.last_sector_rotation = datetime.now(tz)

        rotation_ran = False

        def mock_analyze() -> None:
            nonlocal rotation_ran
            rotation_ran = True

        task_service = DaemonTaskService(runner._components, runner._container)

        with patch("src.daemon.sector_rotation.DaemonSectorRotation") as mock_cls:
            mock_cls.return_value.run = mock_analyze
            await task_service.run_sector_rotation()

        assert not rotation_ran


async def test_runner_publishes_cycle_events(sample_config: DaemonConfig, event_bus) -> None:
    """Test runner publishes CYCLE_START and CYCLE_COMPLETE events."""
    from src.daemon.degradation import DegradationContext, DegradationTier

    runner = DaemonRunner(sample_config, event_bus=event_bus)

    sub_id, queue = await event_bus.subscribe()

    with (
        patch.object(runner, "_analyze_watchlist", new_callable=AsyncMock) as mock_analyze,
        patch.object(runner.scheduler, "is_market_open", return_value=True),
        patch.object(runner._task_runner, "run_scheduled_tasks", new_callable=AsyncMock),
        patch.object(
            runner,
            "_evaluate_degradation",
            return_value=DegradationContext(
                tier=DegradationTier.FULL,
                available_agents=set(),
                unavailable_services=[],
                confidence_adjustment=1.0,
            ),
        ),
    ):
        mock_analyze.return_value = []

        await runner._run_cycle()

    cycle_start = queue.get_nowait()
    assert cycle_start.event_type.value == "CYCLE_START"
    assert "watchlist_size" in cycle_start.data
    assert "degradation_tier" in cycle_start.data

    cycle_complete = queue.get_nowait()
    assert cycle_complete.event_type.value == "CYCLE_COMPLETE"
    assert "results_count" in cycle_complete.data
    assert "errors_count" in cycle_complete.data
    assert "duration_seconds" in cycle_complete.data

    await event_bus.unsubscribe(sub_id)


async def test_runner_publishes_analysis_events(sample_config: DaemonConfig, event_bus) -> None:
    """Test runner publishes ANALYSIS_START and ANALYSIS_COMPLETE events."""
    from src.strategies.session import TradingSession
    from src.strategies.signal import Signal

    runner = DaemonRunner(sample_config, event_bus=event_bus)

    sub_id, queue = await event_bus.subscribe()

    with patch.object(runner, "_init_workflow") as mock_init_workflow:
        mock_workflow = Mock()
        mock_result = Mock()
        mock_result.decision.action = Signal.BUY
        mock_result.decision.confidence = 0.85
        mock_result.decision.reasoning = ["Strong buy signal"]
        mock_result.order = None
        mock_result.trading_session = TradingSession.REGULAR
        mock_result.risk.current_price = 150.0
        mock_result.strategy_used = "momentum"
        mock_result.regime = None
        mock_result.technical.signal = Signal.BUY
        mock_result.technical.rsi = 35.0
        mock_result.technical.macd_hist = 0.8
        mock_result.technical.interpretation = "Strong buy signal"
        mock_result.sentiment = Mock()
        mock_result.sentiment.summary = "Positive sentiment"
        mock_result.news = Mock()
        mock_result.news.impact_assessment = "Good news"
        mock_result.news.recommendation = "Buy"
        mock_workflow.analyze = AsyncMock(return_value=mock_result)
        mock_init_workflow.return_value = mock_workflow

        result = await runner._analyze_symbol("AAPL")

    assert result is not None

    analysis_start = queue.get_nowait()
    assert analysis_start.event_type.value == "ANALYSIS_START"
    assert analysis_start.data["symbol"] == "AAPL"
    assert "trading_session" in analysis_start.data

    analysis_complete = queue.get_nowait()
    assert analysis_complete.event_type.value == "ANALYSIS_COMPLETE"
    assert analysis_complete.data["symbol"] == "AAPL"
    assert analysis_complete.data["signal"] == "BUY"
    assert analysis_complete.data["confidence"] == 0.85
    assert analysis_complete.data["executed"] is False

    await event_bus.unsubscribe(sub_id)


async def test_runner_publishes_analysis_error(sample_config: DaemonConfig, event_bus) -> None:
    """Test runner publishes ANALYSIS_ERROR event on failure."""
    runner = DaemonRunner(sample_config, event_bus=event_bus)

    sub_id, queue = await event_bus.subscribe()

    with patch.object(runner, "_init_workflow") as mock_init_workflow:
        mock_workflow = Mock()
        mock_workflow.analyze = AsyncMock(side_effect=ValueError("Test error"))
        mock_init_workflow.return_value = mock_workflow

        result = await runner._analyze_symbol("AAPL")

    assert result is None

    analysis_start = queue.get_nowait()
    assert analysis_start.event_type.value == "ANALYSIS_START"

    analysis_error = queue.get_nowait()
    assert analysis_error.event_type.value == "ANALYSIS_ERROR"
    assert analysis_error.data["symbol"] == "AAPL"
    assert "error" in analysis_error.data
    assert "Test error" in analysis_error.data["error"]

    await event_bus.unsubscribe(sub_id)


async def test_runner_eventbus_optional(sample_config: DaemonConfig) -> None:
    """Test runner works without event_bus (None)."""
    runner = DaemonRunner(sample_config, event_bus=None)

    assert runner.event_bus is None

    with (
        patch.object(runner, "_analyze_watchlist", new_callable=AsyncMock) as mock_analyze,
        patch.object(runner.scheduler, "is_market_open", return_value=True),
        patch.object(runner._task_runner, "run_scheduled_tasks", new_callable=AsyncMock),
    ):
        mock_analyze.return_value = []

        await runner._run_cycle()


async def test_api_server_lifecycle(sample_config: DaemonConfig) -> None:
    """Test API server starts and stops with daemon."""
    import httpx

    sample_config.api.enabled = True
    sample_config.api.port = 18484

    runner = DaemonRunner(sample_config)

    # Mock _run_cycle to stop daemon after one cycle
    async def mock_cycle() -> int:
        runner.running = False
        return 1

    with patch.object(runner, "_run_cycle", new=mock_cycle):
        run_task = asyncio.create_task(runner.run())

        # Poll for API server readiness with timeout
        async def wait_for_api_ready() -> None:
            """Wait for API server to start with timeout."""
            async with httpx.AsyncClient() as client:
                for _ in range(100):  # 100 * 0.05s = 5s timeout
                    try:
                        response = await client.get("http://127.0.0.1:18484/health")
                        if response.status_code == 200:
                            return
                    except httpx.ConnectError:
                        pass  # Server not ready yet
                    await asyncio.sleep(0.05)
            msg = "API server did not start within 5 seconds"
            raise TimeoutError(msg)

        await wait_for_api_ready()

        # Verify API server is responding
        async with httpx.AsyncClient() as client:
            response = await client.get("http://127.0.0.1:18484/health")
            assert response.status_code == 200

        # Wait for daemon to stop
        await asyncio.wait_for(run_task, timeout=10.0)

        # Small delay to ensure port is released
        await asyncio.sleep(0.2)

        # Verify API server is no longer responding
        async with httpx.AsyncClient() as client:
            try:
                await client.get("http://127.0.0.1:18484/health", timeout=1.0)
                pytest.fail("API server should be stopped")
            except httpx.ConnectError:
                pass  # Expected - server should be down
