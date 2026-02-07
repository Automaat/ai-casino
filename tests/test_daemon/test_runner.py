"""Tests for daemon runner."""

import os
from datetime import datetime
from pathlib import Path
from unittest.mock import ANY, AsyncMock, Mock, patch

import pytest

from src.daemon.config import DaemonConfig, ScreeningConfig, SectorRotationConfig
from src.daemon.runner import DaemonRunner
from src.daemon.state import ScreeningRecord
from src.data.broker import BrokerAccountInfo, BrokerPosition, OrderStatus


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

    watchlist = runner._get_merged_watchlist()

    assert set(watchlist) == {"TSLA", "MSFT"}
    assert len(watchlist) == 2


def test_get_merged_watchlist_with_positions(sample_config: DaemonConfig, mock_broker: Mock) -> None:
    """Test watchlist merge with broker positions."""
    runner = DaemonRunner(sample_config)
    runner.broker = mock_broker

    watchlist = runner._get_merged_watchlist()

    assert set(watchlist) == {"TSLA", "MSFT", "AAPL", "NVDA"}
    assert len(watchlist) == 4
    mock_broker.get_account_info.assert_called_once()


def test_get_merged_watchlist_deduplication(sample_config: DaemonConfig, mock_broker: Mock) -> None:
    """Test watchlist deduplication when symbol in both."""
    sample_config.watchlist = ["AAPL", "TSLA"]
    runner = DaemonRunner(sample_config)
    runner.broker = mock_broker

    watchlist = runner._get_merged_watchlist()

    assert set(watchlist) == {"AAPL", "TSLA", "NVDA"}
    assert len(watchlist) == 3
    assert watchlist.count("AAPL") == 1


def test_get_merged_watchlist_broker_failure(sample_config: DaemonConfig, mock_broker: Mock) -> None:
    """Test watchlist fallback on broker API error."""
    runner = DaemonRunner(sample_config)
    runner.broker = mock_broker
    mock_broker.get_account_info.side_effect = RuntimeError("API unavailable")

    watchlist = runner._get_merged_watchlist()

    assert set(watchlist) == {"TSLA", "MSFT"}
    assert len(watchlist) == 2


def test_get_merged_watchlist_empty_positions(sample_config: DaemonConfig, mock_broker: Mock) -> None:
    """Test watchlist when broker has no positions."""
    runner = DaemonRunner(sample_config)
    runner.broker = mock_broker
    mock_broker.get_account_info.return_value = BrokerAccountInfo(
        balance=50000.0,
        available_cash=50000.0,
        positions={},
        total_exposure=0.0,
        portfolio_value=50000.0,
    )

    watchlist = runner._get_merged_watchlist()

    assert set(watchlist) == {"TSLA", "MSFT"}
    assert len(watchlist) == 2


@patch("src.daemon.runner.AlpacaBroker")
def test_broker_init_with_credentials(mock_broker_class: Mock, sample_config: DaemonConfig) -> None:
    """Test broker initialization for watchlist merging when credentials present."""
    with patch.dict(os.environ, {"ALPACA_API_KEY": "test_key", "ALPACA_SECRET_KEY": "test_secret"}):
        runner = DaemonRunner(sample_config)

        assert runner.broker is not None
        mock_broker_class.assert_called_once_with(paper=True, historical_cache=ANY)


@patch("src.daemon.runner.AlpacaBroker")
def test_broker_init_no_credentials(mock_broker_class: Mock, sample_config: DaemonConfig) -> None:
    """Test broker not initialized without credentials."""
    with patch.dict(os.environ, {}, clear=True):
        runner = DaemonRunner(sample_config)

        assert runner.broker is None
        mock_broker_class.assert_not_called()


@patch("src.daemon.runner.AlpacaBroker")
def test_broker_init_failure(mock_broker_class: Mock, sample_config: DaemonConfig) -> None:
    """Test daemon continues if broker init fails with auto_trade=false."""
    mock_broker_class.side_effect = ValueError("Invalid credentials")

    with patch.dict(os.environ, {"ALPACA_API_KEY": "bad_key", "ALPACA_SECRET_KEY": "bad_secret"}):
        runner = DaemonRunner(sample_config)

        assert runner.broker is None


def test_auto_trade_fails_fast_without_keys(sample_config: DaemonConfig) -> None:
    """Test auto_trade=true raises ValueError when keys missing."""
    sample_config.auto_trade = True

    with patch.dict(os.environ, {}, clear=True), pytest.raises(ValueError, match="auto_trade=true requires"):
        DaemonRunner(sample_config)


@patch("src.daemon.runner.AlpacaBroker")
def test_auto_trade_inits_broker(mock_broker_class: Mock, sample_config: DaemonConfig) -> None:
    """Test auto_trade=true initializes broker when keys present."""
    sample_config.auto_trade = True

    with patch.dict(os.environ, {"ALPACA_API_KEY": "test_key", "ALPACA_SECRET_KEY": "test_secret"}):
        runner = DaemonRunner(sample_config)

        assert runner.broker is not None
        mock_broker_class.assert_called_once_with(paper=True, historical_cache=ANY)


@pytest.mark.asyncio
async def test_analyze_watchlist_uses_merged(
    sample_config: DaemonConfig, mock_broker: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test _analyze_watchlist uses merged watchlist."""
    runner = DaemonRunner(sample_config)
    runner.broker = mock_broker

    # Mock _analyze_symbol to track which symbols are analyzed
    analyzed_symbols: list[str] = []

    async def mock_analyze(symbol: str) -> None:
        analyzed_symbols.append(symbol)

    monkeypatch.setattr(runner, "_analyze_symbol", mock_analyze)

    # Get merged watchlist and pass it to _analyze_watchlist
    merged = runner._get_merged_watchlist()
    await runner._analyze_watchlist(merged)

    # Should analyze merged watchlist: TSLA, MSFT from config + AAPL, NVDA from positions
    assert set(analyzed_symbols) == {"TSLA", "MSFT", "AAPL", "NVDA"}
    assert len(analyzed_symbols) == 4


@pytest.mark.asyncio
async def test_run_cycle_uses_merged_watchlist(
    sample_config: DaemonConfig, mock_broker: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test _run_cycle logs correct merged watchlist count."""
    runner = DaemonRunner(sample_config)
    runner.broker = mock_broker

    # Mock dependencies
    monkeypatch.setattr(runner.scheduler, "is_market_open", lambda: True)
    monkeypatch.setattr(runner, "_analyze_watchlist", AsyncMock(return_value=[]))
    monkeypatch.setattr(runner, "_log_results", Mock())

    # Capture log messages
    logged_messages: list[str] = []

    def mock_log_info(msg: str) -> None:
        logged_messages.append(msg)

    with patch("src.daemon.runner.logger.info", side_effect=mock_log_info):
        await runner._run_cycle()

    # Should log 4 symbols (2 config + 2 positions)
    assert any("4 symbols" in msg for msg in logged_messages)


@pytest.mark.asyncio
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
    mock_result.trading_session = TradingSession.REGULAR
    mock_result.order = mock_order

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
        trading_session="REGULAR",
    )


@pytest.mark.asyncio
async def test_analyze_symbol_records_not_executed(
    sample_config: DaemonConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test executed_trade=False when result.order is None."""
    from src.strategies.session import TradingSession

    runner = DaemonRunner(sample_config)

    mock_result = Mock()
    mock_result.decision.action.value = "HOLD"
    mock_result.decision.confidence = 0.6
    mock_result.trading_session = TradingSession.REGULAR
    mock_result.order = None

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
        trading_session="REGULAR",
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

    watchlist = runner._get_merged_watchlist()

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

    watchlist = runner._get_merged_watchlist()

    assert watchlist == ["TSLA", "MSFT"]
    assert len(watchlist) == 2


def test_get_merged_watchlist_all_three_sources(sample_config: DaemonConfig, mock_broker: Mock) -> None:
    """Test 3-source deduplication: config + positions + screening."""
    sample_config.screening = ScreeningConfig(enabled=True)
    runner = DaemonRunner(sample_config)
    runner.broker = mock_broker

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

    watchlist = runner._get_merged_watchlist()

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

    watchlist = runner._get_merged_watchlist()

    assert watchlist == ["TSLA", "MSFT"]
    assert len(watchlist) == 2


class TestHealthCheckIntegration:
    @pytest.mark.asyncio
    async def test_health_check_disabled(self, sample_config: DaemonConfig) -> None:
        """Test health check skipped when disabled."""
        sample_config.health.enabled = False
        runner = DaemonRunner(sample_config)

        with patch("src.daemon.health.HealthChecker") as mock_checker:
            await runner._maybe_run_health_check()
            mock_checker.assert_not_called()

    @pytest.mark.asyncio
    async def test_health_check_wrong_time(
        self, sample_config: DaemonConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test health check skipped when not at scheduled time."""
        runner = DaemonRunner(sample_config)
        monkeypatch.setattr(runner.scheduler, "is_health_check_time", lambda _t: False)

        with patch("src.daemon.health.HealthChecker") as mock_checker:
            await runner._maybe_run_health_check()
            mock_checker.assert_not_called()

    @pytest.mark.asyncio
    async def test_health_check_dedup(
        self, sample_config: DaemonConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test health check skipped if already ran today."""
        from zoneinfo import ZoneInfo

        runner = DaemonRunner(sample_config)
        tz = ZoneInfo("America/New_York")
        runner.state.last_health_check = datetime.now(tz)
        monkeypatch.setattr(runner.scheduler, "is_health_check_time", lambda _t: True)

        with patch("src.daemon.health.HealthChecker") as mock_checker:
            await runner._maybe_run_health_check()
            mock_checker.assert_not_called()

    @pytest.mark.asyncio
    async def test_health_check_runs(
        self, sample_config: DaemonConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test health check runs when conditions met."""
        from src.daemon.health import HealthReport, ServiceStatus

        runner = DaemonRunner(sample_config)
        monkeypatch.setattr(runner.scheduler, "is_health_check_time", lambda _t: True)

        mock_report = HealthReport(
            timestamp=datetime(2024, 1, 15),
            overall_status=ServiceStatus.HEALTHY,
            service_checks=[],
            cleanup_results=[],
            total_duration_ms=100,
        )

        with patch("src.daemon.health.HealthChecker") as mock_checker_cls:
            mock_checker = AsyncMock()
            mock_checker.run.return_value = mock_report
            mock_checker_cls.return_value = mock_checker

            await runner._maybe_run_health_check()

            mock_checker_cls.assert_called_once_with(sample_config, runner.state)
            mock_checker.run.assert_awaited_once()
            assert runner.state.last_health_check is not None

    @pytest.mark.asyncio
    async def test_health_check_error_doesnt_crash(
        self, sample_config: DaemonConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test health check failure doesn't crash daemon."""
        runner = DaemonRunner(sample_config)
        monkeypatch.setattr(runner.scheduler, "is_health_check_time", lambda _t: True)

        with patch("src.daemon.health.HealthChecker") as mock_checker_cls:
            mock_checker = AsyncMock()
            mock_checker.run.side_effect = RuntimeError("boom")
            mock_checker_cls.return_value = mock_checker

            await runner._maybe_run_health_check()

            assert any("Health check failed" in e for e in runner.state.errors)


class TestSectorRotationIntegration:
    @pytest.mark.asyncio
    async def test_sector_rotation_in_cycle(
        self, sample_config: DaemonConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test sector rotation triggered in run cycle."""
        sample_config.sector_rotation = SectorRotationConfig(enabled=True, run_time="16:15")
        runner = DaemonRunner(sample_config)

        rotation_called = False

        def mock_run_rotation() -> None:
            nonlocal rotation_called
            rotation_called = True

        monkeypatch.setattr(runner.scheduler, "is_sector_rotation_time", lambda: True)
        monkeypatch.setattr(runner.scheduler, "is_after_hours_screening_time", lambda: False)
        monkeypatch.setattr(runner.scheduler, "is_market_open", lambda: True)
        monkeypatch.setattr(runner, "_run_sector_rotation", mock_run_rotation)
        monkeypatch.setattr(runner, "_analyze_watchlist", AsyncMock(return_value=[]))
        monkeypatch.setattr(runner, "_log_results", Mock())

        await runner._run_cycle()

        assert rotation_called

    @pytest.mark.asyncio
    async def test_sector_rotation_skipped_when_disabled(
        self, sample_config: DaemonConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test sector rotation not triggered when disabled."""
        runner = DaemonRunner(sample_config)

        rotation_called = False

        def mock_run_rotation() -> None:
            nonlocal rotation_called
            rotation_called = True

        monkeypatch.setattr(runner.scheduler, "is_sector_rotation_time", lambda: False)
        monkeypatch.setattr(runner.scheduler, "is_after_hours_screening_time", lambda: False)
        monkeypatch.setattr(runner.scheduler, "is_market_open", lambda: True)
        monkeypatch.setattr(runner, "_run_sector_rotation", mock_run_rotation)
        monkeypatch.setattr(runner, "_analyze_watchlist", AsyncMock(return_value=[]))
        monkeypatch.setattr(runner, "_log_results", Mock())

        await runner._run_cycle()

        assert not rotation_called

    @patch.dict(os.environ, {}, clear=True)
    def test_run_sector_rotation_records_state(self, sample_config: DaemonConfig) -> None:
        """Test _run_sector_rotation records analysis in state."""
        from datetime import UTC, datetime

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

        with patch("src.daemon.sector_rotation.DaemonSectorRotation", return_value=mock_daemon_rotation):
            runner._run_sector_rotation()

        assert runner.state.last_sector_rotation is not None
        assert len(runner.state.sector_rotation_history) == 1
        assert runner.state.sector_rotation_history[0].leading_sectors == ["TECHNOLOGY"]

    @patch.dict(os.environ, {}, clear=True)
    def test_run_sector_rotation_dedup(self, sample_config: DaemonConfig) -> None:
        """Test sector rotation deduplication (skip if already ran today)."""
        from datetime import datetime
        from zoneinfo import ZoneInfo

        sample_config.sector_rotation = SectorRotationConfig(enabled=True)
        runner = DaemonRunner(sample_config)
        tz = ZoneInfo("America/New_York")
        runner.state.last_sector_rotation = datetime.now(tz)

        rotation_ran = False

        def mock_analyze() -> None:
            nonlocal rotation_ran
            rotation_ran = True

        with patch("src.daemon.sector_rotation.DaemonSectorRotation") as mock_cls:
            mock_cls.return_value.run = mock_analyze
            runner._run_sector_rotation()

        assert not rotation_ran
