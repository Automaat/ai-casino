"""Tests for daemon runner."""

import os
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.daemon.config import DaemonConfig
from src.daemon.runner import DaemonRunner
from src.data.broker import BrokerAccountInfo, BrokerPosition


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
    """Test broker initialization when credentials present."""
    with patch.dict(os.environ, {"ALPACA_API_KEY": "test_key", "ALPACA_SECRET_KEY": "test_secret"}):
        runner = DaemonRunner(sample_config)

        assert runner.broker is not None
        mock_broker_class.assert_called_once_with(paper=True)


@patch("src.daemon.runner.AlpacaBroker")
def test_broker_init_no_credentials(mock_broker_class: Mock, sample_config: DaemonConfig) -> None:
    """Test broker not initialized without credentials."""
    with patch.dict(os.environ, {}, clear=True):
        runner = DaemonRunner(sample_config)

        assert runner.broker is None
        mock_broker_class.assert_not_called()


@patch("src.daemon.runner.AlpacaBroker")
def test_broker_init_failure(mock_broker_class: Mock, sample_config: DaemonConfig) -> None:
    """Test daemon continues if broker init fails."""
    mock_broker_class.side_effect = ValueError("Invalid credentials")

    with patch.dict(os.environ, {"ALPACA_API_KEY": "bad_key", "ALPACA_SECRET_KEY": "bad_secret"}):
        runner = DaemonRunner(sample_config)

        assert runner.broker is None


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
