"""Tests for daemon tearsheet generation."""

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.daemon.state import AnalysisRecord
from src.daemon.tearsheet import DaemonTearsheetGenerator
from src.metrics.tracker import TearSheet, TradeRecord
from src.strategies.session import TradingSession
from src.strategies.signal import Signal


@pytest.fixture
def sample_analyses() -> list[AnalysisRecord]:
    """Create sample analysis records."""
    base_time = datetime(2024, 1, 15, 10, 0, tzinfo=UTC)
    return [
        AnalysisRecord(
            symbol="AAPL",
            timestamp=base_time,
            signal="BUY",
            confidence=0.8,
            executed_trade=True,
            trading_session=TradingSession.REGULAR,
        ),
        AnalysisRecord(
            symbol="AAPL",
            timestamp=base_time.replace(hour=11),
            signal="SELL",
            confidence=0.75,
            executed_trade=True,
            trading_session=TradingSession.REGULAR,
        ),
        AnalysisRecord(
            symbol="TSLA",
            timestamp=base_time.replace(hour=12),
            signal="BUY",
            confidence=0.85,
            executed_trade=True,
            trading_session=TradingSession.REGULAR,
        ),
        AnalysisRecord(
            symbol="TSLA",
            timestamp=base_time.replace(hour=13),
            signal="HOLD",
            confidence=0.5,
            executed_trade=False,
            trading_session=TradingSession.REGULAR,
        ),
    ]


@pytest.fixture
def sample_trade_records() -> list[TradeRecord]:
    """Create sample closed trade records."""
    base_time = datetime(2024, 1, 15, 10, 0, tzinfo=UTC)
    return [
        TradeRecord(
            timestamp=base_time,
            symbol="AAPL",
            action=Signal.BUY,
            entry_price=150.0,
            exit_price=155.0,
            shares=100,
            stop_loss_price=145.0,
            confidence=0.8,
            risk_level="LOW",
            status="CLOSED",
            pnl=500.0,
            pnl_percent=3.33,
            strategy_name="momentum",
        ),
        TradeRecord(
            timestamp=base_time.replace(hour=11),
            symbol="TSLA",
            action=Signal.BUY,
            entry_price=200.0,
            exit_price=190.0,
            shares=50,
            stop_loss_price=195.0,
            confidence=0.75,
            risk_level="MEDIUM",
            status="CLOSED",
            pnl=-500.0,
            pnl_percent=-5.0,
            strategy_name="momentum",
        ),
    ]


@pytest.fixture
def mock_broker():
    """Create mock Alpaca broker."""
    broker = MagicMock()
    account_info = MagicMock()

    # Create mock closed position
    closed_pos = MagicMock()
    closed_pos.symbol = "AAPL"
    closed_pos.qty = 100
    closed_pos.avg_entry_price = 150.0
    closed_pos.current_price = 155.0
    closed_pos.unrealized_pl = 500.0
    closed_pos.unrealized_plpc = 0.0333
    closed_pos.entry_time = datetime(2024, 1, 15, 10, 0, tzinfo=UTC)

    account_info.closed_positions = [closed_pos]
    broker.get_account_info.return_value = account_info
    return broker


@pytest.fixture
def mock_market_fetcher():
    """Create mock market data fetcher."""
    fetcher = MagicMock()
    market_data = MagicMock()
    market_data.data = pd.DataFrame(
        {
            "close": [100.0, 101.0, 102.0, 101.5, 103.0],
        },
        index=pd.date_range("2024-01-10", periods=5, freq="D"),
    )
    fetcher.fetch_daily.return_value = market_data
    return fetcher


def test_generator_initialization():
    """Test generator initializes correctly."""
    generator = DaemonTearsheetGenerator()
    assert generator.broker is None
    assert generator.market_fetcher is None
    assert generator.reporter is not None


def test_generator_initialization_with_broker(mock_broker, mock_market_fetcher):
    """Test generator initializes with broker and market fetcher."""
    generator = DaemonTearsheetGenerator(
        broker=mock_broker,
        market_fetcher=mock_market_fetcher,
    )
    assert generator.broker is mock_broker
    assert generator.market_fetcher is mock_market_fetcher


def test_simulate_trades_from_analyses(sample_analyses):
    """Test simulating trades from analysis records."""
    generator = DaemonTearsheetGenerator()
    trades = generator._simulate_trades_from_analyses(sample_analyses)

    # Should create 1 closed trade from AAPL BUY->SELL
    assert len(trades) == 1
    assert trades[0].symbol == "AAPL"
    assert trades[0].action == Signal.BUY
    assert trades[0].status == "CLOSED"
    assert trades[0].pnl is not None


def test_simulate_trades_empty_analyses():
    """Test simulating trades with empty analyses."""
    generator = DaemonTearsheetGenerator()
    trades = generator._simulate_trades_from_analyses([])
    assert trades == []


def test_simulate_trades_only_buys(sample_analyses):
    """Test simulating trades with only BUY signals."""
    buy_only = [a for a in sample_analyses if a.signal == "BUY"]
    generator = DaemonTearsheetGenerator()
    trades = generator._simulate_trades_from_analyses(buy_only)

    # No closed trades since no matching SELL signals
    assert trades == []


def test_fetch_benchmark_returns(mock_market_fetcher, sample_trade_records):
    """Test fetching benchmark returns."""
    generator = DaemonTearsheetGenerator(market_fetcher=mock_market_fetcher)
    returns = generator._fetch_benchmark_returns("SPY", sample_trade_records)

    assert returns is not None
    assert len(returns) > 0
    mock_market_fetcher.fetch_daily.assert_called_once()


def test_fetch_benchmark_returns_no_fetcher(sample_trade_records):
    """Test fetching benchmark returns without market fetcher."""
    generator = DaemonTearsheetGenerator()
    returns = generator._fetch_benchmark_returns("SPY", sample_trade_records)
    assert returns is None


def test_fetch_benchmark_returns_empty_trades(mock_market_fetcher):
    """Test fetching benchmark returns with empty trades."""
    generator = DaemonTearsheetGenerator(market_fetcher=mock_market_fetcher)
    returns = generator._fetch_benchmark_returns("SPY", [])
    assert returns is None


@patch("src.daemon.tearsheet.QuantStatsReporter.generate_tearsheet")
def test_generate_portfolio_tearsheet(mock_generate, sample_analyses, mock_broker, mock_market_fetcher):
    """Test generating portfolio tearsheet."""
    mock_tearsheet = TearSheet(
        symbol="PORTFOLIO",
        start_date=datetime(2024, 1, 15, tzinfo=UTC),
        end_date=datetime(2024, 1, 16, tzinfo=UTC),
        cagr=0.15,
        sharpe_ratio=1.5,
        sortino_ratio=1.8,
        calmar_ratio=1.2,
        max_drawdown=-0.1,
        max_drawdown_duration_days=5,
        volatility_annual=0.2,
        win_rate=0.6,
        profit_factor=1.5,
        avg_win=500.0,
        avg_loss=-300.0,
        best_day=0.05,
        worst_day=-0.03,
        monthly_returns={"2024-01": 0.02},
        html_report_path="/path/to/report.html",
        generated_at=datetime.now(UTC),
    )
    mock_generate.return_value = mock_tearsheet

    generator = DaemonTearsheetGenerator(broker=mock_broker, market_fetcher=mock_market_fetcher)
    tearsheet = generator.generate_portfolio_tearsheet(sample_analyses, benchmark_symbol="SPY")

    assert tearsheet is not None
    assert tearsheet.symbol == "PORTFOLIO"
    mock_generate.assert_called_once()


@patch("src.daemon.tearsheet.QuantStatsReporter.generate_tearsheet")
def test_generate_portfolio_tearsheet_no_trades(mock_generate):
    """Test generating tearsheet with no closed trades."""
    # Create broker with no closed positions
    broker = MagicMock()
    account_info = MagicMock()
    account_info.closed_positions = []
    broker.get_account_info.return_value = account_info

    generator = DaemonTearsheetGenerator(broker=broker)
    tearsheet = generator.generate_portfolio_tearsheet([], benchmark_symbol="SPY")

    assert tearsheet is None
    mock_generate.assert_not_called()


def test_cleanup_old_tearsheets(tmp_path):
    """Test cleanup of old tearsheet files."""
    # Create test tearsheet directory
    tearsheet_dir = tmp_path / ".ai-casino" / "tearsheets"
    tearsheet_dir.mkdir(parents=True)

    # Create old and new files
    old_file = tearsheet_dir / "PORTFOLIO_20240101_120000.html"
    new_file = tearsheet_dir / "PORTFOLIO_20240115_120000.html"
    old_file.touch()
    new_file.touch()

    # Modify old file timestamp (31 days ago)
    old_time = datetime.now(UTC).timestamp() - (31 * 24 * 60 * 60)
    old_file.touch(exist_ok=True)
    import os

    os.utime(old_file, (old_time, old_time))

    generator = DaemonTearsheetGenerator()

    with patch("pathlib.Path.home", return_value=tmp_path):
        generator.cleanup_old_tearsheets(retention_days=30)

    # Old file should be deleted, new file should remain
    assert not old_file.exists()
    assert new_file.exists()


def test_cleanup_old_tearsheets_no_directory():
    """Test cleanup when tearsheet directory doesn't exist."""
    generator = DaemonTearsheetGenerator()

    with patch("pathlib.Path.home", return_value=Path("/nonexistent")):
        generator.cleanup_old_tearsheets(retention_days=30)


def test_repr():
    """Test string representation."""
    generator = DaemonTearsheetGenerator()
    assert "no broker" in repr(generator)

    generator = DaemonTearsheetGenerator(broker=MagicMock())
    assert "with broker" in repr(generator)
