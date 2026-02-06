"""Unit tests for QuantStats reporter."""

from datetime import UTC, datetime
from unittest.mock import patch

import pandas as pd
import pytest

from src.metrics.performance import build_daily_equity_curve, equity_curve_to_returns
from src.metrics.quantstats_reporter import QuantStatsReporter
from src.metrics.tracker import TearSheet, TradeRecord
from src.strategies.signal import Signal


@pytest.fixture
def mock_quantstats():
    """Mock QuantStats module."""
    with patch("src.metrics.quantstats_reporter.qs") as mock_qs:
        mock_qs.stats.RISK_FREE_RATE = 0.02
        mock_qs.stats.cagr.return_value = 0.15
        mock_qs.stats.sharpe.return_value = 1.5
        mock_qs.stats.sortino.return_value = 1.8
        mock_qs.stats.calmar.return_value = 2.0
        mock_qs.stats.max_drawdown.return_value = -0.10
        mock_qs.stats.volatility.return_value = 0.20
        mock_qs.stats.monthly_returns.return_value = pd.Series(
            {"2024-01": 0.05, "2024-02": 0.03, "2024-03": 0.02}, name="returns"
        )
        mock_qs.stats.to_drawdown_series.return_value = pd.Series([0.0, -0.05, -0.10, 0.0])
        mock_qs.stats.alpha.return_value = 0.03
        mock_qs.stats.beta.return_value = 0.9
        mock_qs.reports.html.return_value = None
        yield mock_qs


def test_quantstats_reporter_init():
    """Test QuantStatsReporter initialization."""
    reporter = QuantStatsReporter(risk_free_rate=0.03)

    assert reporter.risk_free_rate == 0.03


def test_quantstats_reporter_init_default():
    """Test QuantStatsReporter initialization with default risk-free rate."""
    reporter = QuantStatsReporter()

    assert reporter.risk_free_rate == 0.02


def test_build_daily_equity_curve(sample_trades_for_tearsheet):
    """Test building daily equity curve from trades."""
    equity_curve = build_daily_equity_curve(sample_trades_for_tearsheet)

    assert not equity_curve.empty
    assert len(equity_curve) > len(sample_trades_for_tearsheet)
    first_pnl = sample_trades_for_tearsheet[0].pnl
    assert equity_curve.iloc[0] == pytest.approx(100000.0 + first_pnl, abs=1.0)


def test_build_daily_equity_curve_empty():
    """Test building equity curve from empty trades list."""
    equity_curve = build_daily_equity_curve([])

    assert equity_curve.empty


def test_build_daily_equity_curve_open_trades():
    """Test building equity curve excludes open trades."""
    open_trade = TradeRecord(
        timestamp=datetime.now(UTC),
        symbol="AAPL",
        action=Signal.BUY,
        entry_price=100.0,
        exit_price=None,
        shares=100,
        stop_loss_price=95.0,
        confidence=0.8,
        risk_level="LOW",
        status="OPEN",
        pnl=None,
        pnl_percent=None,
    )

    equity_curve = build_daily_equity_curve([open_trade])

    assert equity_curve.empty


def test_equity_curve_to_returns():
    """Test converting equity curve to returns."""
    equity = pd.Series([100000.0, 105000.0, 103000.0, 108000.0])
    returns = equity_curve_to_returns(equity)

    assert len(returns) == 4
    assert returns.iloc[0] == 0.0
    assert returns.iloc[1] == pytest.approx(0.05, abs=0.001)


def test_generate_tearsheet_without_benchmark(mock_quantstats, sample_trades_for_tearsheet):
    """Test generating tearsheet without benchmark."""
    reporter = QuantStatsReporter()

    tearsheet = reporter.generate_tearsheet("AAPL", sample_trades_for_tearsheet)

    assert isinstance(tearsheet, TearSheet)
    assert tearsheet.symbol == "AAPL"
    assert tearsheet.cagr == 0.15
    assert tearsheet.sharpe_ratio == 1.5
    assert tearsheet.benchmark_symbol is None
    assert tearsheet.alpha is None
    assert "AAPL_" in tearsheet.html_report_path
    assert tearsheet.html_report_path.endswith(".html")


def test_generate_tearsheet_with_benchmark(mock_quantstats, sample_trades_for_tearsheet):
    """Test generating tearsheet with benchmark."""
    reporter = QuantStatsReporter()

    benchmark_returns = pd.Series([0.01, 0.02, 0.01], index=pd.date_range("2024-01-01", periods=3))

    tearsheet = reporter.generate_tearsheet(
        "AAPL", sample_trades_for_tearsheet, benchmark_symbol="SPY", benchmark_returns=benchmark_returns
    )

    assert isinstance(tearsheet, TearSheet)
    assert tearsheet.benchmark_symbol == "SPY"
    assert tearsheet.alpha == 0.03
    assert tearsheet.beta == 0.9


def test_generate_tearsheet_no_closed_trades():
    """Test generating tearsheet with no closed trades."""
    reporter = QuantStatsReporter()

    open_trade = TradeRecord(
        timestamp=datetime.now(UTC),
        symbol="AAPL",
        action=Signal.BUY,
        entry_price=100.0,
        exit_price=None,
        shares=100,
        stop_loss_price=95.0,
        confidence=0.8,
        risk_level="LOW",
        status="OPEN",
        pnl=None,
        pnl_percent=None,
    )

    with pytest.raises(ValueError, match="Cannot generate tearsheet"):
        reporter.generate_tearsheet("AAPL", [open_trade])


def test_calculate_metrics(mock_quantstats):
    """Test calculating QuantStats metrics."""
    reporter = QuantStatsReporter()

    returns = pd.Series([0.01, 0.02, -0.01, 0.03])
    metrics = reporter._calculate_metrics(returns)

    assert metrics["cagr"] == 0.15
    assert metrics["sharpe_ratio"] == 1.5
    assert metrics["sortino_ratio"] == 1.8
    assert metrics["max_drawdown"] == -0.10
    assert metrics["volatility_annual"] == 0.20
    assert isinstance(metrics["monthly_returns"], dict)


def test_calculate_max_dd_duration():
    """Test calculating maximum drawdown duration."""
    reporter = QuantStatsReporter()

    dd_series = pd.Series([0.0, -0.05, -0.10, -0.08, 0.0, -0.02, 0.0])
    duration = reporter._calculate_max_dd_duration(dd_series)

    assert duration == 3


def test_calculate_max_dd_duration_empty():
    """Test calculating max DD duration with empty series."""
    reporter = QuantStatsReporter()

    dd_series = pd.Series([])
    duration = reporter._calculate_max_dd_duration(dd_series)

    assert duration is None


def test_generate_html(mock_quantstats):
    """Test generating HTML report."""
    reporter = QuantStatsReporter()

    returns = pd.Series([0.01, 0.02, -0.01], index=pd.date_range("2024-01-01", periods=3))
    html_path = reporter._generate_html("AAPL", returns)

    assert "AAPL_" in html_path
    assert html_path.endswith(".html")
    assert ".ai-casino/tearsheets" in html_path


def test_repr():
    """Test QuantStatsReporter string representation."""
    reporter = QuantStatsReporter(risk_free_rate=0.03)

    assert "QuantStatsReporter" in repr(reporter)
    assert "0.03" in repr(reporter)
