"""Tests for PortfolioHealthCheckTask."""

from unittest.mock import MagicMock

import pytest

from src.daemon.config.portfolio import PortfolioHealthConfig
from src.daemon.tasks.portfolio_tasks import PortfolioHealthCheckTask, _PortfolioMetrics


def _make_task() -> PortfolioHealthCheckTask:
    """Create task instance with mock dependencies."""
    components = MagicMock()
    container = MagicMock()
    return PortfolioHealthCheckTask(components, container)


def _make_position(
    market_value: float, unrealized_pnl: float, qty: float = 10.0, avg_entry: float = 0.0
) -> MagicMock:
    """Create mock broker position."""
    pos = MagicMock()
    pos.market_value = market_value
    pos.unrealized_pnl = unrealized_pnl
    pos.qty = qty
    pos.avg_entry_price = avg_entry or (market_value - unrealized_pnl) / qty
    return pos


def _make_config(
    max_concentration: float = 0.25,
    min_cash: float = 0.05,
    drawdown_threshold: float = 0.10,
) -> PortfolioHealthConfig:
    """Create test config with overrides."""
    return PortfolioHealthConfig(
        max_position_concentration=max_concentration,
        min_cash_percent=min_cash,
        drawdown_alert_threshold=drawdown_threshold,
    )


class TestComputeMetrics:
    @pytest.mark.unit
    def test_empty_portfolio(self) -> None:
        task = _make_task()
        metrics = task._compute_metrics({}, portfolio_value=50000.0, cash=50000.0)

        assert metrics["total_positions"] == 0
        assert metrics["cash_percent"] == 100.0
        assert metrics["max_concentration_percent"] == 0.0
        assert metrics["biggest_drawdown_symbol"] is None
        assert metrics["biggest_drawdown_percent"] == 0.0

    @pytest.mark.unit
    def test_single_position_metrics(self) -> None:
        task = _make_task()
        positions = {
            "AAPL": _make_position(market_value=40000.0, unrealized_pnl=2000.0, qty=200, avg_entry=190.0)
        }
        metrics = task._compute_metrics(positions, portfolio_value=50000.0, cash=10000.0)

        assert metrics["total_positions"] == 1
        assert metrics["max_concentration_symbol"] == "AAPL"
        assert metrics["max_concentration_percent"] == pytest.approx(80.0)
        assert metrics["cash_percent"] == pytest.approx(20.0)

    @pytest.mark.unit
    def test_biggest_drawdown_tracked(self) -> None:
        task = _make_task()
        positions = {
            "AAPL": _make_position(market_value=20000.0, unrealized_pnl=1000.0, qty=100, avg_entry=190.0),
            "TSLA": _make_position(market_value=10000.0, unrealized_pnl=-2000.0, qty=50, avg_entry=240.0),
        }
        metrics = task._compute_metrics(positions, portfolio_value=50000.0, cash=20000.0)

        assert metrics["biggest_drawdown_symbol"] == "TSLA"
        assert metrics["biggest_drawdown_percent"] < 0

    @pytest.mark.unit
    def test_max_concentration_identifies_largest_position(self) -> None:
        task = _make_task()
        positions = {
            "AAPL": _make_position(market_value=30000.0, unrealized_pnl=0.0, qty=100, avg_entry=300.0),
            "TSLA": _make_position(market_value=10000.0, unrealized_pnl=0.0, qty=50, avg_entry=200.0),
        }
        metrics = task._compute_metrics(positions, portfolio_value=50000.0, cash=10000.0)

        assert metrics["max_concentration_symbol"] == "AAPL"
        assert metrics["max_concentration_percent"] == pytest.approx(60.0)

    @pytest.mark.unit
    def test_zero_portfolio_value_safe(self) -> None:
        task = _make_task()
        positions = {"AAPL": _make_position(market_value=0.0, unrealized_pnl=0.0, qty=10, avg_entry=100.0)}
        metrics = task._compute_metrics(positions, portfolio_value=0.0, cash=0.0)

        assert metrics["cash_percent"] == 100.0
        assert metrics["total_exposure_percent"] == 0.0


class TestDetermineStatus:
    @pytest.mark.unit
    def test_healthy_portfolio(self) -> None:
        task = _make_task()
        config = _make_config()
        metrics: _PortfolioMetrics = {
            "total_positions": 3,
            "total_exposure_percent": 80.0,
            "cash_percent": 20.0,
            "max_concentration_percent": 20.0,
            "max_concentration_symbol": "AAPL",
            "total_pnl_percent": 3.0,
            "biggest_drawdown_percent": -5.0,
            "biggest_drawdown_symbol": "TSLA",
        }
        assert task._determine_status(metrics, config) == "HEALTHY"

    @pytest.mark.unit
    def test_warning_single_threshold_breach(self) -> None:
        task = _make_task()
        config = _make_config(max_concentration=0.25)
        metrics: _PortfolioMetrics = {
            "total_positions": 2,
            "total_exposure_percent": 90.0,
            "cash_percent": 20.0,
            "max_concentration_percent": 30.0,  # Exceeds 25%
            "max_concentration_symbol": "AAPL",
            "total_pnl_percent": 1.0,
            "biggest_drawdown_percent": -5.0,
            "biggest_drawdown_symbol": "AAPL",
        }
        assert task._determine_status(metrics, config) == "WARNING"

    @pytest.mark.unit
    def test_critical_two_threshold_breaches(self) -> None:
        task = _make_task()
        config = _make_config(max_concentration=0.25, min_cash=0.05, drawdown_threshold=0.10)
        metrics: _PortfolioMetrics = {
            "total_positions": 2,
            "total_exposure_percent": 98.0,
            "cash_percent": 2.0,  # Below 5% minimum
            "max_concentration_percent": 35.0,  # Exceeds 25%
            "max_concentration_symbol": "AAPL",
            "total_pnl_percent": -5.0,
            "biggest_drawdown_percent": -5.0,
            "biggest_drawdown_symbol": "AAPL",
        }
        assert task._determine_status(metrics, config) == "CRITICAL"

    @pytest.mark.unit
    def test_critical_all_thresholds_breached(self) -> None:
        task = _make_task()
        config = _make_config(max_concentration=0.25, min_cash=0.05, drawdown_threshold=0.10)
        metrics: _PortfolioMetrics = {
            "total_positions": 1,
            "total_exposure_percent": 99.0,
            "cash_percent": 1.0,
            "max_concentration_percent": 99.0,
            "max_concentration_symbol": "AAPL",
            "total_pnl_percent": -20.0,
            "biggest_drawdown_percent": -25.0,
            "biggest_drawdown_symbol": "AAPL",
        }
        assert task._determine_status(metrics, config) == "CRITICAL"


class TestRuleBasedAnalysis:
    @pytest.mark.unit
    def test_healthy_returns_default_recommendation(self) -> None:
        task = _make_task()
        config = _make_config()
        metrics: _PortfolioMetrics = {
            "total_positions": 3,
            "total_exposure_percent": 80.0,
            "cash_percent": 20.0,
            "max_concentration_percent": 20.0,
            "max_concentration_symbol": "AAPL",
            "total_pnl_percent": 3.0,
            "biggest_drawdown_percent": -5.0,
            "biggest_drawdown_symbol": None,
        }
        recs, constraints = task._rule_based_analysis(metrics, "HEALTHY", config)

        assert len(recs) == 1
        assert "within all thresholds" in recs[0]
        assert constraints == []

    @pytest.mark.unit
    def test_concentration_breach_generates_reduce_constraint(self) -> None:
        task = _make_task()
        config = _make_config(max_concentration=0.25)
        metrics: _PortfolioMetrics = {
            "total_positions": 2,
            "total_exposure_percent": 90.0,
            "cash_percent": 20.0,
            "max_concentration_percent": 40.0,
            "max_concentration_symbol": "AAPL",
            "total_pnl_percent": 1.0,
            "biggest_drawdown_percent": 0.0,
            "biggest_drawdown_symbol": None,
        }
        recs, constraints = task._rule_based_analysis(metrics, "WARNING", config)

        assert any(c == "reduce:AAPL" for c in constraints)
        assert any("AAPL" in r for r in recs)

    @pytest.mark.unit
    def test_low_cash_generates_block_buy_constraint(self) -> None:
        task = _make_task()
        config = _make_config(min_cash=0.05)
        metrics: _PortfolioMetrics = {
            "total_positions": 3,
            "total_exposure_percent": 98.0,
            "cash_percent": 2.0,
            "max_concentration_percent": 20.0,
            "max_concentration_symbol": "AAPL",
            "total_pnl_percent": 1.0,
            "biggest_drawdown_percent": 0.0,
            "biggest_drawdown_symbol": None,
        }
        recs, constraints = task._rule_based_analysis(metrics, "WARNING", config)

        assert "block_buy:ALL" in constraints
        assert any("cash" in r.lower() for r in recs)

    @pytest.mark.unit
    def test_drawdown_breach_generates_force_review_constraint(self) -> None:
        task = _make_task()
        config = _make_config(drawdown_threshold=0.10)
        metrics: _PortfolioMetrics = {
            "total_positions": 2,
            "total_exposure_percent": 80.0,
            "cash_percent": 20.0,
            "max_concentration_percent": 20.0,
            "max_concentration_symbol": "AAPL",
            "total_pnl_percent": -5.0,
            "biggest_drawdown_percent": -15.0,
            "biggest_drawdown_symbol": "TSLA",
        }
        recs, constraints = task._rule_based_analysis(metrics, "WARNING", config)

        assert "force_review:TSLA" in constraints
        assert any("TSLA" in r for r in recs)

    @pytest.mark.unit
    def test_no_drawdown_symbol_skips_drawdown_constraint(self) -> None:
        task = _make_task()
        config = _make_config(drawdown_threshold=0.10)
        metrics: _PortfolioMetrics = {
            "total_positions": 0,
            "total_exposure_percent": 0.0,
            "cash_percent": 100.0,
            "max_concentration_percent": 0.0,
            "max_concentration_symbol": "N/A",
            "total_pnl_percent": 0.0,
            "biggest_drawdown_percent": -15.0,  # big drawdown but no symbol
            "biggest_drawdown_symbol": None,
        }
        _recs, constraints = task._rule_based_analysis(metrics, "HEALTHY", config)

        assert not any("force_review" in c for c in constraints)
