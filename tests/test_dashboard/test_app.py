"""Tests for dashboard Dash app."""

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from src.daemon.api import (
    AnalysesResponse,
    AnalysisRecordResponse,
    ConfigResponse,
    DegradationResponse,
    EventResponse,
    FullConfigResponse,
    HealthResponse,
    PositionResponse,
    PositionsResponse,
    RiskReportResponse,
    SnapshotsResponse,
    StateSummaryResponse,
    WatchlistResponse,
)
from src.dashboard.app import create_dash_app
from src.dashboard.config import DashboardConfig


@pytest.fixture
def mock_daemon_api_client(monkeypatch):
    """Mock DaemonAPIClient with canned responses."""
    mock_client = MagicMock()

    # Health
    mock_client.get_health.return_value = HealthResponse(
        status="healthy",
        uptime_seconds=3600.0,
        running=True,
        last_run="2025-01-01T12:00:00",
    )

    # Summary
    mock_client.get_state_summary.return_value = StateSummaryResponse(
        total_analyses=42,
        total_trades=10,
        error_count=2,
        degradation_tier="FULL",
        trading_mode="paper",
    )

    # Config
    mock_client.get_config.return_value = ConfigResponse(
        watchlist=["AAPL", "TSLA"],
        interval_minutes=15,
        market_hours_only=True,
        auto_trade=False,
        trading_mode="paper",
        pre_market_enabled=False,
    )

    # Degradation
    mock_client.get_degradation.return_value = DegradationResponse(
        tier="FULL",
        unavailable_services=[],
        confidence_adjustment=1.0,
        halt_reason=None,
    )

    # Analyses
    mock_client.get_analyses.return_value = AnalysesResponse(
        analyses=[
            AnalysisRecordResponse(
                symbol="AAPL",
                timestamp=datetime.now(UTC),
                signal="BUY",
                confidence=0.85,
                executed_trade=True,
                trading_session="REGULAR",
                is_paper_trade=True,
            )
        ],
        total_count=100,
        returned_count=1,
    )

    # Positions
    mock_client.get_positions.return_value = PositionsResponse(
        positions=[
            PositionResponse(
                symbol="AAPL",
                entry_price=150.0,
                current_qty=10.0,
                current_stop_loss=145.0,
                entry_timestamp=datetime.now(UTC),
                entry_signal="BUY",
                entry_confidence=0.85,
                days_held=5,
                trailing_stop_activated=False,
                breakeven_activated=False,
                profit_targets=[155.0, 160.0],
                current_price=152.5,
            )
        ],
        count=1,
    )

    # Watchlist
    mock_client.get_watchlist.return_value = WatchlistResponse(
        symbols=["AAPL", "TSLA"],
        count=2,
        sources={"config": 2, "broker": 0, "screening": 0},
    )

    # Risk
    mock_client.get_risk.return_value = RiskReportResponse(
        timestamp=datetime.now(UTC),
        var_95=-0.05,
        var_99=-0.08,
        cvar_95=-0.06,
        cvar_99=-0.09,
        cdar_95=-0.07,
        max_drawdown=-0.10,
        risk_status="MEDIUM",
    )

    # Events
    mock_client.get_events.return_value = EventResponse(
        events=[
            {
                "event_type": "analysis_completed",
                "timestamp": datetime.now(UTC).isoformat(),
                "details": {"symbol": "AAPL"},
            }
        ],
        returned_count=1,
    )

    # Game Plan
    mock_client.get_game_plan.return_value = None

    # Snapshots
    mock_client.get_snapshots.return_value = SnapshotsResponse(snapshots=[], count=0)

    # Rebalance
    mock_client.get_rebalance.return_value = None

    # Patch DaemonAPIClient
    def mock_api_client_init(api_url):
        return mock_client

    monkeypatch.setattr("src.dashboard.app.DaemonAPIClient", mock_api_client_init)

    return mock_client


def test_create_dash_app_returns_dash_instance() -> None:
    """Test create_dash_app returns Dash instance."""
    config = DashboardConfig()
    app = create_dash_app(config)

    assert app is not None
    assert app.title == "AI Casino Dashboard"


def test_dash_app_has_layout() -> None:
    """Test Dash app has layout."""
    config = DashboardConfig()
    app = create_dash_app(config)

    assert app.layout is not None


def test_dash_app_has_interval_component() -> None:
    """Test Dash app has interval component."""
    config = DashboardConfig()
    app = create_dash_app(config)

    # Check for Interval component in layout
    layout_str = str(app.layout)
    assert "Interval" in layout_str


def test_dash_app_has_tabs() -> None:
    """Test Dash app has tabs."""
    config = DashboardConfig()
    app = create_dash_app(config)

    # Check for Tabs component in layout
    layout_str = str(app.layout)
    assert "Tabs" in layout_str


def test_overview_tab_renders(mock_daemon_api_client) -> None:
    """Test overview tab renders without error."""
    from src.dashboard.tabs import overview

    content = overview.render(mock_daemon_api_client)

    assert content is not None
    assert isinstance(content, list)
    mock_daemon_api_client.get_health.assert_called()
    mock_daemon_api_client.get_state_summary.assert_called()


def test_config_tab_renders(mock_daemon_api_client) -> None:
    """Test config tab renders without error."""
    from src.dashboard.tabs import config

    # Mock get_full_config
    mock_daemon_api_client.get_full_config.return_value = FullConfigResponse(
        watchlist=["AAPL"],
        interval_minutes=5,
        market_hours_only=True,
        auto_trade=False,
        max_concurrent_analyses=5,
        trading_mode="paper",
        paper_trading={"enabled": True, "initial_cash": 100000.0},
        schedule={"enabled": False},
        state={"state_dir": ".ai-casino"},
        journal={"enabled": True},
        health={"check_interval_seconds": 60},
        optimization={"enabled": False},
        screening={"enabled": False},
        prefetch={"enabled": False},
        sector_rotation={"enabled": False},
        earnings_calendar={"enabled": False},
        peer_analysis={"enabled": False},
        correlation_audit={"enabled": False},
        reporting={"enabled": False},
        risk_limits={"enabled": False},
        rebalancing={"enabled": False},
        signal_tracking={"enabled": False},
        pre_trade_backtesting={"enabled": False},
        game_plan={"enabled": False},
        position_management={"enabled": False},
        monte_carlo={"enabled": False},
        notifications={"enabled": False},
        analysis_orchestration={"enabled": False},
        news_watcher={"enabled": False},
        social_watcher={"enabled": False},
        filings_watcher={"enabled": False},
        anomaly_watcher={"enabled": False},
        api={"host": "127.0.0.1", "port": 8000},
        llm={"provider": "ollama"},
        api_keys={},
    )

    content = config.render(mock_daemon_api_client)

    assert content is not None
    assert isinstance(content, list)
    mock_daemon_api_client.get_full_config.assert_called()


def test_portfolio_tab_renders(mock_daemon_api_client) -> None:
    """Test portfolio tab renders without error."""
    from src.dashboard.tabs import portfolio

    content = portfolio.render(mock_daemon_api_client)

    assert content is not None
    assert isinstance(content, list)
    mock_daemon_api_client.get_positions.assert_called()


def test_signals_tab_renders(mock_daemon_api_client) -> None:
    """Test signals tab renders without error."""
    from src.dashboard.tabs import signals

    content = signals.render(mock_daemon_api_client)

    assert content is not None
    assert isinstance(content, list)
    mock_daemon_api_client.get_analyses.assert_called()


def test_risk_tab_renders(mock_daemon_api_client) -> None:
    """Test risk tab renders without error."""
    from src.dashboard.tabs import risk

    content = risk.render(mock_daemon_api_client)

    assert content is not None
    assert isinstance(content, list)
    mock_daemon_api_client.get_risk.assert_called()


def test_events_tab_renders(mock_daemon_api_client) -> None:
    """Test events tab renders without error."""
    from src.dashboard.tabs import events

    # Add missing mocks for events tab
    mock_daemon_api_client.get_market_events.return_value = EventResponse(events=[], returned_count=0)
    mock_daemon_api_client.get_degradation_history.return_value = MagicMock(records=[])

    content = events.render(mock_daemon_api_client)

    assert content is not None
    assert isinstance(content, list)
    mock_daemon_api_client.get_events.assert_called()


def test_portfolio_tab_empty_positions(mock_daemon_api_client) -> None:
    """Test portfolio tab with no positions."""
    mock_daemon_api_client.get_positions.return_value = PositionsResponse(positions=[], count=0)

    from src.dashboard.tabs import portfolio

    content = portfolio.render(mock_daemon_api_client)

    assert content is not None
    # Should show "No active positions" alert
    assert len(content) == 1


def test_signals_tab_empty_analyses(mock_daemon_api_client) -> None:
    """Test signals tab with no analyses."""
    mock_daemon_api_client.get_analyses.return_value = AnalysesResponse(
        analyses=[], total_count=0, returned_count=0
    )

    from src.dashboard.tabs import signals

    content = signals.render(mock_daemon_api_client)

    assert content is not None
    # Should show "No analyses yet" alert
    assert len(content) == 1


def test_risk_tab_no_report(mock_daemon_api_client) -> None:
    """Test risk tab with no risk report."""
    mock_daemon_api_client.get_risk.return_value = None

    from src.dashboard.tabs import risk

    content = risk.render(mock_daemon_api_client)

    assert content is not None
    # Should show "No risk report available" alert
    assert len(content) == 1


def test_events_tab_empty_events(mock_daemon_api_client) -> None:
    """Test events tab with no events."""
    mock_daemon_api_client.get_events.return_value = EventResponse(events=[], returned_count=0)
    mock_daemon_api_client.get_market_events.return_value = EventResponse(events=[], returned_count=0)
    mock_daemon_api_client.get_degradation_history.return_value = MagicMock(records=[])

    from src.dashboard.tabs import events

    content = events.render(mock_daemon_api_client)

    assert content is not None
    # Should show "No events yet" alert (updated - now returns filter controls too)
    assert len(content) >= 1
