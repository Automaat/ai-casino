"""Tests for dashboard API client."""

from unittest.mock import MagicMock, patch

import httpx

from src.daemon.api import (
    AnalysesResponse,
    ConfigResponse,
    DegradationResponse,
    EventResponse,
    HealthResponse,
    PositionsResponse,
    RebalanceResponse,
    RiskReportResponse,
    SnapshotsResponse,
    StateSummaryResponse,
    WatchlistResponse,
)
from src.dashboard.api_client import DaemonAPIClient


def test_daemon_api_client_init() -> None:
    """Test API client initialization."""
    client = DaemonAPIClient("http://localhost:8001")

    assert client.api_url == "http://localhost:8001"
    assert client._client is not None


def test_daemon_api_client_init_strips_trailing_slash() -> None:
    """Test API URL trailing slash is stripped."""
    client = DaemonAPIClient("http://localhost:8001/")

    assert client.api_url == "http://localhost:8001"


def test_is_healthy_success() -> None:
    """Test is_healthy returns True when daemon is healthy."""
    client = DaemonAPIClient("http://localhost:8001")

    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None

    with patch.object(client._client, "get", return_value=mock_response):
        assert client.is_healthy() is True


def test_is_healthy_failure() -> None:
    """Test is_healthy returns False when daemon is unreachable."""
    client = DaemonAPIClient("http://localhost:8001")

    with patch.object(client._client, "get", side_effect=httpx.ConnectError("Connection refused")):
        assert client.is_healthy() is False


def test_get_health() -> None:
    """Test get_health endpoint."""
    client = DaemonAPIClient("http://localhost:8001")

    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "status": "healthy",
        "uptime_seconds": 123.45,
        "running": True,
        "last_run": "2025-01-01T12:00:00",
    }

    with patch.object(client._client, "get", return_value=mock_response):
        health = client.get_health()

        assert isinstance(health, HealthResponse)
        assert health.status == "healthy"
        assert health.uptime_seconds == 123.45
        assert health.running is True


def test_get_state_summary() -> None:
    """Test get_state_summary endpoint."""
    client = DaemonAPIClient("http://localhost:8001")

    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "total_analyses": 42,
        "total_trades": 10,
        "error_count": 2,
        "degradation_tier": "FULL",
        "trading_mode": "paper",
    }

    with patch.object(client._client, "get", return_value=mock_response):
        summary = client.get_state_summary()

        assert isinstance(summary, StateSummaryResponse)
        assert summary.total_analyses == 42
        assert summary.total_trades == 10


def test_get_config() -> None:
    """Test get_config endpoint."""
    client = DaemonAPIClient("http://localhost:8001")

    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "watchlist": ["AAPL", "TSLA"],
        "interval_minutes": 15,
        "market_hours_only": True,
        "auto_trade": False,
        "trading_mode": "paper",
        "pre_market_enabled": False,
    }

    with patch.object(client._client, "get", return_value=mock_response):
        config = client.get_config()

        assert isinstance(config, ConfigResponse)
        assert config.watchlist == ["AAPL", "TSLA"]
        assert config.interval_minutes == 15


def test_get_analyses() -> None:
    """Test get_analyses endpoint."""
    client = DaemonAPIClient("http://localhost:8001")

    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "analyses": [
            {
                "symbol": "AAPL",
                "timestamp": "2025-01-01T12:00:00",
                "signal": "BUY",
                "confidence": 0.85,
                "executed_trade": True,
                "trading_session": "REGULAR",
                "is_paper_trade": True,
            }
        ],
        "total_count": 100,
        "returned_count": 1,
    }

    with patch.object(client._client, "get", return_value=mock_response):
        analyses = client.get_analyses(limit=50)

        assert isinstance(analyses, AnalysesResponse)
        assert analyses.total_count == 100
        assert analyses.returned_count == 1
        assert len(analyses.analyses) == 1


def test_get_positions() -> None:
    """Test get_positions endpoint."""
    client = DaemonAPIClient("http://localhost:8001")

    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "positions": [
            {
                "symbol": "AAPL",
                "entry_price": 150.0,
                "current_qty": 10.0,
                "current_stop_loss": 145.0,
                "entry_timestamp": "2025-01-01T12:00:00",
                "entry_signal": "BUY",
                "entry_confidence": 0.85,
                "days_held": 5,
                "trailing_stop_activated": False,
                "breakeven_activated": False,
                "profit_targets": [155.0, 160.0],
                "current_price": 152.5,
            }
        ],
        "count": 1,
    }

    with patch.object(client._client, "get", return_value=mock_response):
        positions = client.get_positions()

        assert isinstance(positions, PositionsResponse)
        assert positions.count == 1
        assert len(positions.positions) == 1


def test_get_watchlist() -> None:
    """Test get_watchlist endpoint."""
    client = DaemonAPIClient("http://localhost:8001")

    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "symbols": ["AAPL", "TSLA", "MSFT"],
        "count": 3,
        "sources": {"config": 2, "broker": 1, "screening": 0},
    }

    with patch.object(client._client, "get", return_value=mock_response):
        watchlist = client.get_watchlist()

        assert isinstance(watchlist, WatchlistResponse)
        assert watchlist.count == 3
        assert len(watchlist.symbols) == 3


def test_get_risk_with_data() -> None:
    """Test get_risk endpoint with risk report."""
    client = DaemonAPIClient("http://localhost:8001")

    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "timestamp": "2025-01-01T12:00:00",
        "var_95": -0.05,
        "var_99": -0.08,
        "cvar_95": -0.06,
        "cvar_99": -0.09,
        "cdar_95": -0.07,
        "max_drawdown": -0.10,
        "risk_status": "MEDIUM",
    }

    with patch.object(client._client, "get", return_value=mock_response):
        risk = client.get_risk()

        assert isinstance(risk, RiskReportResponse)
        assert risk.var_95 == -0.05
        assert risk.risk_status == "MEDIUM"


def test_get_risk_no_data() -> None:
    """Test get_risk endpoint with no risk report."""
    client = DaemonAPIClient("http://localhost:8001")

    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = None

    with patch.object(client._client, "get", return_value=mock_response):
        risk = client.get_risk()

        assert risk is None


def test_get_degradation() -> None:
    """Test get_degradation endpoint."""
    client = DaemonAPIClient("http://localhost:8001")

    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "tier": "FULL",
        "unavailable_services": [],
        "confidence_adjustment": 1.0,
        "halt_reason": None,
    }

    with patch.object(client._client, "get", return_value=mock_response):
        degradation = client.get_degradation()

        assert isinstance(degradation, DegradationResponse)
        assert degradation.tier == "FULL"
        assert degradation.confidence_adjustment == 1.0


def test_get_events() -> None:
    """Test get_events endpoint."""
    client = DaemonAPIClient("http://localhost:8001")

    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "events": [{"event_type": "analysis_completed", "timestamp": "2025-01-01T12:00:00", "details": {}}],
        "returned_count": 1,
    }

    with patch.object(client._client, "get", return_value=mock_response):
        events = client.get_events(limit=100)

        assert isinstance(events, EventResponse)
        assert events.returned_count == 1
        assert len(events.events) == 1


def test_retry_logic() -> None:
    """Test HTTP_RETRY decorator retries on failure (no delays)."""
    import time

    client = DaemonAPIClient("http://localhost:8001")

    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = [
        httpx.ConnectError("Connection refused"),
        httpx.ConnectError("Connection refused"),
        None,
    ]
    mock_response.json.return_value = {
        "status": "healthy",
        "uptime_seconds": 100.0,
        "running": True,
        "last_run": None,
    }

    with (
        patch.object(time, "sleep", return_value=None),
        patch.object(client._client, "get", return_value=mock_response),
    ):
        health = client.get_health()

        assert isinstance(health, HealthResponse)
        assert mock_response.raise_for_status.call_count == 3


def test_close() -> None:
    """Test client close."""
    client = DaemonAPIClient("http://localhost:8001")

    with patch.object(client._client, "close") as mock_close:
        client.close()
        mock_close.assert_called_once()


def test_repr() -> None:
    """Test __repr__ output."""
    client = DaemonAPIClient("http://localhost:8001")
    repr_str = repr(client)

    assert "DaemonAPIClient" in repr_str
    assert "http://localhost:8001" in repr_str


def test_get_snapshots() -> None:
    """Test get_snapshots endpoint."""
    client = DaemonAPIClient("http://localhost:8001")

    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "snapshots": [
            {
                "timestamp": "2025-01-01T12:00:00",
                "portfolio_value": 10000.0,
                "balance": 5000.0,
                "total_exposure": 5000.0,
            }
        ],
        "count": 1,
    }

    with patch.object(client._client, "get", return_value=mock_response):
        snapshots = client.get_snapshots(days=30)

        assert isinstance(snapshots, SnapshotsResponse)
        assert snapshots.count == 1
        assert len(snapshots.snapshots) == 1


def test_get_rebalance() -> None:
    """Test get_rebalance endpoint."""
    client = DaemonAPIClient("http://localhost:8001")

    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "timestamp": "2025-01-01T12:00:00",
        "method": "equal_weight",
        "allocations": [
            {
                "symbol": "AAPL",
                "target_weight": 0.5,
                "current_weight": 0.4,
                "delta": -0.1,
                "action": "INCREASE",
            }
        ],
        "expected_return": 0.15,
        "expected_volatility": 0.20,
        "sharpe_ratio": 0.75,
    }

    with patch.object(client._client, "get", return_value=mock_response):
        rebalance = client.get_rebalance()

        assert isinstance(rebalance, RebalanceResponse)
        assert len(rebalance.allocations) == 1


def test_get_rebalance_null() -> None:
    """Test get_rebalance endpoint with null response."""
    client = DaemonAPIClient("http://localhost:8001")

    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = None

    with patch.object(client._client, "get", return_value=mock_response):
        rebalance = client.get_rebalance()

        assert rebalance is None
