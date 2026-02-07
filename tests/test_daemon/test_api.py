"""Tests for embedded FastAPI server."""

from datetime import UTC, datetime
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from src.daemon.api import create_api_app
from src.daemon.config import ApiConfig, DaemonConfig, ScheduleConfig
from src.daemon.runner import DaemonRunner
from src.daemon.state import DaemonState, DegradationRecord


@pytest.fixture
def mock_runner(tmp_path) -> Mock:
    """Create mock DaemonRunner with config and state."""
    config = DaemonConfig(
        watchlist=["AAPL", "TSLA"],
        interval_minutes=30,
        market_hours_only=True,
        auto_trade=False,
        schedule=ScheduleConfig(enable_pre_market=False),
        api=ApiConfig(enabled=True, host="127.0.0.1", port=8484),
    )

    state = DaemonState(
        total_analyses=42,
        total_trades=15,
        errors=["error1", "error2"],
        current_trading_mode="paper",
        last_run=datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC),
        degradation_history=[
            DegradationRecord(
                timestamp=datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC),
                tier="FULL",
                unavailable_services=[],
                confidence_adjustment=0.0,
            )
        ],
    )

    runner = Mock(spec=DaemonRunner)
    runner.config = config
    runner.state = state
    runner.running = True

    return runner


@pytest.fixture
def client(mock_runner: Mock) -> TestClient:
    """Create FastAPI test client."""
    app = create_api_app(mock_runner)
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_running(self, client: TestClient) -> None:
        """Test health endpoint when daemon is running."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert data["running"] is True
        assert data["last_run"] == "2024-01-15T10:30:00+00:00"
        assert isinstance(data["uptime_seconds"], float)
        assert data["uptime_seconds"] >= 0

    def test_health_degraded(self, client: TestClient, mock_runner: Mock) -> None:
        """Test health endpoint when daemon is degraded."""
        mock_runner.state.degradation_history = [
            DegradationRecord(
                timestamp=datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC),
                tier="REDUCED",
                unavailable_services=["news"],
                confidence_adjustment=-0.1,
            )
        ]

        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "degraded"

    def test_health_stopped(self, client: TestClient, mock_runner: Mock) -> None:
        """Test health endpoint when daemon is stopped."""
        mock_runner.running = False

        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["running"] is False

    def test_health_no_last_run(self, client: TestClient, mock_runner: Mock) -> None:
        """Test health endpoint when no runs yet."""
        mock_runner.state.last_run = None

        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["last_run"] is None


class TestStateSummaryEndpoint:
    """Tests for /state/summary endpoint."""

    def test_state_summary(self, client: TestClient) -> None:
        """Test state summary endpoint."""
        response = client.get("/state/summary")
        assert response.status_code == 200

        data = response.json()
        assert data["total_analyses"] == 42
        assert data["total_trades"] == 15
        assert data["error_count"] == 2
        assert data["degradation_tier"] == "FULL"
        assert data["trading_mode"] == "paper"

    def test_state_summary_degraded(self, client: TestClient, mock_runner: Mock) -> None:
        """Test state summary with degraded tier."""
        mock_runner.state.degradation_history = [
            DegradationRecord(
                timestamp=datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC),
                tier="MINIMAL",
                unavailable_services=["news", "sentiment"],
                confidence_adjustment=-0.2,
            )
        ]

        response = client.get("/state/summary")
        assert response.status_code == 200

        data = response.json()
        assert data["degradation_tier"] == "MINIMAL"

    def test_state_summary_no_errors(self, client: TestClient, mock_runner: Mock) -> None:
        """Test state summary with no errors."""
        mock_runner.state.errors = []

        response = client.get("/state/summary")
        assert response.status_code == 200

        data = response.json()
        assert data["error_count"] == 0


class TestConfigEndpoint:
    """Tests for /config endpoint."""

    def test_config(self, client: TestClient) -> None:
        """Test config endpoint."""
        response = client.get("/config")
        assert response.status_code == 200

        data = response.json()
        assert data["watchlist"] == ["AAPL", "TSLA"]
        assert data["interval_minutes"] == 30
        assert data["market_hours_only"] is True
        assert data["auto_trade"] is False
        assert data["trading_mode"] == "paper"
        assert data["pre_market_enabled"] is False

    def test_config_no_secrets(self, client: TestClient) -> None:
        """Test config endpoint does not expose secrets."""
        response = client.get("/config")
        assert response.status_code == 200

        data = response.json()
        # Verify no API keys or secrets in response
        assert "api_key" not in str(data).lower()
        assert "secret" not in str(data).lower()
        assert "password" not in str(data).lower()

    def test_config_pre_market_enabled(self, client: TestClient, mock_runner: Mock) -> None:
        """Test config endpoint with pre-market enabled."""
        mock_runner.config.schedule.enable_pre_market = True

        response = client.get("/config")
        assert response.status_code == 200

        data = response.json()
        assert data["pre_market_enabled"] is True


class TestCORS:
    """Tests for CORS configuration."""

    def test_cors_allowed_origin(self, client: TestClient) -> None:
        """Test CORS allows localhost:8050."""
        response = client.get(
            "/health",
            headers={"Origin": "http://localhost:8050"},
        )
        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == "http://localhost:8050"

    def test_cors_credentials(self, client: TestClient) -> None:
        """Test CORS allows credentials."""
        response = client.get(
            "/health",
            headers={"Origin": "http://localhost:8050"},
        )
        assert response.status_code == 200
        assert response.headers["access-control-allow-credentials"] == "true"
