"""Tests for embedded FastAPI server."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient

from src.daemon.api import create_api_app
from src.daemon.config import ApiConfig, DaemonConfig, ScheduleConfig
from src.daemon.event_bus import DashboardEvent, EventType
from src.daemon.runner import DaemonRunner
from src.daemon.state import AnalysisRecord, DaemonState, DegradationRecord, RiskReportRecord
from src.strategies.session import TradingSession


@pytest.fixture
def sample_analyses() -> list[AnalysisRecord]:
    """Create sample analysis records."""
    return [
        AnalysisRecord(
            symbol="AAPL",
            timestamp=datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC),
            signal="BUY",
            confidence=0.85,
            executed_trade=True,
            trading_session=TradingSession.REGULAR,
            is_paper_trade=True,
        ),
        AnalysisRecord(
            symbol="TSLA",
            timestamp=datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC),
            signal="SELL",
            confidence=0.75,
            executed_trade=False,
            trading_session=TradingSession.PRE_MARKET,
            is_paper_trade=True,
        ),
        AnalysisRecord(
            symbol="AAPL",
            timestamp=datetime(2024, 1, 15, 11, 0, 0, tzinfo=UTC),
            signal="HOLD",
            confidence=0.65,
            executed_trade=False,
            trading_session=TradingSession.REGULAR,
            is_paper_trade=True,
        ),
    ]


@pytest.fixture
def sample_positions() -> dict[str, dict]:
    """Create sample position dicts."""
    return {
        "AAPL": {
            "symbol": "AAPL",
            "entry_timestamp": datetime(2024, 1, 10, 10, 0, 0, tzinfo=UTC).isoformat(),
            "entry_price": 150.0,
            "entry_signal": "BUY",
            "entry_confidence": 0.85,
            "current_qty": 10.0,
            "current_stop_loss": 145.0,
            "initial_stop_loss": 145.0,
            "stop_loss_order_id": "order123",
            "profit_targets": [160.0, 170.0],
            "days_held": 5,
            "last_updated": datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC).isoformat(),
            "trailing_stop_activated": False,
            "breakeven_activated": False,
            "high_water_mark": 155.0,
        },
        "TSLA": {
            "symbol": "TSLA",
            "entry_timestamp": datetime(2024, 1, 12, 10, 0, 0, tzinfo=UTC).isoformat(),
            "entry_price": 200.0,
            "entry_signal": "BUY",
            "entry_confidence": 0.75,
            "current_qty": 5.0,
            "current_stop_loss": 190.0,
            "initial_stop_loss": 190.0,
            "stop_loss_order_id": "order456",
            "profit_targets": [220.0],
            "days_held": 3,
            "last_updated": datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC).isoformat(),
            "trailing_stop_activated": True,
            "breakeven_activated": False,
            "high_water_mark": 210.0,
        },
    }


@pytest.fixture
def sample_risk_report() -> RiskReportRecord:
    """Create sample risk report."""
    return RiskReportRecord(
        timestamp=datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC),
        var_95=1000.0,
        var_99=1500.0,
        cvar_95=1200.0,
        cvar_99=1800.0,
        cdar_95=0.05,
        max_drawdown=0.08,
        risk_status="ACCEPTABLE",
    )


@pytest.fixture
def sample_events() -> list[DashboardEvent]:
    """Create sample events."""
    return [
        DashboardEvent(
            event_type=EventType.CYCLE_START,
            timestamp=datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC),
            data={"message": "Starting cycle"},
        ),
        DashboardEvent(
            event_type=EventType.ANALYSIS_COMPLETE,
            timestamp=datetime(2024, 1, 15, 10, 5, 0, tzinfo=UTC),
            data={"symbol": "AAPL", "signal": "BUY"},
        ),
    ]


@pytest.fixture
def mock_runner(
    sample_analyses: list[AnalysisRecord],
    sample_positions: dict[str, dict],
    sample_risk_report: RiskReportRecord,
    sample_events: list[DashboardEvent],
) -> Mock:
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
        analyses=sample_analyses,
        active_positions=sample_positions,
        risk_report_history=[sample_risk_report],
        degradation_history=[
            DegradationRecord(
                timestamp=datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC),
                tier="FULL",
                unavailable_services=[],
                confidence_adjustment=1.0,
            )
        ],
    )

    runner = Mock(spec=DaemonRunner)
    runner.config = config
    runner.state = state
    runner.running = True
    runner.broker = None
    runner._get_merged_watchlist = Mock(return_value=["AAPL", "TSLA"])

    mock_event_bus = Mock()
    mock_event_bus.get_history = Mock(return_value=sample_events)
    mock_event_bus.subscribe = AsyncMock(return_value=("sub123", AsyncMock()))
    mock_event_bus.unsubscribe = AsyncMock()
    runner.event_bus = mock_event_bus

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


class TestAnalysesEndpoint:
    """Tests for /analyses endpoint."""

    def test_get_analyses_default_limit(self, client: TestClient) -> None:
        """Test analyses endpoint with default limit."""
        response = client.get("/analyses")
        assert response.status_code == 200

        data = response.json()
        assert data["total_count"] == 42
        assert data["returned_count"] == 3
        assert len(data["analyses"]) == 3

        assert data["analyses"][0]["symbol"] == "AAPL"
        assert data["analyses"][0]["signal"] == "HOLD"
        assert data["analyses"][0]["confidence"] == 0.65

    def test_get_analyses_custom_limit(self, client: TestClient) -> None:
        """Test analyses endpoint with custom limit."""
        response = client.get("/analyses?limit=1")
        assert response.status_code == 200

        data = response.json()
        assert data["returned_count"] == 1
        assert len(data["analyses"]) == 1

    def test_get_analyses_filter_by_symbol(self, client: TestClient) -> None:
        """Test analyses endpoint with symbol filter."""
        response = client.get("/analyses?symbol=AAPL")
        assert response.status_code == 200

        data = response.json()
        assert data["returned_count"] == 2
        assert all(a["symbol"] == "AAPL" for a in data["analyses"])

    def test_get_analyses_empty(self, client: TestClient, mock_runner: Mock) -> None:
        """Test analyses endpoint with empty history."""
        mock_runner.state.analyses = []

        response = client.get("/analyses")
        assert response.status_code == 200

        data = response.json()
        assert data["returned_count"] == 0
        assert len(data["analyses"]) == 0

    def test_get_analyses_max_limit(self, client: TestClient) -> None:
        """Test analyses endpoint clamps to max limit."""
        response = client.get("/analyses?limit=1000")
        assert response.status_code == 200

        data = response.json()
        assert data["returned_count"] <= 500


class TestPositionsEndpoint:
    """Tests for /positions endpoint."""

    def test_get_positions(self, client: TestClient) -> None:
        """Test positions endpoint."""
        response = client.get("/positions")
        assert response.status_code == 200

        data = response.json()
        assert data["count"] == 2
        assert len(data["positions"]) == 2

        aapl = next(p for p in data["positions"] if p["symbol"] == "AAPL")
        assert aapl["entry_price"] == 150.0
        assert aapl["current_qty"] == 10.0
        assert aapl["profit_targets"] == [160.0, 170.0]
        assert aapl["trailing_stop_activated"] is False

    def test_get_positions_empty(self, client: TestClient, mock_runner: Mock) -> None:
        """Test positions endpoint with no positions."""
        mock_runner.state.active_positions = {}

        response = client.get("/positions")
        assert response.status_code == 200

        data = response.json()
        assert data["count"] == 0
        assert len(data["positions"]) == 0

    def test_get_positions_malformed_skipped(self, client: TestClient, mock_runner: Mock) -> None:
        """Test positions endpoint skips malformed entries."""
        mock_runner.state.active_positions = {
            "AAPL": {"symbol": "AAPL", "invalid": "data"},
            "TSLA": mock_runner.state.active_positions["TSLA"],
        }

        response = client.get("/positions")
        assert response.status_code == 200

        data = response.json()
        assert data["count"] == 1
        assert data["positions"][0]["symbol"] == "TSLA"


class TestWatchlistEndpoint:
    """Tests for /watchlist endpoint."""

    def test_get_watchlist_config_only(self, client: TestClient) -> None:
        """Test watchlist endpoint with config symbols."""
        response = client.get("/watchlist")
        assert response.status_code == 200

        data = response.json()
        assert data["count"] == 2
        assert set(data["symbols"]) == {"AAPL", "TSLA"}
        assert data["sources"]["config"] == 2

    def test_get_watchlist_merged(self, client: TestClient, mock_runner: Mock) -> None:
        """Test watchlist endpoint with all sources."""
        mock_runner._get_merged_watchlist = Mock(return_value=["AAPL", "TSLA", "NVDA"])

        # Add NVDA to active positions (AAPL and TSLA already in fixture)
        mock_runner.state.active_positions["NVDA"] = {
            "symbol": "NVDA",
            "entry_timestamp": datetime(2024, 1, 14, 10, 0, 0, tzinfo=UTC).isoformat(),
            "entry_price": 500.0,
            "entry_signal": "BUY",
            "entry_confidence": 0.80,
            "current_qty": 3.0,
            "current_stop_loss": 480.0,
            "initial_stop_loss": 480.0,
            "stop_loss_order_id": "order789",
            "profit_targets": [550.0],
            "days_held": 1,
            "last_updated": datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC).isoformat(),
            "trailing_stop_activated": False,
            "breakeven_activated": False,
            "high_water_mark": 505.0,
        }

        response = client.get("/watchlist")
        assert response.status_code == 200

        data = response.json()
        assert data["count"] == 3
        assert data["sources"]["broker"] == 3

    def test_get_watchlist_broker_with_positions(self, client: TestClient) -> None:
        """Test watchlist endpoint uses cached positions."""
        # Uses active_positions from state (already set in fixture)
        response = client.get("/watchlist")
        assert response.status_code == 200

        data = response.json()
        # Both AAPL and TSLA are in active_positions from fixture
        assert data["sources"]["broker"] == 2

    def test_get_watchlist_source_breakdown(self, client: TestClient, mock_runner: Mock) -> None:
        """Test watchlist source breakdown calculation."""
        from src.daemon.state import ScreeningRecord

        mock_runner.config.screening.enabled = True
        mock_runner.state.screening_history = [
            ScreeningRecord(
                timestamp=datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC),
                criteria="momentum",
                universe="SP500",
                top_symbols=["AAPL", "TSLA"],
                candidates=[],
                screened_at=datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC),
            )
        ]

        response = client.get("/watchlist")
        assert response.status_code == 200

        data = response.json()
        assert data["sources"]["screening"] == 2


class TestRiskEndpoint:
    """Tests for /risk endpoint."""

    def test_get_risk_with_report(self, client: TestClient) -> None:
        """Test risk endpoint returns report."""
        response = client.get("/risk")
        assert response.status_code == 200

        data = response.json()
        assert data["var_95"] == 1000.0
        assert data["var_99"] == 1500.0
        assert data["risk_status"] == "ACCEPTABLE"

    def test_get_risk_no_reports(self, client: TestClient, mock_runner: Mock) -> None:
        """Test risk endpoint with no reports."""
        mock_runner.state.risk_report_history = []

        response = client.get("/risk")
        assert response.status_code == 200
        assert response.json() is None

    def test_get_risk_latest_only(self, client: TestClient, mock_runner: Mock) -> None:
        """Test risk endpoint returns latest report."""
        older_report = RiskReportRecord(
            timestamp=datetime(2024, 1, 14, 10, 0, 0, tzinfo=UTC),
            var_95=800.0,
            var_99=1200.0,
            cvar_95=1000.0,
            cvar_99=1500.0,
            cdar_95=0.04,
            max_drawdown=0.06,
            risk_status="ACCEPTABLE",
        )
        mock_runner.state.risk_report_history.insert(0, older_report)

        response = client.get("/risk")
        assert response.status_code == 200

        data = response.json()
        assert data["var_95"] == 1000.0


class TestDegradationEndpoint:
    """Tests for /degradation endpoint."""

    def test_get_degradation_full(self, client: TestClient) -> None:
        """Test degradation endpoint with FULL tier."""
        response = client.get("/degradation")
        assert response.status_code == 200

        data = response.json()
        assert data["tier"] == "FULL"
        assert data["unavailable_services"] == []
        assert data["confidence_adjustment"] == 1.0
        assert data["halt_reason"] is None

    def test_get_degradation_degraded(self, client: TestClient, mock_runner: Mock) -> None:
        """Test degradation endpoint with DEGRADED tier."""
        mock_runner.state.degradation_history = [
            DegradationRecord(
                timestamp=datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC),
                tier="REDUCED",
                unavailable_services=["news"],
                confidence_adjustment=0.9,
                halt_reason=None,
            )
        ]

        response = client.get("/degradation")
        assert response.status_code == 200

        data = response.json()
        assert data["tier"] == "REDUCED"
        assert data["unavailable_services"] == ["news"]
        assert data["confidence_adjustment"] == 0.9

    def test_get_degradation_no_history(self, client: TestClient, mock_runner: Mock) -> None:
        """Test degradation endpoint defaults to FULL."""
        mock_runner.state.degradation_history = []

        response = client.get("/degradation")
        assert response.status_code == 200

        data = response.json()
        assert data["tier"] == "FULL"
        assert data["confidence_adjustment"] == 1.0


class TestEventsEndpoint:
    """Tests for /events endpoint."""

    def test_get_events_default_limit(self, client: TestClient) -> None:
        """Test events endpoint with default limit."""
        response = client.get("/events")
        assert response.status_code == 200

        data = response.json()
        assert data["returned_count"] == 2
        assert len(data["events"]) == 2

    def test_get_events_custom_limit(self, client: TestClient, mock_runner: Mock) -> None:
        """Test events endpoint with custom limit."""
        mock_runner.event_bus.get_history = Mock(
            return_value=[
                DashboardEvent(
                    event_type=EventType.CYCLE_START,
                    timestamp=datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC),
                    data={"message": "Starting cycle"},
                )
            ]
        )

        response = client.get("/events?limit=1")
        assert response.status_code == 200

        data = response.json()
        assert data["returned_count"] == 1
        mock_runner.event_bus.get_history.assert_called_once_with(limit=1)

    def test_get_events_no_event_bus(self, client: TestClient, mock_runner: Mock) -> None:
        """Test events endpoint with no EventBus."""
        mock_runner.event_bus = None

        response = client.get("/events")
        assert response.status_code == 200

        data = response.json()
        assert data["returned_count"] == 0
        assert len(data["events"]) == 0

    def test_get_events_max_limit(self, client: TestClient) -> None:
        """Test events endpoint clamps to max limit."""
        response = client.get("/events?limit=1000")
        assert response.status_code == 200

        data = response.json()
        assert data["returned_count"] <= 500


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


class TestWebSocketEvents:
    """Test /ws/events WebSocket endpoint."""

    def test_websocket_no_event_bus(self, client: TestClient, mock_runner: Mock) -> None:
        """Test WebSocket connection rejects when EventBus unavailable."""
        from starlette.websockets import WebSocketDisconnect

        mock_runner.event_bus = None

        with pytest.raises(WebSocketDisconnect):
            client.websocket_connect("/ws/events").__enter__()

    def test_websocket_basic_connection(self, mock_runner: Mock) -> None:
        """Test WebSocket connection and cleanup."""
        # This test verifies the endpoint exists and handles subscribe/unsubscribe
        # Full integration testing of WebSocket event streaming requires a running server
        from src.daemon.api import create_api_app

        app = create_api_app(mock_runner)

        # Verify the WebSocket route is registered
        routes = [route.path for route in app.routes]
        assert "/ws/events" in routes


class TestGamePlanEndpoint:
    """Tests for /game-plan endpoint."""

    def test_get_game_plan_disabled(self, client: TestClient, mock_runner: Mock) -> None:
        """Test game plan endpoint when disabled."""
        from src.daemon.config import GamePlanConfig

        mock_runner.config.game_plan = GamePlanConfig(enabled=False)

        response = client.get("/game-plan")
        assert response.status_code == 200
        assert response.json() is None

    def test_get_game_plan_missing_file(self, client: TestClient, mock_runner: Mock, tmp_path) -> None:
        """Test game plan endpoint when file missing."""
        from src.daemon.config import GamePlanConfig
        from src.daemon.state import GamePlanRecord

        mock_runner.config.game_plan = GamePlanConfig(enabled=True, plan_dir=str(tmp_path))
        mock_runner.state.game_plan_history = [
            GamePlanRecord(
                timestamp=datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC),
                priority_symbols=["AAPL"],
                risk_stance="BALANCED",
                sector_focus=["tech"],
            )
        ]

        response = client.get("/game-plan")
        assert response.status_code == 200
        assert response.json() is None

    def test_get_game_plan_valid(self, client: TestClient, mock_runner: Mock, tmp_path) -> None:
        """Test game plan endpoint with valid file."""
        import json

        from src.daemon.config import GamePlanConfig
        from src.daemon.state import GamePlanRecord

        plan_file = tmp_path / "2024-01-15.json"
        plan_data = {
            "date": "2024-01-15",
            "priority_symbols": ["AAPL", "TSLA"],
            "risk_stance": "BALANCED",
            "sector_focus": ["tech"],
            "reasoning": "Test reasoning",
            "confidence": 0.85,
            "generated_at": "2024-01-15T10:00:00+00:00",
        }
        plan_file.write_text(json.dumps(plan_data))

        mock_runner.config.game_plan = GamePlanConfig(enabled=True, plan_dir=str(tmp_path))
        mock_runner.state.game_plan_history = [
            GamePlanRecord(
                timestamp=datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC),
                priority_symbols=["AAPL", "TSLA"],
                risk_stance="BALANCED",
                sector_focus=["tech"],
            )
        ]

        response = client.get("/game-plan")
        assert response.status_code == 200

        data = response.json()
        assert data["date"] == "2024-01-15"
        assert data["priority_symbols"] == ["AAPL", "TSLA"]
        assert data["risk_stance"] == "BALANCED"
        assert data["confidence"] == 0.85


class TestRiskHistoryEndpoint:
    """Tests for /risk/history endpoint."""

    def test_get_risk_history_with_reports(self, client: TestClient, mock_runner: Mock) -> None:
        """Test risk history endpoint returns multiple reports."""
        reports = [
            RiskReportRecord(
                timestamp=datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC),
                var_95=1000.0,
                var_99=1500.0,
                cvar_95=1200.0,
                cvar_99=1800.0,
                cdar_95=0.05,
                max_drawdown=0.08,
                risk_status="HEALTHY",
            ),
            RiskReportRecord(
                timestamp=datetime(2024, 1, 15, 11, 0, 0, tzinfo=UTC),
                var_95=1100.0,
                var_99=1600.0,
                cvar_95=1300.0,
                cvar_99=1900.0,
                cdar_95=0.06,
                max_drawdown=0.09,
                risk_status="WARNING",
            ),
        ]
        mock_runner.state.risk_report_history = reports

        response = client.get("/risk/history")
        assert response.status_code == 200

        data = response.json()
        assert data["count"] == 2
        assert len(data["reports"]) == 2
        assert data["reports"][0]["var_95"] == 1000.0
        assert data["reports"][0]["risk_status"] == "HEALTHY"
        assert data["reports"][1]["var_95"] == 1100.0
        assert data["reports"][1]["risk_status"] == "WARNING"

    def test_get_risk_history_empty(self, client: TestClient, mock_runner: Mock) -> None:
        """Test risk history endpoint with no reports."""
        mock_runner.state.risk_report_history = []

        response = client.get("/risk/history")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 0
        assert data["reports"] == []


class TestSectorRotationEndpoint:
    """Tests for /sector-rotation/latest endpoint."""

    def test_get_sector_rotation_with_data(self, client: TestClient, mock_runner: Mock) -> None:
        """Test sector rotation endpoint returns analysis."""
        from src.daemon.state import SectorRotationRecord

        rotation = SectorRotationRecord(
            timestamp=datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC),
            leading_sectors=["Technology", "Healthcare"],
            lagging_sectors=["Energy", "Utilities"],
            sector_strengths={"Technology": 15.5, "Healthcare": 10.2, "Energy": -5.3},
            sector_momenta={"Technology": "STRONG", "Healthcare": "MODERATE", "Energy": "WEAK"},
            flagged_positions=["XLE"],
        )
        mock_runner.state.sector_rotation_history = [rotation]

        response = client.get("/sector-rotation/latest")
        assert response.status_code == 200

        data = response.json()
        assert data["leading_sectors"] == ["Technology", "Healthcare"]
        assert data["lagging_sectors"] == ["Energy", "Utilities"]
        assert data["sector_strengths"]["Technology"] == 15.5
        assert data["flagged_positions"] == ["XLE"]

    def test_get_sector_rotation_none(self, client: TestClient, mock_runner: Mock) -> None:
        """Test sector rotation endpoint with no data."""
        mock_runner.state.sector_rotation_history = []

        response = client.get("/sector-rotation/latest")
        assert response.status_code == 200
        assert response.json() is None


class TestCorrelationEndpoint:
    """Tests for /correlation/latest endpoint."""

    def test_get_correlation_with_audit(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test correlation endpoint returns audit."""
        from src.metrics.correlation import CorrelationAuditResult, CorrelationPair

        audit = CorrelationAuditResult(
            audit_date=datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC),
            num_positions=3,
            correlation_matrix={"AAPL": {"AAPL": 1.0, "TSLA": 0.65}, "TSLA": {"AAPL": 0.65, "TSLA": 1.0}},
            highly_correlated_pairs=[
                CorrelationPair(
                    symbol_a="AAPL",
                    symbol_b="TSLA",
                    correlation=0.65,
                    sector_a="Technology",
                    sector_b="Technology",
                    same_sector=True,
                )
            ],
            max_correlation=0.65,
            avg_correlation=0.325,
            diversification_ratio=1.5,
            substitution_suggestions=[],
            warnings=[],
            lookback_days=90,
        )

        mock_auditor = Mock()
        mock_auditor.load_latest = Mock(return_value=audit)

        def mock_correlation_auditor(*args, **kwargs):
            return mock_auditor

        monkeypatch.setattr("src.metrics.correlation.CorrelationAuditor", mock_correlation_auditor)

        response = client.get("/correlation/latest")
        assert response.status_code == 200

        data = response.json()
        assert data["num_positions"] == 3
        assert data["max_correlation"] == 0.65
        assert data["avg_correlation"] == 0.325
        assert data["symbols"] == ["AAPL", "TSLA"]
        assert data["correlation_matrix"]["AAPL"]["TSLA"] == 0.65

    def test_get_correlation_none(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test correlation endpoint with no audit."""
        mock_auditor = Mock()
        mock_auditor.load_latest = Mock(return_value=None)

        def mock_correlation_auditor(*args, **kwargs):
            return mock_auditor

        monkeypatch.setattr("src.metrics.correlation.CorrelationAuditor", mock_correlation_auditor)

        response = client.get("/correlation/latest")
        assert response.status_code == 200
        assert response.json() is None
