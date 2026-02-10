"""Integration tests for daemon degradation handling."""

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest

from src.daemon.config import DaemonConfig
from src.daemon.degradation import DegradationTier
from src.daemon.health import HealthReport, ServiceCheckResult, ServiceStatus
from src.daemon.runner import DaemonRunner


@pytest.fixture
def temp_health_dir(tmp_path):
    """Create temporary health directory."""
    health_dir = tmp_path / "health"
    health_dir.mkdir()
    return health_dir


@pytest.fixture
def daemon_config_with_health(tmp_path, temp_health_dir):
    """Create daemon config with health check enabled."""
    config_data = {
        "watchlist": ["AAPL"],
        "interval_minutes": 5,
        "market_hours_only": False,
        "auto_trade": False,
        "health": {
            "enabled": True,
            "check_interval_seconds": 5,
            "health_dir": str(temp_health_dir),
            "archive_dir": str(tmp_path / "archive"),
        },
        "state": {"state_file": str(tmp_path / "daemon-state.json")},
    }
    return DaemonConfig.model_validate(config_data)


def create_health_report(temp_health_dir, service_checks):
    """Create health report file."""
    now = datetime.now(UTC)
    report = HealthReport(
        timestamp=now,
        overall_status=(
            ServiceStatus.UNHEALTHY
            if any(c.status == ServiceStatus.UNHEALTHY for c in service_checks)
            else ServiceStatus.HEALTHY
        ),
        service_checks=service_checks,
        cleanup_results=[],
        total_duration_ms=100.0,
    )

    report_path = temp_health_dir / f"health-{now.strftime('%Y-%m-%d')}.json"
    with report_path.open("w") as f:
        json.dump(report.model_dump(mode="json"), f)

    return report


async def test_daemon_halts_on_alpha_vantage_down(daemon_config_with_health, temp_health_dir):
    """Verify daemon skips cycle when market data unavailable."""
    # Create health report with Alpha Vantage down
    create_health_report(
        temp_health_dir,
        [
            ServiceCheckResult(
                service="alpha_vantage",
                status=ServiceStatus.UNHEALTHY,
                message="Rate limited",
                duration_ms=100.0,
                checked_at=datetime.now(UTC),
            ),
        ],
    )

    runner = DaemonRunner(daemon_config_with_health)

    # Mock health check to avoid overwriting fake report with real check
    with patch.object(runner, "_maybe_run_health_check", new_callable=AsyncMock):
        with patch.object(runner, "_analyze_watchlist", new_callable=AsyncMock) as mock_analyze:
            sleep_time = await runner._run_cycle()

            # Should halt and return 60s retry
            assert sleep_time == 60
            mock_analyze.assert_not_called()

            # Verify degradation recorded in state
            assert runner.state.degradation_history
            assert runner.state.degradation_history[-1].tier == DegradationTier.HALTED.value
            halt_reason = runner.state.degradation_history[-1].halt_reason
            assert halt_reason is not None
            assert "market data" in halt_reason.lower()


async def test_daemon_halts_on_llm_down(daemon_config_with_health, temp_health_dir):
    """Verify daemon skips cycle when LLM unavailable."""
    # Create health report with LLM down
    create_health_report(
        temp_health_dir,
        [
            ServiceCheckResult(
                service="alpha_vantage",
                status=ServiceStatus.HEALTHY,
                message="OK",
                duration_ms=100.0,
                checked_at=datetime.now(UTC),
            ),
            ServiceCheckResult(
                service="llm_anthropic",
                status=ServiceStatus.UNHEALTHY,
                message="API key invalid",
                duration_ms=100.0,
                checked_at=datetime.now(UTC),
            ),
        ],
    )

    runner = DaemonRunner(daemon_config_with_health)

    with patch.object(runner, "_maybe_run_health_check", new_callable=AsyncMock):
        with patch.object(runner, "_analyze_watchlist", new_callable=AsyncMock) as mock_analyze:
            sleep_time = await runner._run_cycle()

            assert sleep_time == 60
            mock_analyze.assert_not_called()

            assert runner.state.degradation_history
            assert runner.state.degradation_history[-1].tier == DegradationTier.HALTED.value
            halt_reason = runner.state.degradation_history[-1].halt_reason
            assert halt_reason is not None
            assert "llm" in halt_reason.lower()


async def test_daemon_continues_in_degraded_mode(daemon_config_with_health, temp_health_dir):
    """Verify daemon continues with degraded analysis when optional services down."""
    # Create health report with Marketaux down
    create_health_report(
        temp_health_dir,
        [
            ServiceCheckResult(
                service="alpha_vantage",
                status=ServiceStatus.HEALTHY,
                message="OK",
                duration_ms=100.0,
                checked_at=datetime.now(UTC),
            ),
            ServiceCheckResult(
                service="marketaux",
                status=ServiceStatus.UNHEALTHY,
                message="Connection timeout",
                duration_ms=5000.0,
                checked_at=datetime.now(UTC),
            ),
            ServiceCheckResult(
                service="llm_ollama",
                status=ServiceStatus.HEALTHY,
                message="OK",
                duration_ms=100.0,
                checked_at=datetime.now(UTC),
            ),
        ],
    )

    runner = DaemonRunner(daemon_config_with_health)

    with patch.object(runner, "_maybe_run_health_check", new_callable=AsyncMock):
        with patch.object(runner, "_analyze_watchlist", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = []

            sleep_time = await runner._run_cycle()

            # Should continue normally
            assert sleep_time == daemon_config_with_health.interval_minutes * 60
            mock_analyze.assert_called_once()

            # Verify degradation context passed
            call_args = mock_analyze.call_args
            degradation_context = call_args[0][1]  # Second positional arg
            assert degradation_context is not None
            assert degradation_context.tier == DegradationTier.DEGRADED

            # Verify degradation recorded
            assert runner.state.degradation_history
            assert runner.state.degradation_history[-1].tier == DegradationTier.DEGRADED.value
            assert runner.state.degradation_history[-1].confidence_adjustment == 0.8


async def test_notification_sent_on_degradation(daemon_config_with_health, temp_health_dir):
    """Verify notification sent when degraded."""
    # Create health report with services down
    create_health_report(
        temp_health_dir,
        [
            ServiceCheckResult(
                service="alpha_vantage",
                status=ServiceStatus.HEALTHY,
                message="OK",
                duration_ms=100.0,
                checked_at=datetime.now(UTC),
            ),
            ServiceCheckResult(
                service="marketaux",
                status=ServiceStatus.UNHEALTHY,
                message="Down",
                duration_ms=100.0,
                checked_at=datetime.now(UTC),
            ),
        ],
    )

    runner = DaemonRunner(daemon_config_with_health)

    # Mock notification service
    mock_notification_service = AsyncMock()
    runner.notification_service = mock_notification_service

    with patch.object(runner, "_maybe_run_health_check", new_callable=AsyncMock):
        with patch.object(runner, "_analyze_watchlist", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = []

            await runner._run_cycle()

            # Verify notification sent
            mock_notification_service.notify.assert_awaited_once()
            await_args = mock_notification_service.notify.await_args
            message = await_args.args[1]
            assert "DEGRADED" in message.title or "degraded" in message.body.lower()


async def test_no_degradation_when_all_healthy(daemon_config_with_health, temp_health_dir):
    """Verify no degradation when all services healthy."""
    # Create healthy health report
    create_health_report(
        temp_health_dir,
        [
            ServiceCheckResult(
                service="alpha_vantage",
                status=ServiceStatus.HEALTHY,
                message="OK",
                duration_ms=100.0,
                checked_at=datetime.now(UTC),
            ),
            ServiceCheckResult(
                service="marketaux",
                status=ServiceStatus.HEALTHY,
                message="OK",
                duration_ms=100.0,
                checked_at=datetime.now(UTC),
            ),
            ServiceCheckResult(
                service="llm_ollama",
                status=ServiceStatus.HEALTHY,
                message="OK",
                duration_ms=100.0,
                checked_at=datetime.now(UTC),
            ),
        ],
    )

    runner = DaemonRunner(daemon_config_with_health)

    with patch.object(runner, "_maybe_run_health_check", new_callable=AsyncMock):
        with patch.object(runner, "_analyze_watchlist", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = []

            sleep_time = await runner._run_cycle()

            assert sleep_time == daemon_config_with_health.interval_minutes * 60

            # Verify no degradation recorded (or FULL tier recorded)
            # Note: Implementation may or may not record FULL tier
            if runner.state.degradation_history:
                assert runner.state.degradation_history[-1].tier == DegradationTier.FULL.value


async def test_degradation_history_limited_to_100(daemon_config_with_health, temp_health_dir):
    """Verify degradation history limited to 100 records."""
    # Pre-populate degradation history
    from src.daemon.degradation import DegradationContext

    runner = DaemonRunner(daemon_config_with_health)

    for _ in range(105):
        context = DegradationContext(
            tier=DegradationTier.DEGRADED,
            available_agents=set(),
            unavailable_services=["marketaux"],
            confidence_adjustment=0.8,
        )
        runner.state.record_degradation(context)

    # Should keep only last 100
    assert len(runner.state.degradation_history) == 100
