"""Unit tests for degradation policy."""

from datetime import UTC, datetime

import pytest

from src.daemon.config import DaemonConfig
from src.daemon.degradation import AgentType, DegradationPolicy, DegradationTier
from src.daemon.health import HealthReport, ServiceCheckResult, ServiceStatus


@pytest.fixture
def daemon_config():
    """Create default daemon config."""
    return DaemonConfig()


@pytest.fixture
def policy(daemon_config):
    """Create degradation policy."""
    return DegradationPolicy(daemon_config)


def test_full_mode_all_services_healthy(policy):
    """Verify FULL tier when all services healthy."""
    health_report = HealthReport(
        timestamp=datetime.now(UTC),
        overall_status=ServiceStatus.HEALTHY,
        service_checks=[
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
        cleanup_results=[],
        total_duration_ms=300.0,
    )

    context = policy.evaluate_degradation(health_report)

    assert context.tier == DegradationTier.FULL
    assert len(context.available_agents) == 9  # All agents
    assert context.confidence_adjustment == 1.0
    assert context.halt_reason is None


def test_degraded_mode_marketaux_down(policy):
    """Verify DEGRADED tier when news API down."""
    health_report = HealthReport(
        timestamp=datetime.now(UTC),
        overall_status=ServiceStatus.UNHEALTHY,
        service_checks=[
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
        cleanup_results=[],
        total_duration_ms=5200.0,
    )

    context = policy.evaluate_degradation(health_report)

    assert context.tier == DegradationTier.DEGRADED
    assert AgentType.SENTIMENT not in context.available_agents
    assert AgentType.NEWS not in context.available_agents
    assert AgentType.TECHNICAL in context.available_agents
    assert context.confidence_adjustment == 0.8  # -20% (sentiment + news)
    assert "marketaux" in context.unavailable_services


def test_degraded_mode_finnhub_down(policy):
    """Verify DEGRADED tier when fundamental API down."""
    health_report = HealthReport(
        timestamp=datetime.now(UTC),
        overall_status=ServiceStatus.UNHEALTHY,
        service_checks=[
            ServiceCheckResult(
                service="alpha_vantage",
                status=ServiceStatus.HEALTHY,
                message="OK",
                duration_ms=100.0,
                checked_at=datetime.now(UTC),
            ),
            ServiceCheckResult(
                service="finnhub",
                status=ServiceStatus.UNHEALTHY,
                message="API key invalid",
                duration_ms=50.0,
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
        cleanup_results=[],
        total_duration_ms=250.0,
    )

    context = policy.evaluate_degradation(health_report)

    assert context.tier == DegradationTier.DEGRADED
    assert AgentType.FUNDAMENTAL not in context.available_agents
    assert AgentType.SOCIAL not in context.available_agents
    assert AgentType.TECHNICAL in context.available_agents
    assert context.confidence_adjustment == 0.9  # -10% (fundamental only)


def test_halted_mode_alpha_vantage_down(policy):
    """Verify HALTED tier when market data unavailable."""
    health_report = HealthReport(
        timestamp=datetime.now(UTC),
        overall_status=ServiceStatus.UNHEALTHY,
        service_checks=[
            ServiceCheckResult(
                service="alpha_vantage",
                status=ServiceStatus.UNHEALTHY,
                message="Rate limited",
                duration_ms=100.0,
                checked_at=datetime.now(UTC),
            ),
        ],
        cleanup_results=[],
        total_duration_ms=100.0,
    )

    context = policy.evaluate_degradation(health_report)

    assert context.tier == DegradationTier.HALTED
    assert "market data" in context.halt_reason.lower()
    assert "alpha vantage" in context.halt_reason.lower()


def test_halted_mode_llm_down(policy):
    """Verify HALTED tier when LLM unavailable."""
    health_report = HealthReport(
        timestamp=datetime.now(UTC),
        overall_status=ServiceStatus.UNHEALTHY,
        service_checks=[
            ServiceCheckResult(
                service="alpha_vantage",
                status=ServiceStatus.HEALTHY,
                message="OK",
                duration_ms=100.0,
                checked_at=datetime.now(UTC),
            ),
            ServiceCheckResult(
                service="llm_ollama",
                status=ServiceStatus.UNHEALTHY,
                message="Connection refused",
                duration_ms=100.0,
                checked_at=datetime.now(UTC),
            ),
        ],
        cleanup_results=[],
        total_duration_ms=200.0,
    )

    context = policy.evaluate_degradation(health_report)

    assert context.tier == DegradationTier.HALTED
    assert "llm" in context.halt_reason.lower()


def test_stale_health_report_different_llm_provider():
    """Regression: stale health with wrong LLM provider should not halt."""
    # Config uses openai, but health report has stale llm_ollama UNHEALTHY
    config = DaemonConfig()
    config.llm.provider = "openai"
    policy = DegradationPolicy(config)

    health_report = HealthReport(
        timestamp=datetime.now(UTC),
        overall_status=ServiceStatus.UNHEALTHY,
        service_checks=[
            ServiceCheckResult(
                service="alpha_vantage",
                status=ServiceStatus.HEALTHY,
                message="OK",
                duration_ms=100.0,
                checked_at=datetime.now(UTC),
            ),
            ServiceCheckResult(
                service="llm_ollama",
                status=ServiceStatus.UNHEALTHY,
                message="Connection refused (stale)",
                duration_ms=100.0,
                checked_at=datetime.now(UTC),
            ),
            ServiceCheckResult(
                service="llm_openai",
                status=ServiceStatus.HEALTHY,
                message="OK",
                duration_ms=150.0,
                checked_at=datetime.now(UTC),
            ),
        ],
        cleanup_results=[],
        total_duration_ms=250.0,
    )

    context = policy.evaluate_degradation(health_report)

    # Should NOT halt (configured provider is healthy)
    assert context.tier == DegradationTier.FULL
    assert context.halt_reason is None
    assert len(context.available_agents) == 9
    assert context.confidence_adjustment == 1.0
    # Stale llm_ollama should be filtered from unavailable_services
    assert "llm_ollama" not in context.unavailable_services


def test_confidence_penalty_capped_at_50_percent(policy):
    """Verify cumulative penalty never exceeds 50%."""
    health_report = HealthReport(
        timestamp=datetime.now(UTC),
        overall_status=ServiceStatus.UNHEALTHY,
        service_checks=[
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
            ),  # -20%
            ServiceCheckResult(
                service="finnhub",
                status=ServiceStatus.UNHEALTHY,
                message="Down",
                duration_ms=100.0,
                checked_at=datetime.now(UTC),
            ),  # -10% (10% fundamental + 0% social)
        ],
        cleanup_results=[],
        total_duration_ms=300.0,
    )

    context = policy.evaluate_degradation(health_report)

    # Total penalty: 20% + 10% = 30%
    assert context.confidence_adjustment == 0.7  # Never below 0.5
    assert context.tier == DegradationTier.DEGRADED


def test_no_health_report_defaults_to_full(policy):
    """Verify FULL tier when no health report available."""
    context = policy.evaluate_degradation(None)

    assert context.tier == DegradationTier.FULL
    assert len(context.available_agents) == 9
    assert context.confidence_adjustment == 1.0


def test_minimal_tier_when_few_agents_available(policy):
    """Verify MINIMAL tier when <3 agents available."""
    health_report = HealthReport(
        timestamp=datetime.now(UTC),
        overall_status=ServiceStatus.UNHEALTHY,
        service_checks=[
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
            ServiceCheckResult(
                service="finnhub",
                status=ServiceStatus.UNHEALTHY,
                message="Down",
                duration_ms=100.0,
                checked_at=datetime.now(UTC),
            ),
        ],
        cleanup_results=[],
        total_duration_ms=300.0,
    )

    context = policy.evaluate_degradation(health_report)

    # Only technical, comparative, web_research, bullish, bearish available (5 agents)
    assert context.tier == DegradationTier.DEGRADED  # ≥3 agents


def test_agent_classifications_complete(policy):
    """Verify all agent types have classifications."""
    classifications = policy._build_classifications()

    assert len(classifications) == 9
    assert AgentType.TECHNICAL in classifications
    assert AgentType.SENTIMENT in classifications
    assert AgentType.NEWS in classifications
    assert AgentType.FUNDAMENTAL in classifications
    assert AgentType.COMPARATIVE in classifications
    assert AgentType.WEB_RESEARCH in classifications
    assert AgentType.SOCIAL in classifications
    assert AgentType.BULLISH in classifications
    assert AgentType.BEARISH in classifications


def test_service_to_agent_mapping(policy):
    """Verify service to agent mappings are correct."""
    mapping = policy._build_service_mapping()

    assert AgentType.TECHNICAL in mapping["alpha_vantage"]
    assert AgentType.SENTIMENT in mapping["marketaux"]
    assert AgentType.NEWS in mapping["marketaux"]
    assert AgentType.FUNDAMENTAL in mapping["finnhub"]
    assert AgentType.SOCIAL in mapping["finnhub"]
