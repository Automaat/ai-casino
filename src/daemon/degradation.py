"""Graceful API degradation policy for trading daemon."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from src.daemon.config import DaemonConfig

if TYPE_CHECKING:
    from src.daemon.health import HealthReport


class DegradationTier(StrEnum):
    """Degradation tier based on API availability."""

    FULL = "FULL"  # All services healthy
    DEGRADED = "DEGRADED"  # Some optional services down
    MINIMAL = "MINIMAL"  # Only cached/free data available
    HALTED = "HALTED"  # Critical services down (no market data or LLM)


class AgentType(StrEnum):
    """Agent types in trading workflow."""

    TECHNICAL = "technical"
    SENTIMENT = "sentiment"
    NEWS = "news"
    FUNDAMENTAL = "fundamental"
    COMPARATIVE = "comparative"
    WEB_RESEARCH = "web_research"
    SOCIAL = "social"
    BULLISH = "bullish"
    BEARISH = "bearish"


class AgentClassification(BaseModel):
    """Agent metadata for degradation policy."""

    agent: AgentType
    required: bool = Field(description="Whether agent is required for analysis")
    confidence_penalty_pct: float = Field(
        description="Confidence penalty % when agent unavailable",
        ge=0.0,
        le=100.0,
    )


class DegradationContext(BaseModel):
    """Context describing current degradation state."""

    tier: DegradationTier
    available_agents: set[AgentType]
    unavailable_services: list[str]
    confidence_adjustment: float = Field(
        description="Multiplier for confidence scores (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )
    halt_reason: str | None = None


class DegradationPolicy:
    """Evaluate degradation tier and agent availability from health reports."""

    def __init__(self, config: DaemonConfig) -> None:
        """Initialize degradation policy.

        Args:
            config: Daemon configuration
        """
        self.config = config
        self._classifications = self._build_classifications()
        self._service_to_agents = self._build_service_mapping()

    def _build_classifications(self) -> dict[AgentType, AgentClassification]:
        """Build agent classification registry with penalties."""
        return {
            AgentType.TECHNICAL: AgentClassification(
                agent=AgentType.TECHNICAL,
                required=True,
                confidence_penalty_pct=0.0,
            ),
            AgentType.SENTIMENT: AgentClassification(
                agent=AgentType.SENTIMENT,
                required=False,
                confidence_penalty_pct=10.0,
            ),
            AgentType.NEWS: AgentClassification(
                agent=AgentType.NEWS,
                required=False,
                confidence_penalty_pct=10.0,
            ),
            AgentType.FUNDAMENTAL: AgentClassification(
                agent=AgentType.FUNDAMENTAL,
                required=False,
                confidence_penalty_pct=10.0,
            ),
            AgentType.COMPARATIVE: AgentClassification(
                agent=AgentType.COMPARATIVE,
                required=False,
                confidence_penalty_pct=5.0,
            ),
            AgentType.WEB_RESEARCH: AgentClassification(
                agent=AgentType.WEB_RESEARCH,
                required=False,
                confidence_penalty_pct=0.0,
            ),
            AgentType.SOCIAL: AgentClassification(
                agent=AgentType.SOCIAL,
                required=False,
                confidence_penalty_pct=0.0,
            ),
            AgentType.BULLISH: AgentClassification(
                agent=AgentType.BULLISH,
                required=False,
                confidence_penalty_pct=0.0,
            ),
            AgentType.BEARISH: AgentClassification(
                agent=AgentType.BEARISH,
                required=False,
                confidence_penalty_pct=0.0,
            ),
        }

    def _build_service_mapping(self) -> dict[str, list[AgentType]]:
        """Map health check services to dependent agents."""
        return {
            "alpha_vantage": [AgentType.TECHNICAL],
            "marketaux": [AgentType.SENTIMENT, AgentType.NEWS],
            "finnhub": [AgentType.FUNDAMENTAL, AgentType.SOCIAL],
        }

    def evaluate_degradation(self, health_report: HealthReport | None) -> DegradationContext:
        """Determine degradation tier from health report.

        Args:
            health_report: Latest health check report, or None if unavailable

        Returns:
            DegradationContext with tier, available agents, and confidence adjustment
        """
        from src.daemon.health import ServiceStatus

        if not health_report:
            return DegradationContext(
                tier=DegradationTier.FULL,
                available_agents=set(AgentType),
                unavailable_services=[],
                confidence_adjustment=1.0,
            )

        unhealthy = [c.service for c in health_report.service_checks if c.status == ServiceStatus.UNHEALTHY]

        # Filter out non-configured LLM providers from unhealthy list to avoid stale health report confusion
        configured_llm_service = f"llm_{self.config.llm.provider}"
        unhealthy_relevant = [s for s in unhealthy if not s.startswith("llm_") or s == configured_llm_service]

        # Critical failures → HALTED
        if any("alpha_vantage" in s for s in unhealthy_relevant):
            return DegradationContext(
                tier=DegradationTier.HALTED,
                available_agents=set(),
                unavailable_services=unhealthy_relevant,
                confidence_adjustment=0.0,
                halt_reason="No market data (Alpha Vantage down)",
            )

        # Check if configured LLM provider is unhealthy
        if configured_llm_service in unhealthy_relevant:
            return DegradationContext(
                tier=DegradationTier.HALTED,
                available_agents=set(),
                unavailable_services=unhealthy_relevant,
                confidence_adjustment=0.0,
                halt_reason="LLM service unavailable",
            )

        # Map services → unavailable agents
        unavailable_agents = set()
        for service in unhealthy_relevant:
            agent_list = self._service_to_agents.get(service, [])
            unavailable_agents.update(agent_list)

        available_agents = set(AgentType) - unavailable_agents

        # Calculate confidence penalty (max 50%)
        penalty = sum(self._classifications[a].confidence_penalty_pct for a in unavailable_agents)
        confidence_adjustment = 1.0 - min(penalty, 50.0) / 100.0

        # Determine tier
        if len(unavailable_agents) == 0:
            tier = DegradationTier.FULL
        elif len(available_agents) >= 3:  # Technical + 2 others
            tier = DegradationTier.DEGRADED
        else:
            tier = DegradationTier.MINIMAL

        return DegradationContext(
            tier=tier,
            available_agents=available_agents,
            unavailable_services=unhealthy_relevant,
            confidence_adjustment=confidence_adjustment,
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return f"DegradationPolicy(agents={len(self._classifications)})"
