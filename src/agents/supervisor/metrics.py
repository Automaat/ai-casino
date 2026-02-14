"""Supervisor metrics domain models and collector."""

from datetime import datetime

from pydantic import BaseModel, Field


class SupervisorCycleMetrics(BaseModel):
    """Supervisor metrics for single analysis cycle."""

    # Identifiers
    id: str | None = None
    created_at: datetime | None = None
    workflow_id: str
    symbol: str
    timestamp: datetime

    # Routing decision
    required_analyses: list[str]
    optional_analyses: list[str]
    skip_analyses: dict[str, str]
    routing_reasoning: str

    # Execution stats
    total_workers: int
    required_workers: int
    optional_workers: int
    successful_workers: int
    failed_workers: int

    # Timing metrics in milliseconds
    routing_decision_ms: float
    group1_execution_ms: float
    research_execution_ms: float
    total_supervisor_overhead_ms: float
    worker_timings: dict[str, float]
    worker_errors: dict[str, str]

    # LLM usage metrics
    total_llm_calls: int
    total_cost_usd: float
    planning_fallback_used: bool
    synthesis_fallback_used: bool

    # Synthesis
    confidence_adjustment: float
    synthesis_reasoning: str

    # Efficiency
    parallel_efficiency_percent: float = Field(ge=0, le=100)
    timeout_triggered: bool

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"SupervisorCycleMetrics(workflow_id={self.workflow_id}, symbol={self.symbol}, "
            f"workers={self.total_workers}, efficiency={self.parallel_efficiency_percent:.1f}%)"
        )
