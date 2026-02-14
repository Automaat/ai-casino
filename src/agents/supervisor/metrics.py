"""Supervisor metrics domain models and collector."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from src.agents.supervisor.models import AnalysisRoutingDecision, AnalysisWeights


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


class SupervisorMetricsCollector:
    """Collects supervisor metrics during analysis cycle."""

    def __init__(self, workflow_id: str, symbol: str) -> None:
        """Initialize metrics collector.

        Args:
            workflow_id: Unique workflow identifier
            symbol: Stock ticker symbol
        """
        self.workflow_id = workflow_id
        self.symbol = symbol
        self.timestamp = datetime.now(UTC)

        # Routing
        self.routing_start: float = 0.0
        self.routing_decision_ms: float = 0.0
        self.required_analyses: list[str] = []
        self.optional_analyses: list[str] = []
        self.skip_analyses: dict[str, str] = {}
        self.routing_reasoning: str = ""
        self.planning_fallback_used: bool = False

        # Workers
        self.total_workers: int = 0
        self.required_workers: int = 0
        self.optional_workers: int = 0
        self.successful_workers: int = 0
        self.failed_workers: int = 0
        self.worker_timings: dict[str, float] = {}
        self.worker_errors: dict[str, str] = {}
        self.worker_start_times: dict[str, float] = {}

        # Execution timing
        self.group1_start: float = 0.0
        self.group1_execution_ms: float = 0.0
        self.research_start: float = 0.0
        self.research_execution_ms: float = 0.0
        self.total_supervisor_overhead_ms: float = 0.0
        self.timeout_triggered: bool = False

        # LLM usage
        self.total_llm_calls: int = 0
        self.total_cost_usd: float = 0.0
        self.synthesis_fallback_used: bool = False

        # Synthesis
        self.confidence_adjustment: float = 1.0
        self.synthesis_reasoning: str = ""

        # Efficiency
        self.parallel_efficiency_percent: float = 0.0

    def record_planning_start(self) -> None:
        """Record start of planning phase."""
        self.routing_start = time.perf_counter()

    def record_planning(
        self,
        decision: AnalysisRoutingDecision,
        fallback_used: bool,
        llm_calls: int = 1,
        cost_usd: float = 0.0,
    ) -> None:
        """Record planning decision.

        Args:
            decision: Routing decision from supervisor
            fallback_used: Whether fallback routing was used
            llm_calls: Number of LLM calls made (0 if fallback)
            cost_usd: LLM cost in USD
        """
        self.routing_decision_ms = (time.perf_counter() - self.routing_start) * 1000
        self.required_analyses = [a.value for a in decision.required_analyses]
        self.optional_analyses = [a.value for a in decision.optional_analyses]
        self.skip_analyses = {k.value: v for k, v in decision.skip_analyses.items()}
        self.routing_reasoning = decision.reasoning
        self.planning_fallback_used = fallback_used
        self.total_llm_calls += llm_calls
        self.total_cost_usd += cost_usd

    def record_group1_start(self) -> None:
        """Record start of group 1 execution."""
        self.group1_start = time.perf_counter()

    def record_group1_complete(self) -> None:
        """Record completion of group 1 execution."""
        self.group1_execution_ms = (time.perf_counter() - self.group1_start) * 1000

    def record_research_start(self) -> None:
        """Record start of research execution."""
        self.research_start = time.perf_counter()

    def record_research_complete(self) -> None:
        """Record completion of research execution."""
        self.research_execution_ms = (time.perf_counter() - self.research_start) * 1000

    def record_worker_start(self, worker_name: str, is_required: bool) -> None:
        """Record worker start.

        Args:
            worker_name: Name of worker
            is_required: Whether worker is required
        """
        self.worker_start_times[worker_name] = time.perf_counter()
        self.total_workers += 1
        if is_required:
            self.required_workers += 1
        else:
            self.optional_workers += 1

    def record_worker_complete(self, worker_name: str, result: object) -> None:
        """Record worker completion.

        Args:
            worker_name: Name of worker
            result: Worker result (None if failed)
        """
        if worker_name in self.worker_start_times:
            elapsed_ms = (time.perf_counter() - self.worker_start_times[worker_name]) * 1000
            self.worker_timings[worker_name] = elapsed_ms
            del self.worker_start_times[worker_name]

        if result is not None:
            self.successful_workers += 1
        else:
            self.failed_workers += 1

    def record_worker_error(self, worker_name: str, error: str) -> None:
        """Record worker error.

        Args:
            worker_name: Name of worker
            error: Error message
        """
        self.worker_errors[worker_name] = error
        self.failed_workers += 1

    def record_timeout(self) -> None:
        """Record timeout event."""
        self.timeout_triggered = True

    def record_synthesis(
        self, weights: AnalysisWeights, fallback_used: bool, llm_calls: int = 1, cost_usd: float = 0.0
    ) -> None:
        """Record synthesis results.

        Args:
            weights: Analysis weights from supervisor
            fallback_used: Whether fallback weighting was used
            llm_calls: Number of LLM calls made (0 if fallback)
            cost_usd: LLM cost in USD
        """
        self.confidence_adjustment = weights.confidence_adjustment
        self.synthesis_reasoning = weights.reasoning
        self.synthesis_fallback_used = fallback_used
        self.total_llm_calls += llm_calls
        self.total_cost_usd += cost_usd

    def calculate_efficiency(self) -> None:
        """Calculate parallel execution efficiency.

        Efficiency = (sum of worker times / actual wall time) / max_workers * 100
        """
        if not self.worker_timings:
            self.parallel_efficiency_percent = 0.0
            return

        sum_worker_times = sum(self.worker_timings.values())
        actual_parallel_time = self.group1_execution_ms + self.research_execution_ms

        if actual_parallel_time <= 0:
            self.parallel_efficiency_percent = 0.0
            return

        max_workers = len(self.worker_timings)
        if max_workers == 0:
            self.parallel_efficiency_percent = 0.0
            return

        efficiency = (sum_worker_times / actual_parallel_time) / max_workers * 100
        self.parallel_efficiency_percent = min(100.0, max(0.0, efficiency))

    def calculate_overhead(self) -> None:
        """Calculate total supervisor overhead."""
        self.total_supervisor_overhead_ms = (
            self.routing_decision_ms + self.group1_execution_ms + self.research_execution_ms
        )

    async def save(self, repository: object | None) -> SupervisorCycleMetrics | None:
        """Save metrics to repository.

        Args:
            repository: Optional SupervisorMetricsRepository to persist metrics

        Returns:
            Saved metrics or None if repository not provided
        """
        if not repository:
            return None

        from src.database.repositories.supervisor_metrics import SupervisorMetricsRepository

        if not isinstance(repository, SupervisorMetricsRepository):
            return None

        self.calculate_efficiency()
        self.calculate_overhead()

        metrics = SupervisorCycleMetrics(
            workflow_id=self.workflow_id,
            symbol=self.symbol,
            timestamp=self.timestamp,
            required_analyses=self.required_analyses,
            optional_analyses=self.optional_analyses,
            skip_analyses=self.skip_analyses,
            routing_reasoning=self.routing_reasoning,
            total_workers=self.total_workers,
            required_workers=self.required_workers,
            optional_workers=self.optional_workers,
            successful_workers=self.successful_workers,
            failed_workers=self.failed_workers,
            routing_decision_ms=self.routing_decision_ms,
            group1_execution_ms=self.group1_execution_ms,
            research_execution_ms=self.research_execution_ms,
            total_supervisor_overhead_ms=self.total_supervisor_overhead_ms,
            worker_timings=self.worker_timings,
            worker_errors=self.worker_errors,
            total_llm_calls=self.total_llm_calls,
            total_cost_usd=self.total_cost_usd,
            planning_fallback_used=self.planning_fallback_used,
            synthesis_fallback_used=self.synthesis_fallback_used,
            confidence_adjustment=self.confidence_adjustment,
            synthesis_reasoning=self.synthesis_reasoning,
            parallel_efficiency_percent=self.parallel_efficiency_percent,
            timeout_triggered=self.timeout_triggered,
        )

        return await repository.create(metrics)  # pyrefly: ignore[bad-return, bad-argument-type]

    def __repr__(self) -> str:
        """Return string representation."""
        return f"SupervisorMetricsCollector(workflow_id={self.workflow_id}, symbol={self.symbol})"
