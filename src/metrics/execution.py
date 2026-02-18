"""Execution performance metrics for workflow instrumentation."""

import os
import time
import uuid
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger
from pydantic import BaseModel

current_agent: ContextVar[str | None] = ContextVar("current_agent", default=None)
current_collector: ContextVar[ExecutionMetricsCollector | None] = ContextVar(
    "current_collector", default=None
)

# Cost per 1M tokens (input, output) — hardcoded estimates
_PRICING: dict[str, tuple[float, float]] = {
    "anthropic/claude-sonnet-4-20250514": (3.0, 15.0),
    "anthropic/claude-haiku-4-20250414": (0.8, 4.0),
    "openai/gpt-4o": (2.5, 10.0),
    "openai/gpt-4o-mini": (0.15, 0.6),
    "openai/gpt-5": (10.0, 30.0),
}


def is_metrics_enabled() -> bool:
    """Check EXECUTION_METRICS env var."""
    return os.getenv("EXECUTION_METRICS", "true").lower() == "true"


class LLMUsageStats(BaseModel):
    """Token usage stats from a single LLM call."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None


class LLMCallMetric(BaseModel):
    """Metrics for a single LLM API call."""

    timestamp: datetime
    agent_name: str
    method: str
    provider: str
    model: str
    latency_ms: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    estimated_cost_usd: float | None = None
    success: bool
    error: str | None = None


class SubOperationMetric(BaseModel):
    """Timing for non-LLM operations (data fetches, model inference, indicator calc)."""

    name: str
    latency_ms: float
    metadata: dict[str, str | int | float] | None = None


class AgentTimingMetric(BaseModel):
    """Timing for a complete agent execution."""

    agent_name: str
    latency_ms: float
    llm_calls: int


class PipelineStageMetric(BaseModel):
    """Timing for a pipeline stage."""

    stage: str
    latency_ms: float


class WorkflowExecutionMetrics(BaseModel):
    """Complete metrics for one workflow execution."""

    workflow_id: str
    symbol: str
    timestamp: datetime
    total_latency_ms: float
    llm_calls: list[LLMCallMetric]
    sub_operations: list[SubOperationMetric]
    agent_timings: list[AgentTimingMetric]
    pipeline_stages: list[PipelineStageMetric]
    total_input_tokens: int
    total_output_tokens: int
    total_cache_creation_tokens: int = 0
    total_cache_read_tokens: int = 0
    total_estimated_cost_usd: float
    provider: str
    model: str


class ExecutionMetricsCollector:
    """Collects execution metrics during a workflow run."""

    def __init__(self, symbol: str, provider: str, model: str) -> None:
        """Initialize collector for a workflow run.

        Args:
            symbol: Stock ticker being analyzed
            provider: LLM provider name
            model: LLM model name
        """
        self._workflow_id = str(uuid.uuid4())
        self._symbol = symbol
        self._provider = provider
        self._model = model
        self._start_time = time.perf_counter()
        self._llm_calls: list[LLMCallMetric] = []
        self._sub_operations: list[SubOperationMetric] = []
        self._agent_timings: list[AgentTimingMetric] = []
        self._pipeline_stages: list[PipelineStageMetric] = []
        self._agent_call_counts: dict[str, int] = defaultdict(int)

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"ExecutionMetricsCollector(symbol={self._symbol}, provider={self._provider}, "
            f"model={self._model})"
        )

    @property
    def workflow_id(self) -> str:
        """Get workflow ID for this execution.

        Returns:
            Workflow ID string
        """
        return self._workflow_id

    def record_llm_call(
        self,
        method: str,
        latency_ms: float,
        usage: LLMUsageStats | None,
        success: bool,
        error: str | None = None,
    ) -> None:
        """Record an LLM API call.

        Args:
            method: LLM method name (acomplete, astructured, etc.)
            latency_ms: Wall-clock latency in milliseconds
            usage: Token usage stats
            success: Whether the call succeeded
            error: Error message if failed
        """
        agent_name = current_agent.get() or "unknown"
        self._agent_call_counts[agent_name] += 1

        input_tokens = usage.input_tokens if usage else None
        output_tokens = usage.output_tokens if usage else None
        cache_creation = usage.cache_creation_input_tokens if usage else None
        cache_read = usage.cache_read_input_tokens if usage else None
        cost = self._estimate_cost(
            self._provider,
            self._model,
            input_tokens,
            output_tokens,
            cache_tokens=(cache_read, cache_creation),
        )

        self._llm_calls.append(
            LLMCallMetric(
                timestamp=datetime.now(UTC),
                agent_name=agent_name,
                method=method,
                provider=self._provider,
                model=self._model,
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_creation_input_tokens=cache_creation,
                cache_read_input_tokens=cache_read,
                estimated_cost_usd=cost,
                success=success,
                error=error,
            )
        )

    def record_sub_operation(
        self,
        name: str,
        latency_ms: float,
        metadata: dict[str, str | int | float] | None = None,
    ) -> None:
        """Record a non-LLM sub-operation timing.

        Args:
            name: Operation name (e.g. "market_data_fetch")
            latency_ms: Wall-clock latency in milliseconds
            metadata: Optional key-value metadata
        """
        self._sub_operations.append(SubOperationMetric(name=name, latency_ms=latency_ms, metadata=metadata))

    def record_agent_timing(self, agent_name: str, latency_ms: float) -> None:
        """Record total agent execution time.

        Args:
            agent_name: Agent name
            latency_ms: Total wall-clock latency in milliseconds
        """
        self._agent_timings.append(
            AgentTimingMetric(
                agent_name=agent_name,
                latency_ms=latency_ms,
                llm_calls=self._agent_call_counts.get(agent_name, 0),
            )
        )

    def record_pipeline_stage(self, stage: str, latency_ms: float) -> None:
        """Record pipeline stage timing.

        Args:
            stage: Stage name (e.g. "fetch_data", "analyses")
            latency_ms: Wall-clock latency in milliseconds
        """
        self._pipeline_stages.append(PipelineStageMetric(stage=stage, latency_ms=latency_ms))

    def finalize(self) -> WorkflowExecutionMetrics:
        """Build final metrics summary.

        Returns:
            Complete workflow execution metrics
        """
        total_latency_ms = (time.perf_counter() - self._start_time) * 1000
        total_input = sum(c.input_tokens for c in self._llm_calls if c.input_tokens)
        total_output = sum(c.output_tokens for c in self._llm_calls if c.output_tokens)
        total_cache_creation = sum(
            c.cache_creation_input_tokens for c in self._llm_calls if c.cache_creation_input_tokens
        )
        total_cache_read = sum(
            c.cache_read_input_tokens for c in self._llm_calls if c.cache_read_input_tokens
        )
        total_cost = sum(c.estimated_cost_usd for c in self._llm_calls if c.estimated_cost_usd)

        return WorkflowExecutionMetrics(
            workflow_id=self._workflow_id,
            symbol=self._symbol,
            timestamp=datetime.now(UTC),
            total_latency_ms=total_latency_ms,
            llm_calls=self._llm_calls,
            sub_operations=self._sub_operations,
            agent_timings=self._agent_timings,
            pipeline_stages=self._pipeline_stages,
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            total_cache_creation_tokens=total_cache_creation,
            total_cache_read_tokens=total_cache_read,
            total_estimated_cost_usd=total_cost,
            provider=self._provider,
            model=self._model,
        )

    @staticmethod
    def _estimate_cost(
        provider: str,
        model: str,
        input_tokens: int | None,
        output_tokens: int | None,
        cache_tokens: tuple[int | None, int | None] = (None, None),
    ) -> float | None:
        """Estimate USD cost for an LLM call with cache-aware pricing.

        Args:
            provider: Provider name
            model: Model name
            input_tokens: Input token count
            output_tokens: Output token count
            cache_tokens: (cache_read_tokens, cache_creation_tokens)

        Returns:
            Estimated cost in USD, or None if pricing unknown
        """
        if input_tokens is None and output_tokens is None:
            return None

        key = f"{provider}/{model}"
        pricing = _PRICING.get(key)
        if not pricing:
            return None

        input_price, output_price = pricing
        total_input = input_tokens or 0
        cache_read = cache_tokens[0] or 0
        cache_write = cache_tokens[1] or 0

        if provider == "anthropic":
            # Anthropic reports input_tokens (non-cached), cache_creation_input_tokens,
            # and cache_read_input_tokens separately — they are additive, not subsets.
            # Cost: regular 1.00x, cache reads 0.10x, cache writes 1.25x base input price.
            input_cost = (total_input / 1_000_000) * input_price
            input_cost += (cache_read / 1_000_000) * input_price * 0.10
            input_cost += (cache_write / 1_000_000) * input_price * 1.25
        else:
            # OpenAI/OpenRouter: prompt_tokens includes cached_tokens (subset accounting).
            # Regular = prompt_tokens - cached_tokens; cached reads billed at 0.50x.
            regular_input = max(0, total_input - cache_read)
            input_cost = (regular_input / 1_000_000) * input_price
            input_cost += (cache_read / 1_000_000) * input_price * 0.50
        output_cost = ((output_tokens or 0) / 1_000_000) * output_price
        return input_cost + output_cost


def persist_jsonl(
    metrics: WorkflowExecutionMetrics,
    path: str = "logs/execution_metrics.jsonl",
) -> None:
    """Append metrics to JSONL file.

    Args:
        metrics: Workflow execution metrics to persist
        path: Output file path
    """
    filepath = Path(path)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with filepath.open("a") as f:
        f.write(metrics.model_dump_json() + "\n")

    logger.debug(f"Persisted execution metrics to {path}")


@contextmanager
def timed_operation(name: str, **metadata: str | int | float) -> Iterator[None]:
    """Context manager that records sub-operation timing to current collector.

    When no collector is active (metrics disabled), this is a no-op.

    Args:
        name: Operation name
        **metadata: Optional key-value metadata
    """
    collector = current_collector.get()
    if collector is None:
        yield
        return

    start = time.perf_counter()
    yield
    elapsed_ms = (time.perf_counter() - start) * 1000
    collector.record_sub_operation(name, elapsed_ms, metadata or None)
