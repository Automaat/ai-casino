"""Cost analytics service for LLM usage aggregation."""

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from loguru import logger

from src.metrics.execution import WorkflowExecutionMetrics


@dataclass
class CostAnalyticsSummary:
    """Aggregated cost metrics summary."""

    total_cost_usd: float
    total_tokens: int
    total_executions: int
    avg_cost_per_execution: float
    avg_cost_per_signal: float
    forecast_30d_usd: float
    date_range: tuple[datetime, datetime]


@dataclass
class CostTrendPoint:
    """Cost metrics for a time bucket."""

    timestamp: datetime
    cost_usd: float
    tokens: int
    execution_count: int


@dataclass
class CostByDimension:
    """Cost breakdown by dimension (symbol/agent/model)."""

    dimension_value: str
    cost_usd: float
    tokens: int
    execution_count: int
    percentage: float


class CostAnalyticsService:
    """Service for aggregating and analyzing LLM cost data from execution metrics."""

    def __init__(self, metrics_path: str = "logs/execution_metrics.jsonl") -> None:
        """Initialize cost analytics service.

        Args:
            metrics_path: Path to JSONL metrics file
        """
        self._metrics_path = Path(metrics_path).expanduser()
        self._cache: dict[str, tuple[datetime, object]] = {}
        self._cache_ttl = 300  # 5 minutes

    def __repr__(self) -> str:
        """Return string representation."""
        return f"CostAnalyticsService(metrics_path={self._metrics_path})"

    async def get_summary(self, start: datetime, end: datetime) -> CostAnalyticsSummary:
        """Get cost summary for date range.

        Args:
            start: Start timestamp (inclusive)
            end: End timestamp (inclusive)

        Returns:
            Aggregated cost summary
        """
        cache_key = f"summary:{start.isoformat()}:{end.isoformat()}"
        cached = self._get_cached(cache_key)
        if cached:
            return cast("CostAnalyticsSummary", cached)

        metrics = await self._read_metrics(start, end)

        total_cost = sum(m.total_estimated_cost_usd for m in metrics)
        total_tokens = sum(m.total_input_tokens + m.total_output_tokens for m in metrics)
        total_executions = len(metrics)
        avg_cost_per_execution = total_cost / total_executions if total_executions > 0 else 0.0

        # Calculate trade signals (BUY/SELL only, exclude HOLD)
        trade_signals = self._count_trade_signals(metrics)
        avg_cost_per_signal = total_cost / trade_signals if trade_signals > 0 else 0.0

        # Forecast 30-day cost
        daily_costs = await self._get_daily_costs(metrics)
        forecast_30d = self._calculate_forecast(daily_costs)

        summary = CostAnalyticsSummary(
            total_cost_usd=total_cost,
            total_tokens=total_tokens,
            total_executions=total_executions,
            avg_cost_per_execution=avg_cost_per_execution,
            avg_cost_per_signal=avg_cost_per_signal,
            forecast_30d_usd=forecast_30d,
            date_range=(start, end),
        )

        self._cache[cache_key] = (datetime.now(UTC), summary)
        return summary

    async def get_trends(
        self,
        period: str,
        start: datetime,
        end: datetime,
    ) -> list[CostTrendPoint]:
        """Get cost trends bucketed by time period.

        Args:
            period: Time bucket ('daily' or 'weekly')
            start: Start timestamp
            end: End timestamp

        Returns:
            List of cost trend points
        """
        cache_key = f"trends:{period}:{start.isoformat()}:{end.isoformat()}"
        cached = self._get_cached(cache_key)
        if cached:
            return cast("list[CostTrendPoint]", cached)

        metrics = await self._read_metrics(start, end)

        # Bucket by period
        buckets: dict[str, list[WorkflowExecutionMetrics]] = {}
        for metric in metrics:
            if period == "daily":
                bucket_key = metric.timestamp.date().isoformat()
            else:  # weekly
                # Use ISO week number
                bucket_key = f"{metric.timestamp.year}-W{metric.timestamp.isocalendar()[1]:02d}"

            if bucket_key not in buckets:
                buckets[bucket_key] = []
            buckets[bucket_key].append(metric)

        # Aggregate buckets
        trends = []
        for bucket_key in sorted(buckets.keys()):
            bucket_metrics = buckets[bucket_key]
            cost = sum(m.total_estimated_cost_usd for m in bucket_metrics)
            tokens = sum(m.total_input_tokens + m.total_output_tokens for m in bucket_metrics)
            count = len(bucket_metrics)

            # Parse timestamp from bucket key
            if period == "daily":
                timestamp = datetime.fromisoformat(bucket_key).replace(tzinfo=UTC)
            else:
                # Parse ISO week (e.g., "2026-W07")
                year, week_str = bucket_key.split("-W")
                timestamp = datetime.strptime(f"{year}-W{week_str}-1", "%G-W%V-%u").replace(tzinfo=UTC)

            trends.append(
                CostTrendPoint(
                    timestamp=timestamp,
                    cost_usd=cost,
                    tokens=tokens,
                    execution_count=count,
                )
            )

        self._cache[cache_key] = (datetime.now(UTC), trends)
        return trends

    async def get_by_symbol(self, start: datetime, end: datetime) -> list[CostByDimension]:
        """Get cost breakdown by symbol.

        Args:
            start: Start timestamp
            end: End timestamp

        Returns:
            List of cost by symbol
        """
        cache_key = f"by_symbol:{start.isoformat()}:{end.isoformat()}"
        cached = self._get_cached(cache_key)
        if cached:
            return cast("list[CostByDimension]", cached)

        metrics = await self._read_metrics(start, end)
        result = self._aggregate_by_dimension(metrics, lambda m: m.symbol)

        self._cache[cache_key] = (datetime.now(UTC), result)
        return result

    async def get_by_agent(self, start: datetime, end: datetime) -> list[CostByDimension]:
        """Get cost breakdown by agent.

        Args:
            start: Start timestamp
            end: End timestamp

        Returns:
            List of cost by agent
        """
        cache_key = f"by_agent:{start.isoformat()}:{end.isoformat()}"
        cached = self._get_cached(cache_key)
        if cached:
            return cast("list[CostByDimension]", cached)

        metrics = await self._read_metrics(start, end)

        # Aggregate across all LLM calls by agent
        agent_data: dict[str, dict[str, float | int]] = {}
        total_cost = 0.0

        for metric in metrics:
            for call in metric.llm_calls:
                agent = call.agent_name
                cost = call.estimated_cost_usd or 0.0
                tokens = (call.input_tokens or 0) + (call.output_tokens or 0)

                if agent not in agent_data:
                    agent_data[agent] = {"cost": 0.0, "tokens": 0, "count": 0}

                agent_data[agent]["cost"] += cost
                agent_data[agent]["tokens"] += tokens
                agent_data[agent]["count"] += 1
                total_cost += cost

        epsilon = 1e-10
        result = [
            CostByDimension(
                dimension_value=agent,
                cost_usd=float(data["cost"]),
                tokens=int(data["tokens"]),
                execution_count=int(data["count"]),
                percentage=(float(data["cost"]) / total_cost * 100) if total_cost > epsilon else 0.0,
            )
            for agent, data in agent_data.items()
        ]

        # Sort by cost descending
        result.sort(key=lambda x: x.cost_usd, reverse=True)

        self._cache[cache_key] = (datetime.now(UTC), result)
        return result

    async def get_by_model(self, start: datetime, end: datetime) -> list[CostByDimension]:
        """Get cost breakdown by model.

        Args:
            start: Start timestamp
            end: End timestamp

        Returns:
            List of cost by model
        """
        cache_key = f"by_model:{start.isoformat()}:{end.isoformat()}"
        cached = self._get_cached(cache_key)
        if cached:
            return cast("list[CostByDimension]", cached)

        metrics = await self._read_metrics(start, end)

        # Aggregate by provider/model
        model_data: dict[str, dict[str, float | int]] = {}
        total_cost = 0.0

        for metric in metrics:
            model_key = f"{metric.provider}/{metric.model}"
            cost = metric.total_estimated_cost_usd
            tokens = metric.total_input_tokens + metric.total_output_tokens

            if model_key not in model_data:
                model_data[model_key] = {"cost": 0.0, "tokens": 0, "count": 0}

            model_data[model_key]["cost"] += cost
            model_data[model_key]["tokens"] += tokens
            model_data[model_key]["count"] += 1
            total_cost += cost

        result = [
            CostByDimension(
                dimension_value=model,
                cost_usd=float(data["cost"]),
                tokens=int(data["tokens"]),
                execution_count=int(data["count"]),
                percentage=(float(data["cost"]) / total_cost * 100) if total_cost > 0 else 0.0,
            )
            for model, data in model_data.items()
        ]

        # Sort by cost descending
        result.sort(key=lambda x: x.cost_usd, reverse=True)

        self._cache[cache_key] = (datetime.now(UTC), result)
        return result

    async def _read_metrics(self, start: datetime, end: datetime) -> list[WorkflowExecutionMetrics]:
        """Read and parse metrics from JSONL file.

        Args:
            start: Start timestamp
            end: End timestamp

        Returns:
            List of parsed metrics
        """

        def _read() -> list[dict]:
            if not self._metrics_path.exists():
                logger.debug(
                    f"Metrics file does not exist at '{self._metrics_path}'. Returning empty metrics list."
                )
                return []

            metrics = []
            with self._metrics_path.open() as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        timestamp = datetime.fromisoformat(data["timestamp"])

                        # Filter by date range
                        if start <= timestamp <= end:
                            metrics.append(data)
                    except (json.JSONDecodeError, KeyError, ValueError) as e:
                        logger.opt(exception=True).warning(f"Malformed JSONL line: {e}")
                        continue

            return metrics

        data = await asyncio.to_thread(_read)
        return [WorkflowExecutionMetrics.model_validate(d) for d in data]

    def _aggregate_by_dimension(
        self,
        metrics: list[WorkflowExecutionMetrics],
        key_fn: Callable[[WorkflowExecutionMetrics], str],
    ) -> list[CostByDimension]:
        """Aggregate metrics by dimension.

        Args:
            metrics: List of workflow metrics
            key_fn: Function to extract dimension key from metric

        Returns:
            List of aggregated cost by dimension
        """
        dimension_data: dict[str, dict[str, float | int]] = {}
        total_cost = 0.0

        for metric in metrics:
            key = key_fn(metric)
            cost = metric.total_estimated_cost_usd
            tokens = metric.total_input_tokens + metric.total_output_tokens

            if key not in dimension_data:
                dimension_data[key] = {"cost": 0.0, "tokens": 0, "count": 0}

            dimension_data[key]["cost"] += cost
            dimension_data[key]["tokens"] += tokens
            dimension_data[key]["count"] += 1
            total_cost += cost

        result = [
            CostByDimension(
                dimension_value=key,
                cost_usd=float(data["cost"]),
                tokens=int(data["tokens"]),
                execution_count=int(data["count"]),
                percentage=(float(data["cost"]) / total_cost * 100) if total_cost > 0 else 0.0,
            )
            for key, data in dimension_data.items()
        ]

        # Sort by cost descending
        result.sort(key=lambda x: x.cost_usd, reverse=True)
        return result

    def _count_trade_signals(self, metrics: list[WorkflowExecutionMetrics]) -> int:
        """Count executions used as a proxy for trade signals.

        This helper currently does not distinguish between BUY, SELL, or HOLD
        outcomes. It simply counts workflow executions, which may include
        non-trading/HOLD decisions.

        Args:
            metrics: List of workflow metrics

        Returns:
            Count of executions (used as proxy for trade signals)
        """
        # NOTE: Proper trade signal counting would require parsing workflow results
        # to extract actual BUY/SELL/HOLD decisions. Until that is implemented,
        # we intentionally use execution count as a coarse proxy.
        return len(metrics)

    async def _get_daily_costs(
        self,
        metrics: list[WorkflowExecutionMetrics],
    ) -> list[float]:
        """Get daily costs for forecast calculation.

        Args:
            metrics: Workflow metrics

        Returns:
            List of daily costs
        """
        daily_buckets: dict[str, float] = {}

        for metric in metrics:
            date_key = metric.timestamp.date().isoformat()
            if date_key not in daily_buckets:
                daily_buckets[date_key] = 0.0
            daily_buckets[date_key] += metric.total_estimated_cost_usd

        return [daily_buckets[key] for key in sorted(daily_buckets.keys())]

    def _calculate_forecast(self, daily_costs: list[float]) -> float:
        """Calculate 30-day cost forecast using linear regression.

        Args:
            daily_costs: List of daily costs

        Returns:
            Forecasted 30-day cost
        """
        min_data_points = 2
        if len(daily_costs) < min_data_points:
            # Insufficient data for trend, use average * 30
            avg = sum(daily_costs) / len(daily_costs) if daily_costs else 0.0
            return avg * 30

        # Simple linear regression: y = mx + b
        n = len(daily_costs)
        x = list(range(n))
        y = daily_costs

        x_mean = sum(x) / n
        y_mean = sum(y) / n

        # Calculate slope (m)
        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        slope = numerator / denominator if denominator != 0 else 0

        # Calculate intercept (b)
        intercept = y_mean - slope * x_mean

        # Project 30 days from last data point
        forecast_sum = 0.0
        for day in range(n, n + 30):
            forecast_sum += slope * day + intercept

        return max(0.0, forecast_sum)

    def _get_cached(self, key: str) -> object | None:
        """Get cached value if not expired.

        Args:
            key: Cache key

        Returns:
            Cached value or None
        """
        if key not in self._cache:
            return None

        cached_time, value = self._cache[key]
        if (datetime.now(UTC) - cached_time).total_seconds() > self._cache_ttl:
            del self._cache[key]
            return None

        return value
