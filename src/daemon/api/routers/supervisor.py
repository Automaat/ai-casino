"""Supervisor metrics endpoints."""

from datetime import UTC, datetime, timedelta
from typing import TypedDict
from uuid import UUID

from fastapi import APIRouter, HTTPException
from loguru import logger

from src.agents.supervisor.metrics import SupervisorCycleMetrics
from src.daemon.api.dependencies import get_supervisor_metrics_repo
from src.daemon.api.models import (
    ErrorSummaryResponse,
    SupervisorMetricResponse,
    SupervisorMetricsListResponse,
    SupervisorSummaryResponse,
    WorkerPerformanceResponse,
    WorkerStats,
)

router = APIRouter(prefix="/supervisor", tags=["supervisor"])


class WorkerStatsAgg(TypedDict):
    """Worker statistics aggregation for internal use."""

    total: int
    successful: int
    failed: int
    durations: list[float]


def to_supervisor_metric_response(metric: SupervisorCycleMetrics) -> SupervisorMetricResponse:
    """Convert SupervisorCycleMetrics to response model.

    Args:
        metric: SupervisorCycleMetrics instance

    Returns:
        SupervisorMetricResponse
    """
    return SupervisorMetricResponse(
        id=str(metric.id),
        created_at=metric.created_at or datetime.now(UTC),
        workflow_id=metric.workflow_id,
        symbol=metric.symbol,
        timestamp=metric.timestamp,
        required_analyses=metric.required_analyses,
        optional_analyses=metric.optional_analyses,
        skip_analyses=metric.skip_analyses,
        routing_reasoning=metric.routing_reasoning,
        total_workers=metric.total_workers,
        required_workers=metric.required_workers,
        optional_workers=metric.optional_workers,
        successful_workers=metric.successful_workers,
        failed_workers=metric.failed_workers,
        routing_decision_ms=metric.routing_decision_ms,
        group1_execution_ms=metric.group1_execution_ms,
        research_execution_ms=metric.research_execution_ms,
        total_supervisor_overhead_ms=metric.total_supervisor_overhead_ms,
        worker_timings=metric.worker_timings,
        worker_errors=metric.worker_errors,
        total_llm_calls=metric.total_llm_calls,
        total_cost_usd=metric.total_cost_usd,
        planning_fallback_used=metric.planning_fallback_used,
        synthesis_fallback_used=metric.synthesis_fallback_used,
        confidence_adjustment=metric.confidence_adjustment,
        synthesis_reasoning=metric.synthesis_reasoning,
        timeout_triggered=metric.timeout_triggered,
    )


@router.get("/metrics/recent", response_model=SupervisorMetricsListResponse)
async def get_recent_metrics(limit: int = 50, symbol: str | None = None) -> SupervisorMetricsListResponse:
    """Get recent supervisor metrics.

    Args:
        limit: Maximum records to return (1-500)
        symbol: Optional symbol filter

    Returns:
        List of recent supervisor metrics
    """
    limit = max(1, min(limit, 500))

    try:
        async with get_supervisor_metrics_repo() as repo:
            metrics = await repo.get_recent(limit=limit, symbol=symbol)
            return SupervisorMetricsListResponse(
                metrics=[to_supervisor_metric_response(m) for m in metrics],
                count=len(metrics),
            )
    except Exception as e:
        logger.opt(exception=True).error(f"Failed to fetch recent supervisor metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch metrics") from e


@router.get("/metrics/summary", response_model=SupervisorSummaryResponse)
async def get_summary(hours: int = 24, symbol: str | None = None) -> SupervisorSummaryResponse:
    """Get supervisor metrics summary for time period.

    Args:
        hours: Hours to look back (1-720)
        symbol: Optional symbol filter

    Returns:
        Aggregated supervisor metrics summary
    """
    hours = max(1, min(hours, 720))

    try:
        async with get_supervisor_metrics_repo() as repo:
            # Use DB aggregation to avoid truncation and Python processing
            stats = await repo.get_efficiency_stats(symbol=symbol, days=hours // 24 or 1)

            return SupervisorSummaryResponse(
                avg_routing_ms=stats["avg_routing_ms"],
                avg_group1_ms=stats["avg_group1_ms"],
                avg_research_ms=stats["avg_research_ms"],
                avg_total_ms=stats["avg_total_ms"],
                timeout_rate_percent=stats["timeout_rate_percent"],
                sample_size=stats["sample_size"],
                symbol=symbol,
            )
    except Exception as e:
        logger.opt(exception=True).error(f"Failed to fetch supervisor summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch summary") from e


@router.get("/metrics/workers", response_model=WorkerPerformanceResponse)
async def get_worker_performance(hours: int = 24) -> WorkerPerformanceResponse:
    """Get worker performance statistics.

    Args:
        hours: Hours to look back (1-720)

    Returns:
        Worker performance by type
    """
    hours = max(1, min(hours, 720))

    try:
        async with get_supervisor_metrics_repo() as repo:
            # Get date range
            end_time = datetime.now(UTC)
            start_time = end_time - timedelta(hours=hours)

            # Get metrics in range
            metrics_list = await repo.get_date_range(start=start_time, end=end_time, limit=10000)

            # Aggregate worker stats
            worker_stats_dict: dict[str, WorkerStatsAgg] = {}

            for metric in metrics_list:
                # Process worker timings
                for worker_name, duration_ms in metric.worker_timings.items():
                    if worker_name not in worker_stats_dict:
                        worker_stats_dict[worker_name] = {
                            "total": 0,
                            "successful": 0,
                            "failed": 0,
                            "durations": [],
                        }

                    worker_stats_dict[worker_name]["total"] += 1
                    worker_stats_dict[worker_name]["durations"].append(duration_ms)

                    # Check if worker failed
                    if worker_name in metric.worker_errors:
                        worker_stats_dict[worker_name]["failed"] += 1
                    else:
                        worker_stats_dict[worker_name]["successful"] += 1

            # Calculate stats
            worker_stats: dict[str, WorkerStats] = {}
            for worker_name, stats_data in worker_stats_dict.items():
                total = stats_data["total"]
                successful = stats_data["successful"]
                failed = stats_data["failed"]
                durations = stats_data["durations"]

                worker_stats[worker_name] = WorkerStats(
                    total_executions=total,
                    successful_executions=successful,
                    failed_executions=failed,
                    success_rate=(successful / total * 100) if total > 0 else 0.0,
                    avg_duration_ms=sum(durations) / len(durations) if durations else 0.0,
                )

            return WorkerPerformanceResponse(
                worker_stats=worker_stats,
                total_workers=len(worker_stats),
                sample_size=len(metrics_list),
            )
    except Exception as e:
        logger.opt(exception=True).error(f"Failed to fetch worker performance: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch worker performance") from e


@router.get("/metrics/errors", response_model=ErrorSummaryResponse)
async def get_error_summary(hours: int = 24) -> ErrorSummaryResponse:
    """Get error summary by worker type.

    Args:
        hours: Hours to look back (1-720)

    Returns:
        Error counts by worker type
    """
    hours = max(1, min(hours, 720))

    try:
        async with get_supervisor_metrics_repo() as repo:
            error_counts = await repo.get_error_summary(hours=hours)

            return ErrorSummaryResponse(
                error_counts=error_counts,
                total_errors=sum(error_counts.values()),
            )
    except Exception as e:
        logger.opt(exception=True).error(f"Failed to fetch error summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch error summary") from e


@router.get("/metrics/{metric_id}", response_model=SupervisorMetricResponse)
async def get_metric_by_id(metric_id: UUID) -> SupervisorMetricResponse:
    """Get supervisor metric by ID.

    Args:
        metric_id: Supervisor metric UUID

    Returns:
        Supervisor metric detail
    """
    try:
        async with get_supervisor_metrics_repo() as repo:
            metric = await repo.get_by_id(str(metric_id))

            if not metric:
                raise HTTPException(status_code=404, detail="Metric not found")

            return to_supervisor_metric_response(metric)
    except HTTPException:
        raise
    except Exception as e:
        logger.opt(exception=True).error(f"Failed to fetch supervisor metric {metric_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch metric") from e
