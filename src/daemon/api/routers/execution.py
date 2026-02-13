"""Execution metrics and graph endpoints."""

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from loguru import logger

from src.daemon.api.models import (
    ActiveExecutionGraphsResponse,
    ExecutionGraphDetailResponse,
    ExecutionGraphHistoryResponse,
    ExecutionMetricsListResponse,
)
from src.daemon.api.routers.shared import get_components

router = APIRouter(tags=["execution"])


@router.get("/api/execution-metrics", response_model=ExecutionMetricsListResponse)
async def get_execution_metrics(limit: int = 50) -> ExecutionMetricsListResponse:
    """Get recent execution metrics from JSONL.

    Args:
        limit: Max number of metrics to return (clamped to 1-500)

    Returns:
        ExecutionMetricsListResponse with list of metrics
    """
    limit = max(1, min(limit, 500))

    def _read_metrics() -> list[dict]:
        metrics_file = Path("logs/execution_metrics.jsonl").expanduser()

        if not metrics_file.exists():
            return []

        metrics = []
        # Read last N lines efficiently (read backwards)
        with metrics_file.open("rb") as f:
            f.seek(0, 2)
            file_size = f.tell()
            if file_size == 0:
                return []

            # Read file in chunks from end
            buffer_size = 8192
            lines = []
            buffer = b""
            pos = file_size

            while pos > 0 and len(lines) < limit:
                chunk_size = min(buffer_size, pos)
                pos -= chunk_size
                f.seek(pos)
                chunk = f.read(chunk_size)
                buffer = chunk + buffer

                # Extract complete lines
                while b"\n" in buffer and len(lines) < limit:
                    buffer, line = buffer.rsplit(b"\n", 1)
                    if line:
                        lines.insert(0, line)

            # Parse JSONL
            for line in lines[-limit:]:
                try:
                    metric = json.loads(line)
                    metrics.append(metric)
                except json.JSONDecodeError as e:
                    logger.opt(exception=True).warning(f"Malformed JSONL line: {e}")
                    continue

            # Reverse to get newest first
            metrics.reverse()

        return metrics

    try:
        metrics = await asyncio.to_thread(_read_metrics)
    except Exception as e:
        logger.opt(exception=True).error(f"Failed to read execution metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to read execution metrics") from e

    return ExecutionMetricsListResponse(metrics=metrics, count=len(metrics))


@router.get("/api/execution-metrics/{workflow_id}", response_model=dict)
async def get_execution_metric_detail(workflow_id: str) -> dict:
    """Get single workflow execution detail.

    Args:
        workflow_id: Workflow ID to fetch

    Returns:
        WorkflowExecutionMetrics as dict
    """

    def _find_metric() -> tuple[dict | None, bool]:
        """Find metric and return (metric, file_exists)."""
        metrics_file = Path("logs/execution_metrics.jsonl").expanduser()

        if not metrics_file.exists():
            return None, False

        with metrics_file.open() as f:
            for line in f:
                try:
                    metric = json.loads(line)
                    if metric.get("workflow_id") == workflow_id:
                        return metric, True
                except json.JSONDecodeError:
                    continue

        return None, True

    try:
        result, file_exists = await asyncio.to_thread(_find_metric)
        if result is None:
            if not file_exists:
                detail = "Execution metrics file not found"
            else:
                detail = f"Workflow {workflow_id} not found"
            raise HTTPException(status_code=404, detail=detail)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.opt(exception=True).error(f"Failed to fetch workflow detail: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch workflow detail") from e


@router.get("/api/execution/active", response_model=ActiveExecutionGraphsResponse)
async def get_active_execution_graphs(request: Request) -> ActiveExecutionGraphsResponse:
    """Get active execution graphs from in-memory trackers."""
    components = get_components(request)

    active_trackers = await components.state.get_active_execution_trackers()
    graphs = [tracker.graph.model_dump(mode="json") for tracker in active_trackers.values()]

    return ActiveExecutionGraphsResponse(graphs=graphs, count=len(graphs))


@router.get("/api/execution/{workflow_id}", response_model=ExecutionGraphDetailResponse)
async def get_execution_graph(request: Request, workflow_id: str) -> ExecutionGraphDetailResponse:
    """Get execution graph by workflow ID.

    Search order: active trackers → in-memory history → database

    Args:
        request: FastAPI request
        workflow_id: Workflow ID to fetch

    Returns:
        ExecutionGraphDetailResponse with graph data and source
    """
    from src.database.connection import get_session
    from src.database.repositories.execution_graph import ExecutionGraphRepository

    components = get_components(request)

    # Check active trackers
    active_trackers = await components.state.get_active_execution_trackers()
    if workflow_id in active_trackers:
        tracker = active_trackers[workflow_id]
        return ExecutionGraphDetailResponse(
            workflow_id=workflow_id,
            graph=tracker.graph.model_dump(mode="json"),
            source="active",
        )

    # Check in-memory history
    execution_history = await components.state.get_execution_graph_history(limit=1000)
    for graph in execution_history:
        if str(graph.workflow_id) == workflow_id:
            return ExecutionGraphDetailResponse(
                workflow_id=workflow_id,
                graph=graph.model_dump(mode="json"),
                source="memory",
            )

    # Check database
    try:
        async with get_session() as session:
            repo = ExecutionGraphRepository(session)
            graph = await repo.get_by_workflow_id(workflow_id)

            if graph:
                return ExecutionGraphDetailResponse(
                    workflow_id=workflow_id,
                    graph=graph.model_dump(mode="json"),
                    source="database",
                )
    except Exception as e:
        logger.opt(exception=True).error(f"Failed to fetch from DB: {e}")

    raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")


@router.get("/api/execution/history", response_model=ExecutionGraphHistoryResponse)
async def get_execution_graph_history(
    limit: int = 50,
    symbol: str | None = None,
    days: int | None = None,
) -> ExecutionGraphHistoryResponse:
    """Get paginated execution graph history.

    Args:
        limit: Max results (1-500)
        symbol: Filter by symbol
        days: Filter by last N days

    Returns:
        ExecutionGraphHistoryResponse with graphs and metadata
    """
    from src.database.connection import get_session
    from src.database.engine import MissingDatabaseURLError
    from src.database.repositories.execution_graph import ExecutionGraphRepository

    limit = max(1, min(limit, 500))

    try:
        async with get_session() as session:
            repo = ExecutionGraphRepository(session)

            if days:
                start = datetime.now(UTC) - timedelta(days=days)
                end = datetime.now(UTC)
                graphs = await repo.get_by_date_range(start, end, symbol, limit)
            else:
                graphs = await repo.list_recent(limit, symbol)

            return ExecutionGraphHistoryResponse(
                graphs=[g.model_dump(mode="json") for g in graphs],
                count=len(graphs),
                database_enabled=True,
            )
    except MissingDatabaseURLError:
        logger.debug("Database not configured, returning empty history")
        return ExecutionGraphHistoryResponse(
            graphs=[],
            count=0,
            database_enabled=False,
        )
    except Exception as e:
        logger.opt(exception=True).error(f"Failed to fetch history: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch execution graph history") from e
