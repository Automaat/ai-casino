"""Context managers for execution tracking."""

from contextlib import asynccontextmanager, contextmanager
from typing import Any, AsyncGenerator, Generator
from uuid import UUID

from loguru import logger

from src.execution_tracking.models import ExecutionNodeType
from src.execution_tracking.tracker import (
    ExecutionGraphTracker,
    get_current_tracker,
    set_current_tracker,
)


@contextmanager
def track_execution(
    node_type: ExecutionNodeType,
    name: str,
    parent_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Generator[str | None, None, None]:
    """Context manager for tracking execution node (sync version).

    Args:
        node_type: Type of node (TOOL/AGENT/WORKFLOW_STAGE)
        name: Human-readable name
        parent_id: Parent node ID (None to use current)
        metadata: Additional context

    Yields:
        Node ID if tracker available, None otherwise

    Example:
        ```python
        with track_execution(ExecutionNodeType.TOOL, "FetchMarketDataTool") as node_id:
            result = fetch_market_data()
        ```
    """
    tracker = get_current_tracker()
    if not tracker:
        yield None
        return

    node_id = tracker.start_node(node_type, name, parent_id, metadata)
    exception_occurred = False
    try:
        yield node_id
    except Exception as e:
        exception_occurred = True
        tracker.fail_node(node_id, str(e))
        raise
    finally:
        if not exception_occurred:
            tracker.complete_node(node_id)


@asynccontextmanager
async def atrack_execution(
    node_type: ExecutionNodeType,
    name: str,
    parent_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> AsyncGenerator[str | None, None]:
    """Context manager for tracking execution node (async version).

    Args:
        node_type: Type of node (TOOL/AGENT/WORKFLOW_STAGE)
        name: Human-readable name
        parent_id: Parent node ID (None to use current)
        metadata: Additional context

    Yields:
        Node ID if tracker available, None otherwise

    Example:
        ```python
        async with atrack_execution(ExecutionNodeType.AGENT, "TechnicalAnalyst") as node_id:
            result = await analyst.analyze(symbol, data)
        ```
    """
    tracker = get_current_tracker()
    if not tracker:
        yield None
        return

    node_id = tracker.start_node(node_type, name, parent_id, metadata)
    exception_occurred = False
    try:
        yield node_id
    except Exception as e:
        exception_occurred = True
        tracker.fail_node(node_id, str(e))
        raise
    finally:
        if not exception_occurred:
            tracker.complete_node(node_id)


@contextmanager
def track_workflow(
    workflow_id: str | UUID,
    symbol: str | None = None,
    event_bus: Any = None,
) -> Generator[ExecutionGraphTracker, None, None]:
    """Context manager for tracking entire workflow execution.

    Sets up ExecutionGraphTracker in ContextVar for the workflow scope.

    Args:
        workflow_id: Unique workflow/analysis run ID
        symbol: Stock symbol being analyzed
        event_bus: EventBus for real-time event publishing

    Yields:
        ExecutionGraphTracker instance

    Example:
        ```python
        with track_workflow(workflow_id, symbol="AAPL", event_bus=bus) as tracker:
            # All nested track_execution calls will use this tracker
            result = run_analysis(symbol)
        ```
    """
    # Create new tracker
    tracker = ExecutionGraphTracker(workflow_id, symbol, event_bus)

    # Save previous tracker (for nested workflow support)
    previous_tracker = get_current_tracker()

    try:
        # Set as current tracker
        set_current_tracker(tracker)
        logger.debug(f"Started workflow tracking: {workflow_id} (symbol={symbol})")

        yield tracker

    finally:
        # Restore previous tracker
        set_current_tracker(previous_tracker)
        logger.debug(
            f"Completed workflow tracking: {workflow_id} "
            f"({len(tracker.graph.nodes)} nodes, "
            f"{len(tracker.graph.get_running_nodes())} still running)"
        )
