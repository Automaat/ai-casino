"""Execution graph tracker using ContextVar for thread-local state."""

from contextvars import ContextVar
from typing import TYPE_CHECKING
from uuid import UUID

from loguru import logger

from src.execution_tracking.models import (
    ExecutionGraph,
    ExecutionNode,
    ExecutionNodeType,
    ExecutionStatus,
)

if TYPE_CHECKING:
    from src.daemon.event_bus import EventBus

# ContextVars for thread-local execution state
_current_node: ContextVar[str | None] = ContextVar("_current_node", default=None)
_execution_stack: ContextVar[list[str] | None] = ContextVar("_execution_stack", default=None)
_workflow_tracker: ContextVar[ExecutionGraphTracker | None] = ContextVar("_workflow_tracker", default=None)


class ExecutionGraphTracker:
    """Tracks execution graph for a workflow with EventBus integration."""

    def __init__(
        self, workflow_id: str | UUID, symbol: str | None = None, event_bus: EventBus | None = None
    ) -> None:
        """Initialize tracker.

        Args:
            workflow_id: Unique workflow/analysis run ID
            symbol: Stock symbol being analyzed
            event_bus: EventBus for real-time event publishing
        """
        self.graph = ExecutionGraph(workflow_id=workflow_id, symbol=symbol)
        self._event_bus = event_bus
        logger.debug(f"Initialized ExecutionGraphTracker for workflow {workflow_id}")

    def start_node(
        self,
        node_type: ExecutionNodeType,
        name: str,
        parent_id: str | None = None,
        metadata: dict | None = None,
    ) -> str:
        """Start new execution node.

        Args:
            node_type: Type of node (TOOL/AGENT/WORKFLOW_STAGE)
            name: Human-readable name
            parent_id: Parent node ID (None for root)
            metadata: Additional context (args, etc.)

        Returns:
            Created node ID
        """
        # Use current node as parent if not specified
        if parent_id is None:
            parent_id = _current_node.get()

        node = ExecutionNode(
            node_type=node_type,
            name=name,
            parent_id=parent_id,
            status=ExecutionStatus.RUNNING,
            metadata=metadata or {},
        )

        self.graph.add_node(node)

        # Update ContextVar state
        stack = _execution_stack.get()
        if stack is None:
            stack = []
        stack = stack.copy()  # Create new list to avoid mutation
        stack.append(node.node_id)
        _execution_stack.set(stack)
        _current_node.set(node.node_id)

        # Publish event
        self._publish_event("EXECUTION_NODE_START", node)

        logger.debug(
            f"Started {node_type} node: {name} (id={node.node_id}, parent={parent_id}, "
            f"stack_depth={len(stack)})"
        )

        return node.node_id

    def complete_node(self, node_id: str, metadata: dict | None = None) -> None:
        """Mark node as completed.

        Args:
            node_id: Node ID to complete
            metadata: Additional result metadata
        """
        if node := self.graph.get_node(node_id):
            node.complete()
            if metadata:
                node.metadata.update(metadata)

            # Update ContextVar state
            self._pop_from_stack(node_id)

            # Publish event
            self._publish_event("EXECUTION_NODE_COMPLETE", node)

            logger.debug(f"Completed node: {node.name} (id={node_id}, duration={node.duration_ms:.1f}ms)")
        else:
            logger.warning(f"Attempted to complete unknown node {node_id}")

    def fail_node(self, node_id: str, error: str) -> None:
        """Mark node as failed.

        Args:
            node_id: Node ID to fail
            error: Error message
        """
        if node := self.graph.get_node(node_id):
            node.fail(error)

            # Update ContextVar state
            self._pop_from_stack(node_id)

            # Publish event
            self._publish_event("EXECUTION_NODE_COMPLETE", node)

            logger.warning(f"Failed node: {node.name} (id={node_id}, error={error})")
        else:
            logger.warning(f"Attempted to fail unknown node {node_id}")

    def _pop_from_stack(self, node_id: str) -> None:
        """Remove node from execution stack and update current node."""
        stack = _execution_stack.get()
        if stack is None:
            stack = []
        if stack and stack[-1] == node_id:
            stack = stack.copy()  # Create new list to avoid mutation
            stack.pop()
            _execution_stack.set(stack)
            # Set current node to new top of stack (or None if empty)
            _current_node.set(stack[-1] if stack else None)

    def _publish_event(self, event_type: str, node: ExecutionNode) -> None:
        """Publish execution event to EventBus.

        Args:
            event_type: Event type (EXECUTION_NODE_START/COMPLETE)
            node: Execution node
        """
        if not self._event_bus:
            return

        try:
            # Import here to avoid circular dependency
            import asyncio

            from src.daemon.event_bus import DashboardEvent, EventType

            event = DashboardEvent(
                event_type=EventType(event_type),
                data={
                    "workflow_id": str(self.graph.workflow_id),
                    "symbol": self.graph.symbol,
                    "node_id": node.node_id,
                    "node_type": node.node_type,
                    "name": node.name,
                    "parent_id": node.parent_id,
                    "status": node.status,
                    "start_time": node.start_time.isoformat(),
                    "end_time": node.end_time.isoformat() if node.end_time else None,
                    "duration_ms": node.duration_ms,
                    "error": node.error,
                },
            )

            # Fire and forget - don't block on event publishing
            # Only publish if there's a running event loop
            try:
                loop = asyncio.get_running_loop()
                asyncio.create_task(self._event_bus.publish(event))
            except RuntimeError:
                # No event loop running - skip publishing (common in tests)
                logger.debug(f"No event loop running, skipping event publish for {event_type}")

        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to publish execution event: {e}")

    def get_current_node_id(self) -> str | None:
        """Get current executing node ID from ContextVar.

        Returns:
            Current node ID or None
        """
        return _current_node.get()

    def get_execution_depth(self) -> int:
        """Get current execution stack depth.

        Returns:
            Number of nodes in execution stack
        """
        stack = _execution_stack.get()
        return len(stack) if stack else 0

    def __repr__(self) -> str:
        """String representation."""
        running = len(self.graph.get_running_nodes())
        return (
            f"ExecutionGraphTracker(workflow_id={self.graph.workflow_id}, "
            f"nodes={len(self.graph.nodes)}, running={running})"
        )


# Module-level helpers for context manager usage
def get_current_tracker() -> ExecutionGraphTracker | None:
    """Get current workflow tracker from ContextVar.

    Returns:
        Current ExecutionGraphTracker or None
    """
    return _workflow_tracker.get()


def set_current_tracker(tracker: ExecutionGraphTracker | None) -> None:
    """Set current workflow tracker in ContextVar.

    Args:
        tracker: ExecutionGraphTracker to set
    """
    _workflow_tracker.set(tracker)
