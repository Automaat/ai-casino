"""Tests for ExecutionGraphTracker."""

from unittest.mock import Mock

import pytest

from src.execution_tracking.models import ExecutionNodeType, ExecutionStatus
from src.execution_tracking.tracker import (
    ExecutionGraphTracker,
    _current_node,
    _execution_stack,
    set_current_tracker,
)


@pytest.fixture(autouse=True)
def reset_context() -> None:
    """Reset ContextVar state before each test."""
    set_current_tracker(None)
    _current_node.set(None)
    _execution_stack.set(None)


@pytest.fixture
def mock_event_bus() -> Mock:
    """Mock EventBus for testing."""
    return Mock()


@pytest.fixture
def tracker(mock_event_bus: Mock) -> ExecutionGraphTracker:
    """Create tracker instance for testing."""
    return ExecutionGraphTracker(workflow_id="test-workflow-123", symbol="AAPL", event_bus=mock_event_bus)


def test_tracker_initialization(tracker: ExecutionGraphTracker) -> None:
    """Test tracker initializes with empty graph."""
    assert tracker.graph.workflow_id == "test-workflow-123"
    assert tracker.graph.symbol == "AAPL"
    assert len(tracker.graph.nodes) == 0
    assert tracker.graph.root_node_id is None


def test_start_node_creates_node(tracker: ExecutionGraphTracker) -> None:
    """Test starting a node creates it in graph."""
    node_id = tracker.start_node(ExecutionNodeType.AGENT, "TechnicalAnalyst", metadata={"key": "value"})

    assert node_id in tracker.graph.nodes
    node = tracker.graph.get_node(node_id)
    assert node is not None
    assert node.node_type == ExecutionNodeType.AGENT
    assert node.name == "TechnicalAnalyst"
    assert node.status == ExecutionStatus.RUNNING
    assert node.metadata == {"key": "value"}


def test_start_node_sets_root(tracker: ExecutionGraphTracker) -> None:
    """Test first node becomes root."""
    node_id = tracker.start_node(ExecutionNodeType.WORKFLOW_STAGE, "DataFetch")

    assert tracker.graph.root_node_id == node_id
    node = tracker.graph.get_node(node_id)
    assert node is not None
    assert node.parent_id is None


def test_start_node_parent_child_relationship(tracker: ExecutionGraphTracker) -> None:
    """Test parent-child relationships are correct."""
    parent_id = tracker.start_node(ExecutionNodeType.WORKFLOW_STAGE, "Analysis")
    child_id = tracker.start_node(ExecutionNodeType.AGENT, "TechnicalAnalyst", parent_id=parent_id)

    parent = tracker.graph.get_node(parent_id)
    child = tracker.graph.get_node(child_id)

    assert parent is not None
    assert child is not None
    assert child.parent_id == parent_id
    assert child in tracker.graph.get_children(parent_id)


def test_complete_node_updates_status(tracker: ExecutionGraphTracker) -> None:
    """Test completing node updates status and sets duration."""
    node_id = tracker.start_node(ExecutionNodeType.TOOL, "FetchMarketDataTool")

    tracker.complete_node(node_id, metadata={"result": "success"})

    node = tracker.graph.get_node(node_id)
    assert node is not None
    assert node.status == ExecutionStatus.COMPLETED
    assert node.end_time is not None
    assert node.duration_ms is not None
    assert node.duration_ms >= 0
    assert node.metadata["result"] == "success"


def test_fail_node_updates_status(tracker: ExecutionGraphTracker) -> None:
    """Test failing node updates status and sets error."""
    node_id = tracker.start_node(ExecutionNodeType.AGENT, "SentimentAnalyst")

    tracker.fail_node(node_id, "API error")

    node = tracker.graph.get_node(node_id)
    assert node is not None
    assert node.status == ExecutionStatus.FAILED
    assert node.error == "API error"
    assert node.end_time is not None
    assert node.duration_ms is not None


def test_complete_unknown_node(tracker: ExecutionGraphTracker) -> None:
    """Test completing unknown node (no-op)."""
    # Should not raise, just log warning
    tracker.complete_node("nonexistent-node-id")

    # Node should not exist in graph
    assert tracker.graph.get_node("nonexistent-node-id") is None


def test_get_running_nodes(tracker: ExecutionGraphTracker) -> None:
    """Test getting running nodes."""
    node1_id = tracker.start_node(ExecutionNodeType.AGENT, "Agent1")
    node2_id = tracker.start_node(ExecutionNodeType.AGENT, "Agent2")
    tracker.start_node(ExecutionNodeType.AGENT, "Agent3")

    # Complete first two
    tracker.complete_node(node1_id)
    tracker.fail_node(node2_id, "error")

    running = tracker.graph.get_running_nodes()
    assert len(running) == 1
    assert running[0].name == "Agent3"


def test_is_completed(tracker: ExecutionGraphTracker) -> None:
    """Test graph completion detection."""
    node1 = tracker.start_node(ExecutionNodeType.AGENT, "Agent1")
    node2 = tracker.start_node(ExecutionNodeType.AGENT, "Agent2")

    assert not tracker.graph.is_completed()

    tracker.complete_node(node1)
    assert not tracker.graph.is_completed()

    tracker.complete_node(node2)
    assert tracker.graph.is_completed()


def test_execution_depth() -> None:
    """Test execution stack depth method exists (stack managed by context manager)."""
    from src.execution_tracking.tracker import _execution_stack, set_current_tracker

    # Reset context
    _execution_stack.set(None)
    set_current_tracker(None)

    tracker = ExecutionGraphTracker(workflow_id="test", symbol="AAPL")

    # Get_execution_depth should work even without context manager usage
    # (returns 0 when no stack or when stack is empty)
    depth = tracker.get_execution_depth()
    assert depth == 0


def test_repr(tracker: ExecutionGraphTracker) -> None:
    """Test string representation."""
    tracker.start_node(ExecutionNodeType.AGENT, "Agent1")
    tracker.start_node(ExecutionNodeType.AGENT, "Agent2")

    repr_str = repr(tracker)
    assert "test-workflow-123" in repr_str
    assert "nodes=2" in repr_str
