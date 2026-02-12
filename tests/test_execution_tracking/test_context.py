"""Tests for execution tracking context managers."""

from unittest.mock import Mock

import pytest

from src.execution_tracking.context import (
    atrack_execution,
    track_execution,
    track_workflow,
)
from src.execution_tracking.models import ExecutionNodeType, ExecutionStatus
from src.execution_tracking.tracker import get_current_tracker, set_current_tracker


@pytest.fixture(autouse=True)
def reset_context() -> None:
    """Reset ContextVar state before each test."""
    from src.execution_tracking.tracker import _current_node, _execution_stack

    set_current_tracker(None)
    _current_node.set(None)
    _execution_stack.set(None)


def test_track_workflow_creates_tracker() -> None:
    """Test track_workflow creates and sets tracker."""
    assert get_current_tracker() is None

    with track_workflow("workflow-123", symbol="AAPL") as tracker:
        assert tracker is not None
        assert tracker.graph.workflow_id == "workflow-123"
        assert tracker.graph.symbol == "AAPL"
        assert get_current_tracker() == tracker

    # Tracker cleared after context
    assert get_current_tracker() is None


def test_track_workflow_restores_previous_tracker() -> None:
    """Test nested track_workflow restores previous tracker."""
    with track_workflow("outer-workflow") as outer_tracker:
        assert get_current_tracker() == outer_tracker

        with track_workflow("inner-workflow") as inner_tracker:
            assert get_current_tracker() == inner_tracker
            assert inner_tracker != outer_tracker

        # Outer restored
        assert get_current_tracker() == outer_tracker

    # All cleared
    assert get_current_tracker() is None


def test_track_execution_without_tracker() -> None:
    """Test track_execution with no active tracker yields None."""
    assert get_current_tracker() is None

    with track_execution(ExecutionNodeType.AGENT, "TestAgent") as node_id:
        assert node_id is None


def test_track_execution_with_tracker() -> None:
    """Test track_execution creates node when tracker active."""
    with track_workflow("workflow-123") as tracker:
        with track_execution(ExecutionNodeType.AGENT, "TechnicalAnalyst") as node_id:
            assert node_id is not None
            assert node_id in tracker.graph.nodes

            node = tracker.graph.get_node(node_id)
            assert node is not None
            assert node.name == "TechnicalAnalyst"
            assert node.status == ExecutionStatus.RUNNING

        # Node completed after context
        node = tracker.graph.get_node(node_id)
        assert node is not None
        assert node.status == ExecutionStatus.COMPLETED


def test_track_execution_on_error() -> None:
    """Test track_execution marks node as failed on exception."""
    with track_workflow("workflow-123") as tracker:
        with pytest.raises(ValueError, match="Test error"):
            with track_execution(ExecutionNodeType.TOOL, "FetchDataTool") as node_id:
                raise ValueError("Test error")

        # Node marked as failed
        node = tracker.graph.get_node(node_id)
        assert node is not None
        assert node.status == ExecutionStatus.FAILED
        assert node.error == "Test error"


def test_track_execution_nested_hierarchy() -> None:
    """Test nested track_execution creates parent-child relationships."""
    with track_workflow("workflow-123") as tracker:
        with track_execution(ExecutionNodeType.WORKFLOW_STAGE, "Analysis") as parent_id:
            with track_execution(ExecutionNodeType.AGENT, "TechnicalAnalyst") as child_id:
                # Verify parent-child
                parent = tracker.graph.get_node(parent_id)
                child = tracker.graph.get_node(child_id)

                assert child is not None
                assert parent is not None
                assert child.parent_id == parent_id
                assert child in tracker.graph.get_children(parent_id)


def test_track_execution_with_metadata() -> None:
    """Test track_execution stores metadata."""
    with track_workflow("workflow-123") as tracker:
        metadata = {"symbol": "AAPL", "period": 30}
        with track_execution(ExecutionNodeType.TOOL, "FetchTool", metadata=metadata) as node_id:
            pass

        node = tracker.graph.get_node(node_id)
        assert node is not None
        assert node.metadata == metadata


@pytest.mark.asyncio
async def test_atrack_execution_async() -> None:
    """Test async version of track_execution."""
    with track_workflow("workflow-123") as tracker:
        async with atrack_execution(ExecutionNodeType.AGENT, "AsyncAgent") as node_id:
            assert node_id is not None
            node = tracker.graph.get_node(node_id)
            assert node is not None
            assert node.status == ExecutionStatus.RUNNING

        # Completed after context
        node = tracker.graph.get_node(node_id)
        assert node is not None
        assert node.status == ExecutionStatus.COMPLETED


@pytest.mark.asyncio
async def test_atrack_execution_on_error() -> None:
    """Test async track_execution handles errors."""
    with track_workflow("workflow-123") as tracker:
        with pytest.raises(RuntimeError, match="Async error"):
            async with atrack_execution(ExecutionNodeType.AGENT, "AsyncAgent") as node_id:
                raise RuntimeError("Async error")

        node = tracker.graph.get_node(node_id)
        assert node is not None
        assert node.status == ExecutionStatus.FAILED
        assert node.error == "Async error"


def test_track_workflow_with_event_bus() -> None:
    """Test track_workflow passes event_bus to tracker."""
    mock_bus = Mock()

    with track_workflow("workflow-123", event_bus=mock_bus) as tracker:
        assert tracker._event_bus == mock_bus
