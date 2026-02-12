"""Tests for execution tracking decorators."""

import pytest

from src.execution_tracking.context import track_workflow
from src.execution_tracking.decorators import track_agent
from src.execution_tracking.models import ExecutionNodeType, ExecutionStatus
from src.execution_tracking.tracker import set_current_tracker


@pytest.fixture(autouse=True)
def reset_context() -> None:
    """Reset ContextVar state before each test."""
    from src.execution_tracking.tracker import _current_node, _execution_stack

    set_current_tracker(None)
    _current_node.set(None)
    _execution_stack.set(None)


def test_track_agent_decorator_sync() -> None:
    """Test @track_agent decorator on sync method."""

    class TestAnalyst:
        @track_agent
        def analyze(self, symbol: str, data: str) -> str:
            return f"Analysis for {symbol}: {data}"

    analyst = TestAnalyst()

    with track_workflow("workflow-123") as tracker:
        result = analyst.analyze("AAPL", "test-data")

        assert result == "Analysis for AAPL: test-data"
        assert len(tracker.graph.nodes) == 1

        node = list(tracker.graph.nodes.values())[0]
        assert node.node_type == ExecutionNodeType.AGENT
        assert "TestAnalyst" in node.name  # May include test function path for local classes
        assert node.status == ExecutionStatus.COMPLETED
        assert "TestAnalyst" in node.metadata["agent"]
        assert node.metadata["method"] == "analyze"
        assert node.metadata["symbol"] == "AAPL"


@pytest.mark.asyncio
async def test_track_agent_decorator_async() -> None:
    """Test @track_agent decorator on async method."""

    class AsyncAnalyst:
        @track_agent
        async def analyze(self, symbol: str, data: str) -> str:
            return f"Async analysis for {symbol}: {data}"

    analyst = AsyncAnalyst()

    with track_workflow("workflow-123") as tracker:
        result = await analyst.analyze("TSLA", "test-data")

        assert result == "Async analysis for TSLA: test-data"
        assert len(tracker.graph.nodes) == 1

        node = list(tracker.graph.nodes.values())[0]
        assert node.node_type == ExecutionNodeType.AGENT
        assert "AsyncAnalyst" in node.name  # May include test function path for local classes
        assert node.status == ExecutionStatus.COMPLETED


def test_track_agent_without_tracker() -> None:
    """Test @track_agent works without active tracker (no-op)."""

    class TestAnalyst:
        @track_agent
        def analyze(self, symbol: str) -> str:
            return f"Analysis for {symbol}"

    analyst = TestAnalyst()
    result = analyst.analyze("AAPL")

    assert result == "Analysis for AAPL"


def test_track_agent_on_exception() -> None:
    """Test @track_agent marks node as failed on exception."""

    class FailingAnalyst:
        @track_agent
        def analyze(self, symbol: str) -> str:
            raise ValueError(f"Analysis failed for {symbol}")

    analyst = FailingAnalyst()

    with track_workflow("workflow-123") as tracker:
        with pytest.raises(ValueError, match="Analysis failed for AAPL"):
            analyst.analyze("AAPL")

        assert len(tracker.graph.nodes) == 1
        node = list(tracker.graph.nodes.values())[0]
        assert node.status == ExecutionStatus.FAILED
        assert "Analysis failed for AAPL" in node.error


@pytest.mark.asyncio
async def test_track_agent_async_exception() -> None:
    """Test @track_agent marks async node as failed on exception."""

    class AsyncFailingAnalyst:
        @track_agent
        async def analyze(self, symbol: str) -> str:
            raise RuntimeError(f"Async failure for {symbol}")

    analyst = AsyncFailingAnalyst()

    with track_workflow("workflow-123") as tracker:
        with pytest.raises(RuntimeError, match="Async failure for TSLA"):
            await analyst.analyze("TSLA")

        node = list(tracker.graph.nodes.values())[0]
        assert node.status == ExecutionStatus.FAILED


def test_track_agent_extracts_symbol_from_kwargs() -> None:
    """Test @track_agent extracts symbol from kwargs."""

    class TestAnalyst:
        @track_agent
        def analyze(self, data: str, symbol: str) -> str:
            return f"Analysis for {symbol}"

    analyst = TestAnalyst()

    with track_workflow("workflow-123") as tracker:
        analyst.analyze("data", symbol="NVDA")

        node = list(tracker.graph.nodes.values())[0]
        assert node.metadata["symbol"] == "NVDA"


def test_track_agent_no_symbol() -> None:
    """Test @track_agent works when no symbol in args."""

    class TestAnalyst:
        @track_agent
        def process(self, data: str) -> str:
            return f"Processed: {data}"

    analyst = TestAnalyst()

    with track_workflow("workflow-123") as tracker:
        result = analyst.process("test")

        assert result == "Processed: test"
        node = list(tracker.graph.nodes.values())[0]
        assert "symbol" not in node.metadata


def test_track_agent_preserves_function_name() -> None:
    """Test @track_agent preserves original function metadata."""

    class TestAnalyst:
        @track_agent
        def analyze(self, symbol: str) -> str:
            """Analyze symbol."""
            return f"Analysis for {symbol}"

    assert TestAnalyst.analyze.__name__ == "analyze"
    assert "Analyze symbol" in TestAnalyst.analyze.__doc__
