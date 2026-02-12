"""E2E tests for execution tracking visualization."""

import asyncio
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from loguru import logger

from src.daemon.api.app import create_api_app
from src.daemon.event_bus import DashboardEvent, EventBus, EventType
from src.execution_tracking.models import ExecutionGraph, ExecutionNode, ExecutionNodeType, ExecutionStatus


@pytest.fixture
def mock_daemon_components():
    """Mock DaemonComponents with execution tracking."""
    from src.daemon.config import DaemonConfig
    from src.daemon.factory import DaemonComponents

    # Create minimal config
    config = DaemonConfig()
    config.database.enable_persistence = False
    config.api.enabled = True
    config.api.cors_origins = ["testclient"]  # Allow TestClient origin

    # Create components
    components = Mock(spec=DaemonComponents)
    components.config = config

    # Mock state with active execution trackers
    state = Mock()
    state.active_execution_trackers = {}
    components.state = state

    components.event_bus = EventBus(history_size=100, queue_size=50)
    components.broker = None

    return components


@pytest.mark.integration
@pytest.mark.asyncio
async def test_execution_graph_active_endpoint(mock_daemon_components):
    """Test /api/execution/active returns active execution graphs."""
    # Create test execution graph
    workflow_id = "test-workflow-123"
    graph = ExecutionGraph(
        workflow_id=workflow_id,
        symbol="AAPL",
        root_node_id="root-node",
        nodes={
            "root-node": ExecutionNode(
                node_id="root-node",
                node_type=ExecutionNodeType.WORKFLOW_STAGE,
                name="TradingWorkflow",
                parent_id=None,
                status=ExecutionStatus.RUNNING,
            )
        },
    )

    # Add graph to active trackers
    mock_tracker = Mock()
    mock_tracker.graph = graph
    mock_daemon_components.state.active_execution_trackers[workflow_id] = mock_tracker

    # Create FastAPI app
    app = create_api_app(mock_daemon_components)

    # Test endpoint
    with TestClient(app) as client:
        response = client.get("/api/execution/active")
        assert response.status_code == 200

        data = response.json()
        assert data["count"] == 1
        assert len(data["graphs"]) == 1

        graph_data = data["graphs"][0]
        assert graph_data["workflow_id"] == workflow_id
        assert graph_data["symbol"] == "AAPL"
        assert graph_data["root_node_id"] == "root-node"
        assert len(graph_data["nodes"]) == 1
        assert "root-node" in graph_data["nodes"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_execution_node_websocket_events(mock_daemon_components):
    """Test WebSocket receives execution node events."""
    app = create_api_app(mock_daemon_components)

    # Publish test events
    node_start_event = DashboardEvent(
        event_type=EventType.EXECUTION_NODE_START,
        data={
            "workflow_id": "test-workflow-456",
            "node": {
                "node_id": "node-1",
                "node_type": "AGENT",
                "name": "TechnicalAnalyst",
                "parent_id": None,
                "status": "RUNNING",
                "start_time": "2024-01-01T00:00:00Z",
                "end_time": None,
                "duration_ms": None,
                "error": None,
                "metadata": {"symbol": "TSLA"},
            },
        },
    )

    node_complete_event = DashboardEvent(
        event_type=EventType.EXECUTION_NODE_COMPLETE,
        data={
            "workflow_id": "test-workflow-456",
            "node": {
                "node_id": "node-1",
                "node_type": "AGENT",
                "name": "TechnicalAnalyst",
                "parent_id": None,
                "status": "COMPLETED",
                "start_time": "2024-01-01T00:00:00Z",
                "end_time": "2024-01-01T00:00:02Z",
                "duration_ms": 2000.0,
                "error": None,
                "metadata": {"symbol": "TSLA"},
            },
        },
    )

    # Connect WebSocket (using TestClient WebSocket support)
    with TestClient(app) as client:
        with client.websocket_connect("/ws/events", headers={"origin": "testclient"}) as websocket:
            # Publish events via EventBus
            await mock_daemon_components.event_bus.publish(node_start_event)
            await mock_daemon_components.event_bus.publish(node_complete_event)

            # Give time for events to propagate
            await asyncio.sleep(0.1)

            # Receive events from WebSocket
            received_events = []
            try:
                # Receive with timeout to avoid hanging
                for _ in range(2):
                    data = websocket.receive_json()
                    if data.get("type") != "ping":  # Skip ping messages
                        received_events.append(data)
            except Exception as e:
                logger.opt(exception=True).warning(f"WebSocket receive failed: {e}")

            # Verify events received
            assert len(received_events) >= 2

            # Verify EXECUTION_NODE_START event
            start_event = next(
                (e for e in received_events if e["event_type"] == "EXECUTION_NODE_START"), None
            )
            assert start_event is not None
            assert start_event["data"]["workflow_id"] == "test-workflow-456"
            assert start_event["data"]["node"]["node_id"] == "node-1"
            assert start_event["data"]["node"]["status"] == "RUNNING"

            # Verify EXECUTION_NODE_COMPLETE event
            complete_event = next(
                (e for e in received_events if e["event_type"] == "EXECUTION_NODE_COMPLETE"), None
            )
            assert complete_event is not None
            assert complete_event["data"]["workflow_id"] == "test-workflow-456"
            assert complete_event["data"]["node"]["status"] == "COMPLETED"
            assert complete_event["data"]["node"]["duration_ms"] == 2000.0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_execution_tracking_full_workflow(mock_daemon_components):
    """Test full execution tracking workflow with real-time updates."""
    workflow_id = "workflow-full-test"

    # Create execution graph with multiple nodes
    root_node = ExecutionNode(
        node_id="root",
        node_type=ExecutionNodeType.WORKFLOW_STAGE,
        name="TradingWorkflow",
        parent_id=None,
        status=ExecutionStatus.RUNNING,
    )

    agent_node = ExecutionNode(
        node_id="agent-1",
        node_type=ExecutionNodeType.AGENT,
        name="TechnicalAnalyst",
        parent_id="root",
        status=ExecutionStatus.RUNNING,
    )

    tool_node = ExecutionNode(
        node_id="tool-1",
        node_type=ExecutionNodeType.TOOL,
        name="FetchMarketDataTool",
        parent_id="agent-1",
        status=ExecutionStatus.COMPLETED,
    )
    tool_node.complete()

    graph = ExecutionGraph(
        workflow_id=workflow_id,
        symbol="NVDA",
        root_node_id="root",
        nodes={
            "root": root_node,
            "agent-1": agent_node,
            "tool-1": tool_node,
        },
    )

    # Add to active trackers
    mock_tracker = Mock()
    mock_tracker.graph = graph
    mock_daemon_components.state.active_execution_trackers[workflow_id] = mock_tracker

    # Create app and test
    app = create_api_app(mock_daemon_components)

    with TestClient(app) as client:
        # Query active graphs endpoint
        response = client.get("/api/execution/active")
        assert response.status_code == 200

        data = response.json()
        assert data["count"] == 1

        graph_data = data["graphs"][0]
        assert graph_data["workflow_id"] == workflow_id
        assert graph_data["symbol"] == "NVDA"
        assert len(graph_data["nodes"]) == 3

        # Verify node hierarchy
        assert graph_data["nodes"]["root"]["parent_id"] is None
        assert graph_data["nodes"]["agent-1"]["parent_id"] == "root"
        assert graph_data["nodes"]["tool-1"]["parent_id"] == "agent-1"

        # Verify status
        assert graph_data["nodes"]["root"]["status"] == "RUNNING"
        assert graph_data["nodes"]["agent-1"]["status"] == "RUNNING"
        assert graph_data["nodes"]["tool-1"]["status"] == "COMPLETED"
        assert graph_data["nodes"]["tool-1"]["duration_ms"] is not None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_multiple_concurrent_workflows(mock_daemon_components):
    """Test tracking multiple concurrent workflows."""
    # Create two workflows
    workflow1 = ExecutionGraph(
        workflow_id="workflow-1",
        symbol="AAPL",
        root_node_id="root-1",
        nodes={
            "root-1": ExecutionNode(
                node_id="root-1",
                node_type=ExecutionNodeType.WORKFLOW_STAGE,
                name="Workflow1",
                parent_id=None,
                status=ExecutionStatus.RUNNING,
            )
        },
    )

    workflow2 = ExecutionGraph(
        workflow_id="workflow-2",
        symbol="TSLA",
        root_node_id="root-2",
        nodes={
            "root-2": ExecutionNode(
                node_id="root-2",
                node_type=ExecutionNodeType.WORKFLOW_STAGE,
                name="Workflow2",
                parent_id=None,
                status=ExecutionStatus.RUNNING,
            )
        },
    )

    # Add both to trackers
    mock_tracker1 = Mock()
    mock_tracker1.graph = workflow1
    mock_tracker2 = Mock()
    mock_tracker2.graph = workflow2

    mock_daemon_components.state.active_execution_trackers["workflow-1"] = mock_tracker1
    mock_daemon_components.state.active_execution_trackers["workflow-2"] = mock_tracker2

    # Test endpoint
    app = create_api_app(mock_daemon_components)

    with TestClient(app) as client:
        response = client.get("/api/execution/active")
        assert response.status_code == 200

        data = response.json()
        assert data["count"] == 2
        assert len(data["graphs"]) == 2

        # Verify both workflows present
        symbols = [g["symbol"] for g in data["graphs"]]
        assert "AAPL" in symbols
        assert "TSLA" in symbols
