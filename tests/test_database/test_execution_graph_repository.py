"""Tests for ExecutionGraphRepository."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.database.repositories.execution_graph import ExecutionGraphRepository
from src.execution_tracking.models import ExecutionGraph, ExecutionNodeType, ExecutionStatus


@pytest.fixture
def mock_session():
    """Mock SQLAlchemy async session."""
    session = MagicMock()
    session.add = MagicMock()
    session.commit = AsyncMock()
    session.execute = AsyncMock()
    session.close = AsyncMock()
    return session


@pytest.mark.asyncio
async def test_create_execution_graph(mock_session, sample_execution_graph):
    """Test creating execution graph."""
    repo = ExecutionGraphRepository(mock_session)

    result = await repo.create(sample_execution_graph)

    assert result.workflow_id == sample_execution_graph.workflow_id
    assert result.symbol == sample_execution_graph.symbol
    assert mock_session.add.called
    assert mock_session.commit.called


@pytest.mark.asyncio
async def test_get_by_workflow_id(mock_session, sample_execution_graph):
    """Test retrieving execution graph by workflow ID."""
    from src.database.models import ExecutionGraphORM

    orm = MagicMock(spec=ExecutionGraphORM)
    orm.workflow_id = str(sample_execution_graph.workflow_id)
    orm.symbol = sample_execution_graph.symbol
    orm.graph_jsonb = sample_execution_graph.model_dump(mode="json")

    execute_result = MagicMock()
    execute_result.scalar_one_or_none = MagicMock(return_value=orm)
    mock_session.execute = AsyncMock(return_value=execute_result)

    repo = ExecutionGraphRepository(mock_session)
    result = await repo.get_by_workflow_id(str(sample_execution_graph.workflow_id))

    assert result is not None
    assert result.workflow_id == sample_execution_graph.workflow_id
    assert result.symbol == sample_execution_graph.symbol


@pytest.mark.asyncio
async def test_get_by_workflow_id_not_found(mock_session):
    """Test retrieving non-existent workflow ID."""
    execute_result = MagicMock()
    execute_result.scalar_one_or_none = MagicMock(return_value=None)
    mock_session.execute = AsyncMock(return_value=execute_result)

    repo = ExecutionGraphRepository(mock_session)
    result = await repo.get_by_workflow_id("nonexistent")

    assert result is None


@pytest.mark.asyncio
async def test_list_recent(mock_session):
    """Test listing recent execution graphs."""
    from src.database.models import ExecutionGraphORM

    graphs = [ExecutionGraph(workflow_id=f"workflow-{i}", symbol="AAPL") for i in range(3)]

    orms = []
    for graph in graphs:
        orm = MagicMock(spec=ExecutionGraphORM)
        orm.workflow_id = str(graph.workflow_id)
        orm.symbol = graph.symbol
        orm.graph_jsonb = graph.model_dump(mode="json")
        orms.append(orm)

    scalars_result = MagicMock()
    scalars_result.all = MagicMock(return_value=orms)

    execute_result = MagicMock()
    execute_result.scalars = MagicMock(return_value=scalars_result)
    mock_session.execute = AsyncMock(return_value=execute_result)

    repo = ExecutionGraphRepository(mock_session)
    results = await repo.list_recent(limit=3)

    assert len(results) == 3
    assert all(isinstance(g, ExecutionGraph) for g in results)


@pytest.mark.asyncio
async def test_delete_before(mock_session):
    """Test deleting old execution graphs."""
    execute_result = MagicMock()
    execute_result.rowcount = 5
    mock_session.execute = AsyncMock(return_value=execute_result)

    repo = ExecutionGraphRepository(mock_session)
    deleted_count = await repo.delete_before(datetime.now(UTC))

    assert deleted_count == 5
    assert mock_session.commit.called


@pytest.mark.asyncio
async def test_to_graph_invalid_jsonb():
    """Test that invalid JSONB raises TypeError."""
    from src.database.models import ExecutionGraphORM

    orm = MagicMock(spec=ExecutionGraphORM)
    orm.workflow_id = "test"
    orm.graph_jsonb = "not a dict"  # Invalid JSONB

    repo = ExecutionGraphRepository(MagicMock())

    with pytest.raises(TypeError, match="Invalid JSONB data"):
        repo._to_graph(orm)


@pytest.mark.asyncio
async def test_jsonb_preserves_metadata(mock_session):
    """Test that metadata is preserved in JSONB roundtrip."""
    from src.database.models import ExecutionGraphORM
    from src.execution_tracking.models import ExecutionNode

    graph = ExecutionGraph(workflow_id="metadata-test", symbol="AAPL")
    node = ExecutionNode(
        node_type=ExecutionNodeType.TOOL,
        name="FetchMarketData",
        status=ExecutionStatus.COMPLETED,
        metadata={
            "symbol": "AAPL",
            "nested": {"key": "value"},
            "list": [1, 2, 3],
        },
    )
    node.complete()
    graph.add_node(node)

    orm = MagicMock(spec=ExecutionGraphORM)
    orm.workflow_id = "metadata-test"
    orm.symbol = "AAPL"
    orm.graph_jsonb = graph.model_dump(mode="json")

    repo = ExecutionGraphRepository(mock_session)
    result = repo._to_graph(orm)

    assert len(result.nodes) == 1
    retrieved_node = next(iter(result.nodes.values()))
    assert retrieved_node.metadata["symbol"] == "AAPL"
    assert retrieved_node.metadata["nested"]["key"] == "value"
    assert retrieved_node.metadata["list"] == [1, 2, 3]
