"""Execution tracking models for real-time visualization."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class ExecutionNodeType(StrEnum):
    """Type of execution node."""

    TOOL = "TOOL"
    AGENT = "AGENT"
    WORKFLOW_STAGE = "WORKFLOW_STAGE"


class ExecutionStatus(StrEnum):
    """Status of execution node."""

    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class ExecutionNode(BaseModel):
    """Execution node in the graph."""

    node_id: str = Field(default_factory=lambda: uuid4().hex[:8])
    node_type: ExecutionNodeType
    name: str = Field(description="Human-readable name (e.g., 'FetchMarketDataTool', 'TechnicalAnalyst')")
    parent_id: str | None = Field(default=None, description="Parent node ID for hierarchy")
    status: ExecutionStatus = ExecutionStatus.RUNNING
    start_time: datetime = Field(default_factory=lambda: datetime.now(UTC))
    end_time: datetime | None = None
    duration_ms: float | None = Field(default=None, description="Duration in milliseconds")
    error: str | None = Field(default=None, description="Error message if failed")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional context (args, result summary)"
    )

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ExecutionNode(id={self.node_id}, type={self.node_type}, name={self.name}, status={self.status})"
        )

    def complete(self) -> None:
        """Mark node as completed."""
        self.status = ExecutionStatus.COMPLETED
        self.end_time = datetime.now(UTC)
        if self.end_time:
            self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000

    def fail(self, error: str) -> None:
        """Mark node as failed."""
        self.status = ExecutionStatus.FAILED
        self.error = error
        self.end_time = datetime.now(UTC)
        if self.end_time:
            self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000


class ExecutionGraph(BaseModel):
    """Execution graph for a workflow run."""

    workflow_id: str | UUID = Field(description="Workflow/analysis run ID")
    symbol: str | None = Field(default=None, description="Stock symbol being analyzed")
    root_node_id: str | None = Field(default=None, description="Root node of the graph")
    nodes: dict[str, ExecutionNode] = Field(default_factory=dict, description="Nodes by node_id")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ExecutionGraph(workflow_id={self.workflow_id}, symbol={self.symbol}, nodes={len(self.nodes)})"
        )

    def add_node(self, node: ExecutionNode) -> None:
        """Add node to graph."""
        self.nodes[node.node_id] = node
        self.updated_at = datetime.now(UTC)
        if self.root_node_id is None and node.parent_id is None:
            self.root_node_id = node.node_id

    def get_node(self, node_id: str) -> ExecutionNode | None:
        """Get node by ID."""
        return self.nodes.get(node_id)

    def update_node(self, node_id: str, **updates: Any) -> None:
        """Update node fields."""
        if node := self.nodes.get(node_id):
            for key, value in updates.items():
                setattr(node, key, value)
            self.updated_at = datetime.now(UTC)

    def get_children(self, parent_id: str) -> list[ExecutionNode]:
        """Get child nodes of a parent."""
        return [node for node in self.nodes.values() if node.parent_id == parent_id]

    def get_running_nodes(self) -> list[ExecutionNode]:
        """Get all running nodes."""
        return [node for node in self.nodes.values() if node.status == ExecutionStatus.RUNNING]

    def is_completed(self) -> bool:
        """Check if all nodes completed (success or failure)."""
        return all(
            node.status in (ExecutionStatus.COMPLETED, ExecutionStatus.FAILED) for node in self.nodes.values()
        )
