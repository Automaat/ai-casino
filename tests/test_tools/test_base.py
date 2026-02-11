"""Tests for BaseTool interface."""

import asyncio

import pytest

from src.tools.base import BaseTool
from src.tools.models import ToolDefinition, ToolFunction, ToolParametersSchema


class ConcreteTool(BaseTool):
    """Concrete implementation for testing."""

    @property
    def name(self) -> str:
        """Tool name."""
        return "test_tool"

    def get_tool_definition(self) -> ToolDefinition:
        """Get tool definition."""
        return ToolDefinition(
            function=ToolFunction(
                name=self.name,
                description="Test tool",
                parameters=ToolParametersSchema(properties={}, required=[]),
            )
        )

    def execute(self, **kwargs: str | int | float | bool) -> str:
        """Execute tool."""
        return f"executed with {kwargs}"


class ConcreteToolWithConfirmation(ConcreteTool):
    """Concrete implementation with confirmation."""

    @property
    def requires_confirmation(self) -> bool:
        """Requires confirmation."""
        return True


class AsyncConcreteTool(BaseTool):
    """Concrete implementation with native async execute."""

    @property
    def name(self) -> str:
        return "async_test_tool"

    def get_tool_definition(self) -> ToolDefinition:
        return ToolDefinition(
            function=ToolFunction(
                name=self.name,
                description="Async test tool",
                parameters=ToolParametersSchema(properties={}, required=[]),
            )
        )

    def execute(self, **kwargs: str | int | float | bool) -> str:
        return f"sync executed with {kwargs}"

    async def aexecute(self, **kwargs: str | int | float | bool) -> str:
        await asyncio.sleep(0.01)  # Simulate async operation
        return f"async executed with {kwargs}"


class TestBaseTool:
    """Tests for BaseTool."""

    def test_cannot_instantiate_abstract(self):
        """Test that BaseTool cannot be instantiated directly."""
        with pytest.raises(TypeError, match="abstract"):
            BaseTool()  # type: ignore[abstract]

    def test_concrete_tool_name(self):
        """Test name property on concrete implementation."""
        tool = ConcreteTool()
        assert tool.name == "test_tool"

    def test_concrete_tool_definition(self):
        """Test get_tool_definition on concrete implementation."""
        tool = ConcreteTool()
        definition = tool.get_tool_definition().model_dump(mode="json", by_alias=True, exclude_none=True)

        assert definition["type"] == "function"
        assert definition["function"]["name"] == "test_tool"
        assert "description" in definition["function"]

    def test_concrete_tool_execute(self):
        """Test execute on concrete implementation."""
        tool = ConcreteTool()
        result = tool.execute(foo="bar", num=42)

        assert "executed with" in result
        assert "foo" in result
        assert "bar" in result

    def test_requires_confirmation_default_false(self):
        """Test that requires_confirmation defaults to False."""
        tool = ConcreteTool()
        assert tool.requires_confirmation is False

    def test_requires_confirmation_override(self):
        """Test requires_confirmation can be overridden."""
        tool = ConcreteToolWithConfirmation()
        assert tool.requires_confirmation is True

    def test_repr(self):
        """Test string representation."""
        tool = ConcreteTool()
        repr_str = repr(tool)

        assert "ConcreteTool" in repr_str
        assert "test_tool" in repr_str

    @pytest.mark.asyncio
    async def test_aexecute_default_offloads_to_thread(self):
        """Test that default aexecute offloads sync execute to thread."""
        tool = ConcreteTool()
        result = await tool.aexecute(foo="bar", num=42)
        assert "executed with" in result
        assert "foo" in result

    @pytest.mark.asyncio
    async def test_aexecute_can_be_overridden(self):
        """Test that aexecute can be overridden for native async."""
        tool = AsyncConcreteTool()
        result = await tool.aexecute(foo="bar")
        assert "async executed with" in result

    @pytest.mark.asyncio
    async def test_aexecute_and_execute_both_work(self):
        """Test that both sync and async execution work on same tool."""
        tool = AsyncConcreteTool()
        sync_result = tool.execute(test="sync")
        assert "sync executed with" in sync_result
        async_result = await tool.aexecute(test="async")
        assert "async executed with" in async_result
