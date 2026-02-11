"""Tests for ToolRegistry."""

import asyncio

import pytest

from src.tools.base import BaseTool
from src.tools.models import ToolDefinition, ToolFunction, ToolParameter, ToolParametersSchema
from src.tools.registry import ToolRegistry


class MockTool(BaseTool):
    """Mock tool for testing."""

    def __init__(self, tool_name: str = "mock_tool", confirms: bool = False) -> None:
        """Initialize mock tool."""
        self._name = tool_name
        self._confirms = confirms

    @property
    def name(self) -> str:
        """Tool name."""
        return self._name

    @property
    def requires_confirmation(self) -> bool:
        """Requires confirmation."""
        return self._confirms

    def get_tool_definition(self) -> ToolDefinition:
        """Get tool definition."""
        return ToolDefinition(
            function=ToolFunction(
                name=self.name,
                description=f"Mock tool: {self.name}",
                parameters=ToolParametersSchema(
                    properties={"arg1": ToolParameter(type="string", description="Argument 1")},
                    required=[],
                ),
            )
        )

    def execute(self, **kwargs: str | int | float | bool) -> str:
        """Execute tool."""
        arg1 = str(kwargs.get("arg1", "default"))
        return f"mock result: {arg1}"


class TestToolRegistry:
    """Tests for ToolRegistry."""

    def test_register_tool(self):
        """Test registering a tool."""
        registry = ToolRegistry()
        tool = MockTool()

        registry.register(tool)

        assert len(registry) == 1
        assert "mock_tool" in registry.tool_names

    def test_register_multiple_tools(self):
        """Test registering multiple tools."""
        registry = ToolRegistry()
        tool1 = MockTool("tool_one")
        tool2 = MockTool("tool_two")

        registry.register(tool1)
        registry.register(tool2)

        assert len(registry) == 2
        assert "tool_one" in registry.tool_names
        assert "tool_two" in registry.tool_names

    def test_register_overwrites_existing(self):
        """Test that registering same name overwrites."""
        registry = ToolRegistry()
        tool1 = MockTool("same_name")
        tool2 = MockTool("same_name")

        registry.register(tool1)
        registry.register(tool2)

        assert len(registry) == 1

    def test_get_tool(self):
        """Test getting a tool by name."""
        registry = ToolRegistry()
        tool = MockTool()
        registry.register(tool)

        retrieved = registry.get("mock_tool")

        assert retrieved is tool

    def test_get_nonexistent_returns_none(self):
        """Test getting nonexistent tool returns None."""
        registry = ToolRegistry()

        retrieved = registry.get("nonexistent")

        assert retrieved is None

    def test_get_definitions(self):
        """Test getting all tool definitions."""
        registry = ToolRegistry()
        registry.register(MockTool("tool_one"))
        registry.register(MockTool("tool_two"))

        definitions = registry.get_definitions()

        assert len(definitions) == 2
        names = [d.function.name for d in definitions]
        assert "tool_one" in names
        assert "tool_two" in names

    def test_execute_tool(self):
        """Test executing a tool."""
        registry = ToolRegistry()
        registry.register(MockTool())

        result = registry.execute("mock_tool", {"arg1": "test_value"})

        assert "mock result: test_value" in result

    def test_execute_nonexistent_raises(self):
        """Test executing nonexistent tool raises KeyError."""
        registry = ToolRegistry()

        with pytest.raises(KeyError, match="Tool not found"):
            registry.execute("nonexistent", {})

    def test_requires_confirmation_false(self):
        """Test requires_confirmation for tool that doesn't require it."""
        registry = ToolRegistry()
        registry.register(MockTool(confirms=False))

        assert registry.requires_confirmation("mock_tool") is False

    def test_requires_confirmation_true(self):
        """Test requires_confirmation for tool that requires it."""
        registry = ToolRegistry()
        registry.register(MockTool(confirms=True))

        assert registry.requires_confirmation("mock_tool") is True

    def test_requires_confirmation_nonexistent(self):
        """Test requires_confirmation for nonexistent tool."""
        registry = ToolRegistry()

        assert registry.requires_confirmation("nonexistent") is False

    def test_tool_names_property(self):
        """Test tool_names property."""
        registry = ToolRegistry()
        registry.register(MockTool("alpha"))
        registry.register(MockTool("beta"))

        names = registry.tool_names

        assert set(names) == {"alpha", "beta"}

    def test_len(self):
        """Test __len__."""
        registry = ToolRegistry()
        assert len(registry) == 0

        registry.register(MockTool())
        assert len(registry) == 1

    def test_repr(self):
        """Test string representation."""
        registry = ToolRegistry()
        registry.register(MockTool("my_tool"))

        repr_str = repr(registry)

        assert "ToolRegistry" in repr_str
        assert "my_tool" in repr_str

    @pytest.mark.asyncio
    async def test_aexecute_tool(self):
        """Test executing a tool asynchronously."""
        registry = ToolRegistry()
        registry.register(MockTool())
        result = await registry.aexecute("mock_tool", {"arg1": "test_value"})
        assert "mock result: test_value" in result

    @pytest.mark.asyncio
    async def test_aexecute_nonexistent_raises(self):
        """Test executing nonexistent tool asynchronously raises KeyError."""
        registry = ToolRegistry()
        with pytest.raises(KeyError, match="Tool not found"):
            await registry.aexecute("nonexistent", {})

    @pytest.mark.asyncio
    async def test_aexecute_with_custom_async_tool(self):
        """Test executing tool with custom async implementation."""

        class AsyncMockTool(MockTool):
            async def aexecute(self, **kwargs: str | int | float | bool) -> str:
                await asyncio.sleep(0.01)
                arg1 = str(kwargs.get("arg1", "default"))
                return f"async mock result: {arg1}"

        registry = ToolRegistry()
        registry.register(AsyncMockTool())
        result = await registry.aexecute("mock_tool", {"arg1": "async_test"})
        assert "async mock result: async_test" in result
