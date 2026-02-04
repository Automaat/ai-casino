"""Tests for BaseTool interface."""

import pytest

from src.tools.base import BaseTool


class ConcreteTool(BaseTool):
    """Concrete implementation for testing."""

    @property
    def name(self) -> str:
        """Tool name."""
        return "test_tool"

    def get_tool_definition(self) -> dict:
        """Get tool definition."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "Test tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }

    def execute(self, **kwargs: str | int | float | bool) -> str:
        """Execute tool."""
        return f"executed with {kwargs}"


class ConcreteToolWithConfirmation(ConcreteTool):
    """Concrete implementation with confirmation."""

    @property
    def requires_confirmation(self) -> bool:
        """Requires confirmation."""
        return True


class TestBaseTool:
    """Tests for BaseTool."""

    def test_cannot_instantiate_abstract(self):
        """Test that BaseTool cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseTool()

    def test_concrete_tool_name(self):
        """Test name property on concrete implementation."""
        tool = ConcreteTool()
        assert tool.name == "test_tool"

    def test_concrete_tool_definition(self):
        """Test get_tool_definition on concrete implementation."""
        tool = ConcreteTool()
        definition = tool.get_tool_definition()

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
