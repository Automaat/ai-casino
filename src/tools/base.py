"""Base tool interface for LLM function calling."""

from abc import ABC, abstractmethod

from src.tools.models import ToolDefinition


class BaseTool(ABC):
    """Abstract base class for LLM tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name used in function calling.

        Returns:
            Unique tool identifier
        """

    @property
    def requires_confirmation(self) -> bool:
        """Whether tool requires user confirmation before execution.

        Returns:
            True if tool should prompt user before running
        """
        return False

    @abstractmethod
    def get_tool_definition(self) -> ToolDefinition:
        """Get tool definition in LiteLLM/OpenAI format.

        Returns:
            Tool definition for LLM function calling
        """

    @abstractmethod
    def execute(self, **kwargs: str | int | float | bool) -> str:
        """Execute the tool with given arguments.

        Args:
            **kwargs: Tool-specific arguments

        Returns:
            Tool execution result as string
        """

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name={self.name})"
