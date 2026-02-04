"""Tool registry for managing LLM tools."""

from loguru import logger

from src.tools.base import BaseTool


class ToolRegistry:
    """Registry for LLM-callable tools."""

    def __init__(self) -> None:
        """Initialize empty registry."""
        self._tools: dict[str, BaseTool] = {}
        logger.debug("Initialized ToolRegistry")

    def register(self, tool: BaseTool) -> None:
        """Register a tool.

        Args:
            tool: Tool instance to register
        """
        if tool.name in self._tools:
            logger.warning(f"Overwriting existing tool: {tool.name}")
        self._tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name}")

    def get(self, name: str) -> BaseTool | None:
        """Get tool by name.

        Args:
            name: Tool name

        Returns:
            Tool instance or None if not found
        """
        return self._tools.get(name)

    def get_definitions(self) -> list[dict]:
        """Get all tool definitions for LLM.

        Returns:
            List of tool definition dicts
        """
        return [tool.get_tool_definition() for tool in self._tools.values()]

    def execute(self, name: str, args: dict) -> str:
        """Execute a tool by name.

        Args:
            name: Tool name
            args: Arguments to pass to tool

        Returns:
            Tool execution result

        Raises:
            KeyError: If tool not found
        """
        tool = self._tools.get(name)
        if not tool:
            msg = f"Tool not found: {name}"
            raise KeyError(msg)

        logger.info(f"Executing tool: {name} with args: {args}")
        return tool.execute(**args)

    def requires_confirmation(self, name: str) -> bool:
        """Check if tool requires user confirmation.

        Args:
            name: Tool name

        Returns:
            True if tool requires confirmation
        """
        tool = self._tools.get(name)
        return tool.requires_confirmation if tool else False

    @property
    def tool_names(self) -> list[str]:
        """Get list of registered tool names.

        Returns:
            List of tool names
        """
        return list(self._tools.keys())

    def __len__(self) -> int:
        """Return number of registered tools."""
        return len(self._tools)

    def __repr__(self) -> str:
        """String representation."""
        return f"ToolRegistry(tools={self.tool_names})"
