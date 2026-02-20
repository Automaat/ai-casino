"""Analysis history tool for coordinator."""

import asyncio
from typing import TYPE_CHECKING

from loguru import logger

from src.tools.base import BaseTool
from src.tools.models import ToolDefinition, ToolFunction, ToolParameter, ToolParametersSchema

if TYPE_CHECKING:
    from src.v1.coordinator.memory import CoordinatorMemory


class AnalysisHistoryTool(BaseTool):
    """Tool to retrieve historical analysis from database via memory layer."""

    def __init__(self, memory: CoordinatorMemory) -> None:
        """Initialize tool with coordinator memory.

        Args:
            memory: Coordinator memory instance
        """
        self._memory = memory

    @property
    def name(self) -> str:
        """Tool name."""
        return "analysis_history"

    def get_tool_definition(self) -> ToolDefinition:
        """Get tool definition in LiteLLM/OpenAI format.

        Returns:
            Tool definition for LLM function calling
        """
        return ToolDefinition(
            function=ToolFunction(
                name=self.name,
                description=(
                    "Retrieve historical analysis for a specific symbol from database. "
                    "Shows signals, confidence, execution status, and technical indicators (RSI/MACD). "
                    "Useful for understanding past trading decisions and patterns for a symbol."
                ),
                parameters=ToolParametersSchema(
                    properties={
                        "symbol": ToolParameter(
                            type="string",
                            description="Stock ticker symbol (required, e.g., AAPL)",
                        ),
                        "days": ToolParameter(
                            type="integer",
                            description="Number of days to look back (default: 7, max: 30)",
                        ),
                    },
                    required=["symbol"],
                ),
            ),
        )

    async def aexecute(self, **kwargs: str | int | float | bool) -> str:
        """Execute analysis history retrieval asynchronously.

        Args:
            **kwargs: Tool arguments (symbol: str required, days: int = 7)

        Returns:
            Formatted analysis history
        """
        symbol = kwargs.get("symbol")
        if not symbol or not isinstance(symbol, str):
            return "Error: symbol parameter is required"

        days = min(int(kwargs.get("days", 7)), 30)  # Cap at 30 days

        logger.info(f"Retrieving analysis history for {symbol} (last {days} days)")

        try:
            return await self._memory.get_analysis_history(symbol, days=days)
        except Exception as e:
            logger.opt(exception=True).error(f"Analysis history retrieval failed for {symbol}: {e}")
            return f"Failed to retrieve analysis history for {symbol}: {e}"

    def execute(self, **kwargs: str | int | float | bool) -> str:
        """Execute analysis history retrieval for specific symbol (sync wrapper).

        Args:
            **kwargs: Tool arguments (symbol: str required, days: int = 7)

        Returns:
            Formatted analysis history

        Raises:
            RuntimeError: If called from within a running event loop
        """
        # Guard against being called from within an existing event loop
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No running loop; safe to use asyncio.run
            return asyncio.run(self.aexecute(**kwargs))
        else:
            # There is a running loop; callers should use the async API directly
            msg = (
                "AnalysisHistoryTool.execute() cannot be called from a running "
                "event loop. Use 'aexecute' instead."
            )
            raise RuntimeError(msg)

    def __repr__(self) -> str:
        """String representation."""
        return "AnalysisHistoryTool()"
