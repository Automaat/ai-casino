"""Analysis history tool for coordinator."""

from typing import TYPE_CHECKING

from loguru import logger

from src.tools.base import BaseTool
from src.tools.models import ToolDefinition, ToolFunction, ToolParameter, ToolParametersSchema

if TYPE_CHECKING:
    from src.daemon.state import DaemonState


class AnalysisHistoryTool(BaseTool):
    """Tool to retrieve recent analysis history."""

    def __init__(self, daemon_state: DaemonState) -> None:
        """Initialize tool with daemon state.

        Args:
            daemon_state: Daemon state instance
        """
        self._state = daemon_state

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
                    "Retrieve recent analysis history with signals, confidence, execution status, "
                    "and technical indicators (RSI/MACD). Useful for understanding recent trading activity."
                ),
                parameters=ToolParametersSchema(
                    properties={
                        "limit": ToolParameter(
                            type="integer",
                            description="Maximum number of records to retrieve (default: 10, max: 50)",
                        ),
                        "symbol_filter": ToolParameter(
                            type="string",
                            description="Optional symbol filter (e.g., AAPL)",
                        ),
                    },
                    required=[],
                ),
            ),
        )

    def execute(self, **kwargs: str | int | float | bool) -> str:
        """Execute analysis history retrieval.

        Args:
            **kwargs: Tool arguments (limit: int = 10, symbol_filter: str = None)

        Returns:
            Formatted analysis history
        """
        limit = min(int(kwargs.get("limit", 10)), 50)
        symbol_filter = kwargs.get("symbol_filter")

        logger.info(f"Retrieving analysis history (limit={limit}, symbol={symbol_filter})")

        try:
            # Filter and limit records
            records = self._state.analyses
            if symbol_filter:
                records = [r for r in records if r.symbol.upper() == str(symbol_filter).upper()]

            # Get most recent records
            recent = records[-limit:] if len(records) > limit else records
            recent.reverse()  # Most recent first

            if not recent:
                return "No analysis history found"

            lines = [
                "# Analysis History",
                "",
            ]

            for i, record in enumerate(recent, 1):
                executed = "✓" if record.executed_trade else "✗"
                rsi_text = f"RSI: {record.rsi:.1f}" if record.rsi else "RSI: N/A"
                macd_text = f"MACD: {record.macd_hist:.4f}" if record.macd_hist else "MACD: N/A"

                lines.extend(
                    [
                        f"## {i}. {record.symbol} - {record.signal}",
                        f"- **Timestamp:** {record.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
                        f"- **Confidence:** {record.confidence:.0%}",
                        f"- **Executed:** {executed}",
                        f"- **Session:** {record.trading_session.value}",
                        f"- **Indicators:** {rsi_text} | {macd_text}",
                        "",
                    ]
                )

            return "\n".join(lines)

        except Exception as e:
            logger.error(f"Analysis history retrieval failed: {e}")
            return f"Failed to retrieve analysis history: {e}"

    def __repr__(self) -> str:
        """String representation."""
        return "AnalysisHistoryTool()"
