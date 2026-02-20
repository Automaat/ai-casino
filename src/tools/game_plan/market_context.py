"""Fetch overnight market context tool for game plan agent."""

from src.data.market import MarketDataFetcher
from src.tools.base import BaseTool
from src.tools.models import ToolDefinition, ToolFunction, ToolParameter, ToolParametersSchema


class FetchMarketContextTool(BaseTool):
    """Fetch overnight futures data for market context."""

    def __init__(self, market_fetcher: MarketDataFetcher) -> None:
        """Initialize with market fetcher.

        Args:
            market_fetcher: Market data fetcher for futures
        """
        self._fetcher = market_fetcher

    @property
    def name(self) -> str:
        """Tool name."""
        return "fetch_market_context"

    def get_tool_definition(self) -> ToolDefinition:
        """Get tool definition for LLM.

        Returns:
            Tool definition
        """
        return ToolDefinition(
            function=ToolFunction(
                name=self.name,
                description=(
                    "Fetch overnight futures performance (ES=F, NQ=F, etc.) to gauge market direction."
                ),
                parameters=ToolParametersSchema(
                    properties={
                        "symbols": ToolParameter(
                            type="string",
                            description="Comma-separated futures symbols (default: ES=F,NQ=F)",
                        ),
                    },
                    required=[],
                ),
            ),
        )

    def execute(self, **kwargs: str | int | float | bool) -> str:
        """Fetch overnight futures data.

        Args:
            **kwargs: Optional symbols parameter

        Returns:
            Formatted futures context
        """
        symbols_str = str(kwargs.get("symbols", "ES=F,NQ=F"))
        symbols = [s.strip() for s in symbols_str.split(",")]

        futures = self._fetcher.fetch_overnight_futures(symbols)

        if not futures:
            return "Futures data unavailable"

        lines = ["## Overnight Futures"]
        for symbol, change in futures.items():
            direction = "up" if change > 0 else "down" if change < 0 else "flat"
            lines.append(f"- {symbol}: {change:+.2f}% ({direction})")

        return "\n".join(lines)

    def __repr__(self) -> str:
        """String representation."""
        return "FetchMarketContextTool()"
