"""Market overview tool for coordinator."""

from typing import TYPE_CHECKING, Final

from loguru import logger

from src.tools.base import BaseTool
from src.tools.models import ToolDefinition, ToolFunction, ToolParameter, ToolParametersSchema

if TYPE_CHECKING:
    from src.data.market import MarketDataFetcher

BULLISH_THRESHOLD: Final[float] = 0.3
BEARISH_THRESHOLD: Final[float] = -0.3


class MarketOverviewTool(BaseTool):
    """Tool to fetch overnight futures and market sentiment."""

    def __init__(self, market_fetcher: MarketDataFetcher) -> None:
        """Initialize tool with market data fetcher.

        Args:
            market_fetcher: Market data fetcher instance
        """
        self._market_fetcher = market_fetcher

    @property
    def name(self) -> str:
        """Tool name."""
        return "market_overview"

    def get_tool_definition(self) -> ToolDefinition:
        """Get tool definition in LiteLLM/OpenAI format.

        Returns:
            Tool definition for LLM function calling
        """
        return ToolDefinition(
            function=ToolFunction(
                name=self.name,
                description=(
                    "Get overnight futures performance and market sentiment. "
                    "Provides S&P 500 and Nasdaq futures data with sentiment analysis "
                    "(BULLISH/BEARISH/NEUTRAL)."
                ),
                parameters=ToolParametersSchema(
                    properties={
                        "futures_symbols": ToolParameter(
                            type="string",
                            description="Comma-separated futures symbols (default: ES=F,NQ=F)",
                        ),
                    },
                    required=[],
                ),
            ),
        )

    def execute(self, **kwargs: str | int | float | bool) -> str:
        """Execute market overview.

        Args:
            **kwargs: Tool arguments (futures_symbols: str = "ES=F,NQ=F")

        Returns:
            Formatted market overview with sentiment
        """
        futures_symbols = str(kwargs.get("futures_symbols", "ES=F,NQ=F"))
        symbols = [s.strip() for s in futures_symbols.split(",")]

        logger.info(f"Fetching market overview for futures: {symbols}")

        try:
            futures_data = self._market_fetcher.fetch_overnight_futures(symbols)

            # Calculate sentiment
            changes = list(futures_data.values())
            avg_change = sum(changes) / len(changes) if changes else 0.0
            if avg_change > BULLISH_THRESHOLD:
                sentiment = "BULLISH"
            elif avg_change < BEARISH_THRESHOLD:
                sentiment = "BEARISH"
            else:
                sentiment = "NEUTRAL"

            # Format output
            lines = [
                "# Market Overview",
                "",
                f"**Overall Sentiment:** {sentiment}",
                "",
                "## Futures Performance",
            ]

            for symbol, change_pct in futures_data.items():
                sign = "+" if change_pct >= 0 else ""
                lines.append(f"- **{symbol}**: {sign}{change_pct:.2f}%")

            return "\n".join(lines)

        except Exception as e:
            logger.error(f"Market overview failed: {e}")
            return f"Failed to fetch market overview: {e}"

    def __repr__(self) -> str:
        """String representation."""
        return "MarketOverviewTool()"
