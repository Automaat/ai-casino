"""Market data tool for fetching stock prices."""

from typing import TYPE_CHECKING

from loguru import logger

from src.tools.base import BaseTool

if TYPE_CHECKING:
    from src.data.market import MarketData
    from src.di.container import AppContainer


class GetMarketDataTool(BaseTool):
    """Tool to fetch current market data for a stock."""

    def __init__(self, container: "AppContainer | None" = None) -> None:
        """Initialize tool with optional container.

        Args:
            container: DI container (auto-created if not provided)
        """
        from src.di.container import create_container

        self._container = container or create_container()

    @property
    def name(self) -> str:
        """Tool name."""
        return "get_market_data"

    def get_tool_definition(self) -> dict:
        """Get tool definition in LiteLLM/OpenAI format.

        Returns:
            Tool definition dict for LLM function calling
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Get current market data for a stock including price, volume, and recent performance. "
                    "Use this to check current stock prices and basic market metrics."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock ticker symbol (e.g., AAPL, TSLA, MSFT)",
                        },
                        "days": {
                            "type": "integer",
                            "description": "Number of days of historical data (default: 30)",
                            "default": 30,
                        },
                    },
                    "required": ["symbol"],
                },
            },
        }

    def execute(self, **kwargs: str | int | float | bool) -> str:
        """Fetch market data for a stock.

        Args:
            **kwargs: Tool arguments (symbol: str, days: int = 30)

        Returns:
            Formatted market data summary
        """
        symbol = str(kwargs["symbol"])
        days = int(kwargs.get("days", 30))

        logger.info(f"Fetching market data for {symbol} ({days} days)")

        try:
            fetcher = self._container.market_fetcher()
            data = fetcher.fetch_daily(symbol.upper(), days)

            return self._format_data(data)
        except Exception as e:
            logger.error(f"Failed to fetch market data for {symbol}: {e}")
            return f"Failed to fetch market data for {symbol}: {e}"

    def _format_data(self, data: "MarketData") -> str:
        """Format market data as summary.

        Args:
            data: MarketData object

        Returns:
            Formatted summary string
        """
        df = data.data
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest

        price = latest["Close"]
        prev_price = prev["Close"]
        change = price - prev_price
        change_pct = (change / prev_price) * 100

        high_52w = df["High"].max()
        low_52w = df["Low"].min()
        avg_volume = df["Volume"].mean()

        lines = [
            f"# {data.symbol} Market Data",
            "",
            f"**Current Price:** ${price:.2f}",
            f"**Change:** ${change:+.2f} ({change_pct:+.2f}%)",
            "",
            "## Today's Range",
            f"- Open: ${latest['Open']:.2f}",
            f"- High: ${latest['High']:.2f}",
            f"- Low: ${latest['Low']:.2f}",
            f"- Volume: {latest['Volume']:,.0f}",
            "",
            f"## {len(df)}-Day Summary",
            f"- High: ${high_52w:.2f}",
            f"- Low: ${low_52w:.2f}",
            f"- Avg Volume: {avg_volume:,.0f}",
            "",
            f"*Last updated: {data.last_updated.strftime('%Y-%m-%d %H:%M')}*",
        ]

        return "\n".join(lines)

    def __repr__(self) -> str:
        """String representation."""
        return "GetMarketDataTool()"
