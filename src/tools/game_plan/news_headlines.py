"""Fetch news headlines tool for game plan agent."""

from loguru import logger

from src.data.news import NewsFetcher
from src.tools.base import BaseTool
from src.tools.models import ToolDefinition, ToolFunction, ToolParameter, ToolParametersSchema


class FetchNewsHeadlinesTool(BaseTool):
    """Fetch market and company news headlines."""

    def __init__(self, news_fetcher: NewsFetcher) -> None:
        """Initialize with news fetcher.

        Args:
            news_fetcher: News fetcher for headlines
        """
        self._fetcher = news_fetcher

    @property
    def name(self) -> str:
        """Tool name."""
        return "fetch_news_headlines"

    def get_tool_definition(self) -> ToolDefinition:
        """Get tool definition for LLM.

        Returns:
            Tool definition
        """
        return ToolDefinition(
            function=ToolFunction(
                name=self.name,
                description=(
                    "Fetch recent market news headlines and optionally company-specific news. "
                    "Useful for identifying catalysts and sentiment drivers."
                ),
                parameters=ToolParametersSchema(
                    properties={
                        "symbol": ToolParameter(
                            type="string",
                            description="Optional: specific stock symbol to get news for",
                        ),
                    },
                    required=[],
                ),
            ),
        )

    def execute(self, **_kwargs: str | int | float | bool) -> str:
        """Not supported — tool is async-only.

        Raises:
            RuntimeError: Always — use aexecute() instead
        """
        msg = (
            "FetchNewsHeadlinesTool.execute() cannot be called from a running "
            "event loop. Use 'aexecute' instead."
        )
        raise RuntimeError(msg)

    async def aexecute(self, **kwargs: str | int | float | bool) -> str:
        """Fetch news headlines asynchronously.

        Args:
            **kwargs: Optional symbol parameter

        Returns:
            Formatted news headlines
        """
        symbol = str(kwargs["symbol"]).strip().upper() if "symbol" in kwargs else None

        lines = ["## News Headlines"]

        try:
            market_news = await self._fetcher.afetch_market_news(limit=10)
            if market_news:
                lines.append("\n### Market News")
                for article in market_news[:5]:
                    lines.append(f"- [{article.source}] {article.title}")
        except Exception as e:
            logger.opt(exception=True).warning(f"Market news fetch failed: {e}")
            lines.append("Market news unavailable")

        if symbol:
            try:
                company_news = await self._fetcher.afetch_company_news(symbol, limit=5)
                if company_news:
                    lines.append(f"\n### {symbol} News")
                    for article in company_news[:3]:
                        lines.append(f"- [{article.source}] {article.title}")
                else:
                    lines.append(f"\nNo recent news for {symbol}")
            except Exception as e:
                logger.opt(exception=True).warning(f"Company news fetch failed for {symbol}: {e}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        """String representation."""
        return "FetchNewsHeadlinesTool()"
