"""News tool for fetching stock news."""

from typing import TYPE_CHECKING

from loguru import logger

from src.tools.base import BaseTool
from src.tools.models import ToolDefinition, ToolFunction, ToolParameter, ToolParametersSchema

if TYPE_CHECKING:
    from src.di.container import AppContainer

DESCRIPTION_TRUNCATE_LENGTH = 300


class GetNewsTool(BaseTool):
    """Tool to fetch recent news for a stock."""

    def __init__(self, container: AppContainer | None = None) -> None:
        """Initialize tool with optional container.

        Args:
            container: DI container (auto-created if not provided)
        """
        from src.di.container import create_container

        self._container = container or create_container()

    @property
    def name(self) -> str:
        """Tool name."""
        return "get_news"

    def get_tool_definition(self) -> ToolDefinition:
        """Get tool definition in LiteLLM/OpenAI format.

        Returns:
            Tool definition for LLM function calling
        """
        return ToolDefinition(
            function=ToolFunction(
                name=self.name,
                description=(
                    "Get recent news articles for a stock. Returns headlines, sources, and dates. "
                    "Use this to understand recent events and news sentiment for a company."
                ),
                parameters=ToolParametersSchema(
                    properties={
                        "symbol": ToolParameter(
                            type="string",
                            description="Stock ticker symbol (e.g., AAPL, TSLA, MSFT)",
                        ),
                        "limit": ToolParameter(
                            type="integer",
                            description="Maximum number of articles to return (default: 5)",
                        ),
                    },
                    required=["symbol"],
                ),
            ),
        )

    def execute(self, **kwargs: str | int | float | bool) -> str:
        """Fetch news for a stock.

        Args:
            **kwargs: Tool arguments (symbol: str, limit: int = 5)

        Returns:
            Formatted news summary
        """
        symbol = str(kwargs["symbol"])
        limit = int(kwargs.get("limit", 5))

        logger.info(f"Fetching news for {symbol} (limit={limit})")

        try:
            fetcher = self._container.news_fetcher()
            articles = fetcher.fetch_company_news(symbol.upper(), limit=limit)

            return self._format_articles(symbol.upper(), articles)
        except Exception as e:
            logger.error(f"Failed to fetch news for {symbol}: {e}")
            return f"Failed to fetch news for {symbol}: {e}"

    def _format_articles(self, symbol: str, articles: list) -> str:
        """Format news articles as summary.

        Args:
            symbol: Stock ticker
            articles: List of NewsArticle objects

        Returns:
            Formatted summary string
        """
        if not articles:
            return f"No recent news found for {symbol}"

        lines = [f"# {symbol} Recent News", ""]

        for i, article in enumerate(articles, 1):
            pub_date = article.published_at.strftime("%Y-%m-%d %H:%M")
            lines.extend(
                [
                    f"## {i}. {article.title}",
                    f"*{article.source} | {pub_date}*",
                    "",
                    (
                        article.description[:DESCRIPTION_TRUNCATE_LENGTH] + "..."
                        if len(article.description) > DESCRIPTION_TRUNCATE_LENGTH
                        else article.description
                    ),
                    "",
                ]
            )

        return "\n".join(lines)

    def __repr__(self) -> str:
        """String representation."""
        return "GetNewsTool()"
