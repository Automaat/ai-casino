"""Web search tool for LLM function calling."""

from loguru import logger

from src.data.websearch import SearchType, WebSearchFetcher
from src.tools.base import BaseTool

BODY_TRUNCATE_LENGTH = 300


class WebSearchTool(BaseTool):
    """Web search tool wrapper for LLM agents."""

    def __init__(self, fetcher: WebSearchFetcher | None = None) -> None:
        """Initialize web search tool.

        Args:
            fetcher: WebSearchFetcher instance. Creates default if not provided.
        """
        self.fetcher = fetcher or WebSearchFetcher()
        logger.info("Initialized WebSearchTool")

    @property
    def name(self) -> str:
        """Tool name."""
        return "web_search"

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
                    "Search the web for information. Use 'news' search_type for recent news "
                    "and events, 'general' for broader information and company details."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query",
                        },
                        "search_type": {
                            "type": "string",
                            "enum": ["general", "news"],
                            "description": "Type of search: 'general' for broad info, 'news' for recent news",
                        },
                    },
                    "required": ["query", "search_type"],
                },
            },
        }

    def execute(self, query: str, search_type: str = "general", max_results: int = 5) -> str:
        """Execute web search and return formatted results.

        Args:
            query: Search query
            search_type: "general" or "news"
            max_results: Maximum results to return

        Returns:
            Formatted string with search results
        """
        logger.info(f"Executing web search: query='{query}', type={search_type}")

        search_type_enum = SearchType.NEWS if search_type == "news" else SearchType.GENERAL

        if search_type_enum == SearchType.NEWS:
            response = self.fetcher.search_news(query, max_results=max_results)
        else:
            response = self.fetcher.search(query, max_results=max_results)

        if not response.results:
            return f"No results found for: {query}"

        formatted = [f"Search results for '{query}' ({search_type}):"]
        for i, result in enumerate(response.results, 1):
            formatted.append(f"\n{i}. {result.title}")
            formatted.append(f"   URL: {result.url}")
            if len(result.body) > BODY_TRUNCATE_LENGTH:
                formatted.append(f"   {result.body[:BODY_TRUNCATE_LENGTH]}...")
            else:
                formatted.append(f"   {result.body}")
            if result.source:
                formatted.append(f"   Source: {result.source}")

        return "\n".join(formatted)

    def __repr__(self) -> str:
        """String representation."""
        return f"WebSearchTool(fetcher={self.fetcher})"
