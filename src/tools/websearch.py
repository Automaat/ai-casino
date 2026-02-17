"""Web search tool for LLM function calling."""

from typing import TYPE_CHECKING

from loguru import logger

from src.data.websearch import SearchType
from src.tools.base import BaseTool
from src.tools.models import ToolDefinition, ToolFunction, ToolParameter, ToolParametersSchema

if TYPE_CHECKING:
    from src.data.websearch import WebSearchFetcher

BODY_TRUNCATE_LENGTH = 300


class WebSearchTool(BaseTool):
    """Web search tool wrapper for LLM agents."""

    TOOL_NAME = "web_search"

    def __init__(self, fetcher: WebSearchFetcher) -> None:
        """Initialize web search tool with websearch fetcher.

        Args:
            fetcher: WebSearchFetcher for executing searches
        """
        self.fetcher = fetcher
        logger.info("Initialized WebSearchTool")

    @property
    def name(self) -> str:
        """Tool name."""
        return "web_search"

    def get_tool_definition(self) -> ToolDefinition:
        """Get tool definition in LiteLLM/OpenAI format.

        Returns:
            Tool definition for LLM function calling
        """
        return ToolDefinition(
            function=ToolFunction(
                name=self.name,
                description=(
                    "Search the web for information. Use 'news' search_type for recent news "
                    "and events, 'general' for broader information and company details."
                ),
                parameters=ToolParametersSchema(
                    properties={
                        "query": ToolParameter(
                            type="string",
                            description="The search query",
                        ),
                        "search_type": ToolParameter(
                            type="string",
                            enum=["general", "news"],
                            description="Type of search: 'general' for broad info, 'news' for recent news",
                        ),
                    },
                    required=["query"],
                ),
            ),
        )

    def execute(self, **kwargs: str | int | float | bool) -> str:
        """Execute web search and return formatted results.

        Args:
            **kwargs: Tool arguments (query: str, search_type: str = "general", max_results: int = 5)

        Returns:
            Formatted string with search results
        """
        query = str(kwargs["query"])
        search_type = str(kwargs.get("search_type", "general"))
        max_results = int(kwargs.get("max_results", 5))

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
