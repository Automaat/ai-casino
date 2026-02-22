"""Social sentiment tool for Reddit/Finnhub analysis."""

import asyncio
import concurrent.futures
from typing import TYPE_CHECKING

from loguru import logger

from src.tools.base import BaseTool
from src.tools.models import ToolDefinition, ToolFunction, ToolParameter, ToolParametersSchema

if TYPE_CHECKING:
    from src.agents.social import SocialSentimentAnalysis
    from src.di.container import AppContainer


class GetSocialSentimentTool(BaseTool):
    """Tool to analyze social sentiment from Reddit and Finnhub."""

    def __init__(self, container: AppContainer) -> None:
        """Initialize tool with DI container.

        Args:
            container: DI container for dependency resolution
        """
        self._container = container

    @property
    def name(self) -> str:
        """Tool name."""
        return "get_social_sentiment"

    def get_tool_definition(self) -> ToolDefinition:
        """Get tool definition in LiteLLM/OpenAI format.

        Returns:
            Tool definition for LLM function calling
        """
        return ToolDefinition(
            function=ToolFunction(
                name=self.name,
                description=(
                    "Analyze social sentiment for a stock from Reddit (WSB, r/stocks, r/investing) "
                    "and Finnhub social data. Returns social score, momentum, WSB mentions, "
                    "and sentiment breakdown."
                ),
                parameters=ToolParametersSchema(
                    properties={
                        "symbol": ToolParameter(
                            type="string",
                            description="Stock ticker symbol (e.g., AAPL, TSLA, MSFT)",
                        ),
                    },
                    required=["symbol"],
                ),
            ),
        )

    def execute(self, **kwargs: str | int | float | bool) -> str:
        """Analyze social sentiment for a stock.

        Args:
            **kwargs: Tool arguments (symbol: str)

        Returns:
            Formatted social sentiment summary
        """
        symbol = str(kwargs["symbol"])

        logger.info(f"Analyzing social sentiment for {symbol}")

        def run_in_thread() -> str:
            return asyncio.run(self._run_analysis(symbol.upper()))

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_in_thread)
                return future.result()
        except Exception as e:
            logger.opt(exception=True).error(f"Social sentiment analysis failed for {symbol}: {e}")
            return f"Social sentiment analysis failed for {symbol}: {e}"

    async def _run_analysis(self, symbol: str) -> str:
        """Run social sentiment analysis asynchronously.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Formatted analysis summary
        """
        worker = self._container.social_sentiment_worker()
        result = await worker.analyze(symbol)

        return self._format_result(symbol, result)

    def _format_result(self, symbol: str, result: SocialSentimentAnalysis) -> str:
        """Format social sentiment result as markdown.

        Args:
            symbol: Stock ticker symbol
            result: SocialSentimentAnalysis result

        Returns:
            Formatted markdown string
        """
        lines = [
            f"# {symbol} Social Sentiment",
            "",
            f"**Sentiment:** {result.sentiment_label}",
            f"**Social Score:** {result.overall_social_score:.2f} (-1 to 1)",
            f"**Momentum:** {result.social_momentum}",
            f"**Confidence:** {result.confidence:.0%}",
            "",
            "## Source Breakdown",
        ]

        if result.finnhub_sentiment is not None:
            lines.append(f"- Finnhub: {result.finnhub_sentiment:.2f}")
        else:
            lines.append("- Finnhub: N/A")

        if result.reddit_sentiment is not None:
            lines.append(f"- Reddit: {result.reddit_sentiment:.2f}")
        else:
            lines.append("- Reddit: N/A")

        lines.append(f"- WSB Mentions (24h): {result.wsb_mentions_24h}")

        if result.apewisdom_rank is not None:
            delta_str = (
                f", {result.apewisdom_mention_delta_pct:+.0f}% vs 24h"
                if result.apewisdom_mention_delta_pct is not None
                else ""
            )
            lines.append(
                f"- ApeWisdom: rank #{result.apewisdom_rank}, {result.apewisdom_mentions} mentions{delta_str}"
            )
        else:
            lines.append("- ApeWisdom: Not in trending")

        lines.extend(
            [
                "",
                "## Interpretation",
                result.interpretation,
            ]
        )

        return "\n".join(lines)

    def __repr__(self) -> str:
        """String representation."""
        return "GetSocialSentimentTool()"
