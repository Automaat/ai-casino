"""Trump analysis tool for analyzing Truth Social posts."""

import asyncio
import concurrent.futures
from collections.abc import Coroutine
from typing import TYPE_CHECKING, Any

from loguru import logger

from src.agents.trump import TrumpAnalysis
from src.tools.base import BaseTool

if TYPE_CHECKING:
    from src.di.container import AppContainer


def _run_async(coro: Coroutine[Any, Any, TrumpAnalysis]) -> TrumpAnalysis:
    """Run async coroutine, handling existing event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No running loop, safe to use asyncio.run()
        return asyncio.run(coro)

    # Already in an event loop, run in a thread pool
    with concurrent.futures.ThreadPoolExecutor() as pool:
        future = pool.submit(asyncio.run, coro)
        return future.result()


class TrumpAnalysisTool(BaseTool):
    """Tool to analyze Trump's recent Truth Social posts for trading signals."""

    TOOL_NAME = "analyze_trump_posts"

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
        return self.TOOL_NAME

    def get_tool_definition(self) -> dict:
        """Get tool definition for LLM function calling."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Analyze Trump's recent Truth Social posts for market-moving signals. "
                    "Detects tariff announcements, trade deals, crypto mentions, and company references. "
                    "Returns trading signal, affected tickers, and interpretation."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "days": {
                            "type": "integer",
                            "description": "Number of days to look back (default: 3, max: 7)",
                            "default": 3,
                        },
                    },
                    "required": [],
                },
            },
        }

    def execute(self, **kwargs: str | int | float | bool) -> str:
        """Analyze Trump's recent posts.

        Args:
            **kwargs: Tool arguments (days: int = 3)

        Returns:
            Formatted analysis result
        """
        days = int(kwargs.get("days", 3))
        days = min(max(days, 1), 7)  # Clamp to 1-7 days
        hours = days * 24

        logger.info(f"Analyzing Trump posts from last {days} days")

        try:
            from src.agents.trump import TrumpAnalyst

            fetcher = self._container.truth_social_fetcher()
            post_data = fetcher.fetch_recent(hours=hours)

            if not post_data.posts:
                return f"No Trump posts found in the last {days} days."

            llm = self._container.llm_client()
            analyst = TrumpAnalyst(llm)

            # Run async analysis (handles existing event loop)
            analysis = _run_async(analyst.analyze(post_data.posts))

            return self._format_analysis(analysis, days)
        except Exception as e:
            logger.error(f"Failed to analyze Trump posts: {e}")
            return f"Failed to analyze Trump posts: {e}"

    def _format_analysis(self, analysis: TrumpAnalysis, days: int) -> str:
        """Format analysis result."""
        lines = [
            f"# Trump Analysis (Last {days} Days)",
            "",
            f"**Posts Analyzed:** {analysis.post_count}",
            f"**Market Relevant:** {'Yes' if analysis.market_relevant else 'No'}",
            "",
            "## Trading Signal",
            f"- **Signal:** {analysis.signal.value}",
            f"- **Confidence:** {analysis.confidence:.0%}",
            f"- **Sentiment:** {analysis.sentiment}",
            "",
        ]

        if analysis.mentioned_tickers:
            lines.extend(
                [
                    "## Mentioned Tickers",
                    ", ".join(f"${t}" for t in analysis.mentioned_tickers),
                    "",
                ]
            )

        if analysis.key_phrases:
            lines.extend(["## Key Phrases"])
            for phrase in analysis.key_phrases[:5]:
                lines.append(f"- {phrase}")
            lines.append("")

        lines.extend(
            [
                "## Interpretation",
                analysis.interpretation,
            ]
        )

        return "\n".join(lines)

    def __repr__(self) -> str:
        """String representation."""
        return "TrumpAnalysisTool()"
