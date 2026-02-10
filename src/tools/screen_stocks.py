"""Screen stocks tool for agentic stock discovery."""

import asyncio
import concurrent.futures
from typing import TYPE_CHECKING

from loguru import logger

from src.tools.base import BaseTool

if TYPE_CHECKING:
    from src.di.container import AppContainer
    from src.screening.analyzer import ScreeningAnalysis
    from src.screening.screener import ScreeningOutput


class ScreenStocksTool(BaseTool):
    """Tool to screen stocks for investment opportunities."""

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
        return "screen_stocks"

    @property
    def requires_confirmation(self) -> bool:
        """Requires confirmation due to expensive operations."""
        return True

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
                    "Screen stocks for investment opportunities using technical criteria. "
                    "Supports momentum (oversold with bullish reversal), value (low P/E and P/B), "
                    "and breakout (near 52-week high with volume) strategies. "
                    "Returns top matching stocks with scores and LLM analysis. "
                    "This is an expensive operation that fetches data for many stocks."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "criteria": {
                            "type": "string",
                            "enum": ["momentum", "value", "breakout"],
                            "description": (
                                "Screening criteria: "
                                "momentum (RSI oversold + MACD bullish), "
                                "value (low P/E + P/B), "
                                "breakout (near 52-week high + volume spike)"
                            ),
                        },
                        "universe": {
                            "type": "string",
                            "enum": ["SP500", "NASDAQ100", "COMBINED"],
                            "description": "Stock universe to screen (default: COMBINED)",
                            "default": "COMBINED",
                        },
                        "top_n": {
                            "type": "integer",
                            "description": "Number of top results to return (default: 10)",
                            "default": 10,
                            "minimum": 1,
                            "maximum": 50,
                        },
                    },
                    "required": ["criteria"],
                },
            },
        }

    def execute(self, **kwargs: str | int | float | bool) -> str:
        """Execute stock screening.

        Args:
            **kwargs: Tool arguments (criteria: str, universe: str = "COMBINED", top_n: int = 10)

        Returns:
            Formatted screening results with analysis
        """
        criteria = str(kwargs["criteria"])
        universe = str(kwargs.get("universe", "COMBINED"))
        top_n = int(kwargs.get("top_n", 10))

        logger.info(f"Screening {universe} for {criteria} (top {top_n})")

        def run_in_thread() -> str:
            return asyncio.run(self._run_screening(criteria, universe, top_n))

        try:
            # Run in thread to avoid nested event loop issues
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_in_thread)
                return future.result()
        except Exception as e:
            logger.error(f"Screening failed: {e}")
            return f"Screening failed: {e}"

    async def _run_screening(self, criteria: str, universe: str, top_n: int) -> str:
        """Run screening workflow asynchronously.

        Args:
            criteria: Screening criteria
            universe: Stock universe
            top_n: Number of top results

        Returns:
            Formatted screening results
        """
        from src.screening.analyzer import ScreeningAnalyzer
        from src.screening.screener import ScreeningCriteria

        screener = self._container.stock_screener()
        llm = self._container.llm_client()
        analyzer = ScreeningAnalyzer(llm_client=llm)

        screening_criteria = ScreeningCriteria(criteria)
        output = screener.screen(criteria=screening_criteria, universe=universe, top_n=top_n)

        if not output.results:
            return (
                f"No stocks matched {criteria} criteria in {universe}. "
                f"Screened {output.total_screened} stocks."
            )

        analysis = await analyzer.analyze(output)

        return self._format_output(output, analysis)

    def _format_output(self, output: "ScreeningOutput", analysis: "ScreeningAnalysis") -> str:
        """Format screening output as markdown.

        Args:
            output: ScreeningOutput from screener
            analysis: ScreeningAnalysis from analyzer

        Returns:
            Formatted markdown string
        """
        lines = [
            f"# {output.criteria.value.title()} Screening Results",
            f"**Universe:** {output.universe} | **Screened:** {output.total_screened} stocks",
            "",
            "## Analysis",
            analysis.summary,
            "",
            "### Top Picks",
        ]

        for pick in analysis.top_picks:
            lines.append(f"- {pick}")

        lines.extend(
            [
                "",
                f"**Sector Insights:** {analysis.sector_insights}",
                "",
                f"**Risk Factors:** {analysis.risk_factors}",
                "",
                f"**Next Steps:** {analysis.next_steps}",
                "",
                "## Screening Results",
                "",
            ]
        )

        for i, result in enumerate(output.results, 1):
            metrics_str = ", ".join(f"{k}={v}" for k, v in result.metrics.items())
            lines.extend(
                [
                    f"### {i}. {result.symbol} - {result.name}",
                    f"**Sector:** {result.sector} | **Score:** {result.score:.2f} | "
                    f"**Signal:** {result.signal.value}",
                    f"**Metrics:** {metrics_str}",
                    f"**Reason:** {result.reason}",
                    "",
                ]
            )

        if output.errors:
            lines.append(f"*Note: {len(output.errors)} symbols failed to screen.*")

        return "\n".join(lines)

    def __repr__(self) -> str:
        """String representation."""
        return "ScreenStocksTool()"
