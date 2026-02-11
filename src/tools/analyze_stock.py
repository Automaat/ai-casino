"""Analyze stock tool for full trading workflow."""

import asyncio
import concurrent.futures
from typing import TYPE_CHECKING

from loguru import logger

from src.tools.base import BaseTool
from src.tools.models import ToolDefinition, ToolFunction, ToolParameter, ToolParametersSchema

if TYPE_CHECKING:
    from src.di.container import AppContainer
    from src.workflows.types import TradingWorkflowResult


class AnalyzeStockTool(BaseTool):
    """Tool to run full trading analysis workflow."""

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
        return "analyze_stock"

    @property
    def requires_confirmation(self) -> bool:
        """Requires confirmation due to expensive LLM calls."""
        return True

    def get_tool_definition(self) -> ToolDefinition:
        """Get tool definition in LiteLLM/OpenAI format.

        Returns:
            Tool definition for LLM function calling
        """
        return ToolDefinition(
            function=ToolFunction(
                name=self.name,
                description=(
                    "Run comprehensive trading analysis on a stock. Includes technical analysis "
                    "(RSI, MACD), sentiment analysis (FinBERT), news analysis, fundamental analysis, "
                    "and generates a trading recommendation (BUY/SELL/HOLD) with confidence score. "
                    "This is an expensive operation that makes multiple API calls."
                ),
                parameters=ToolParametersSchema(
                    properties={
                        "symbol": ToolParameter(
                            type="string",
                            description="Stock ticker symbol (e.g., AAPL, TSLA, MSFT)",
                        ),
                        "period_days": ToolParameter(
                            type="integer",
                            description="Number of days of historical data to analyze (default: 90)",
                        ),
                    },
                    required=["symbol"],
                ),
            ),
        )

    def execute(self, **kwargs: str | int | float | bool) -> str:
        """Execute full trading analysis.

        Args:
            **kwargs: Tool arguments (symbol: str, period_days: int = 90)

        Returns:
            Formatted analysis summary
        """
        symbol = str(kwargs["symbol"])
        period_days = int(kwargs.get("period_days", 90))

        logger.info(f"Running full analysis for {symbol} ({period_days} days)")

        def run_in_thread() -> str:
            return asyncio.run(self._run_analysis(symbol.upper(), period_days))

        try:
            # Run in thread to avoid nested event loop issues with Python 3.14
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_in_thread)
                return future.result()
        except Exception as e:
            logger.error(f"Analysis failed for {symbol}: {e}")
            return f"Analysis failed for {symbol}: {e}"

    async def _run_analysis(self, symbol: str, period_days: int) -> str:
        """Run analysis workflow asynchronously.

        Args:
            symbol: Stock ticker symbol
            period_days: Days of historical data

        Returns:
            Formatted analysis summary
        """
        # Use workflow from container (momentum strategy by default)
        workflow = self._container.workflow_momentum(container=self._container)

        result = await workflow.analyze(symbol, period_days)

        return self._format_result(result)

    def _format_result(self, result: TradingWorkflowResult) -> str:
        """Format workflow result as markdown summary.

        Args:
            result: TradingWorkflowResult

        Returns:
            Formatted markdown string
        """
        lines = [
            f"# {result.symbol} Trading Analysis",
            "",
            f"**Recommendation:** {result.decision.action.value}",
            f"**Confidence:** {result.decision.confidence:.0%}",
            f"**Risk Level:** {result.risk.validation.risk_level}",
            "",
            "## Technical Analysis",
            f"- Signal: {result.technical.signal.value}",
            f"- RSI: {result.technical.rsi:.1f}",
            f"- MACD Histogram: {result.technical.macd_hist:.4f}",
            f"- Interpretation: {result.technical.interpretation}",
            "",
            "## Sentiment Analysis",
            f"- Sentiment: {result.sentiment.overall_sentiment}",
            f"- Score: {result.sentiment.sentiment_score:.2f}",
            f"- Positive: {result.sentiment.positive_ratio:.0%} | "
            f"Negative: {result.sentiment.negative_ratio:.0%} | "
            f"Neutral: {result.sentiment.neutral_ratio:.0%}",
            "",
            "## News Analysis",
            f"- Key Themes: {', '.join(result.news.key_themes)}",
            f"- Impact: {result.news.impact_assessment}",
            f"- Recommendation: {result.news.recommendation}",
            "",
            "## Decision Rationale",
            "\n".join(f"- {r}" for r in result.decision.reasoning),
        ]

        if result.warnings:
            lines.extend(["", "## Warnings", *[f"- {w}" for w in result.warnings]])

        return "\n".join(lines)

    def __repr__(self) -> str:
        """String representation."""
        return "AnalyzeStockTool()"
