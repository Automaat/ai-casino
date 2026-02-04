"""Analyze stock tool for full trading workflow."""

import asyncio
from typing import TYPE_CHECKING

from loguru import logger

from src.tools.base import BaseTool

if TYPE_CHECKING:
    from src.workflows.trading import TradingWorkflowResult


class AnalyzeStockTool(BaseTool):
    """Tool to run full trading analysis workflow."""

    @property
    def name(self) -> str:
        """Tool name."""
        return "analyze_stock"

    @property
    def requires_confirmation(self) -> bool:
        """Requires confirmation due to expensive LLM calls."""
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
                    "Run comprehensive trading analysis on a stock. Includes technical analysis "
                    "(RSI, MACD), sentiment analysis (FinBERT), news analysis, fundamental analysis, "
                    "and generates a trading recommendation (BUY/SELL/HOLD) with confidence score. "
                    "This is an expensive operation that makes multiple API calls."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock ticker symbol (e.g., AAPL, TSLA, MSFT)",
                        },
                        "period_days": {
                            "type": "integer",
                            "description": "Number of days of historical data to analyze (default: 90)",
                            "default": 90,
                        },
                    },
                    "required": ["symbol"],
                },
            },
        }

    def execute(self, symbol: str, period_days: int = 90) -> str:
        """Execute full trading analysis.

        Args:
            symbol: Stock ticker symbol
            period_days: Days of historical data

        Returns:
            Formatted analysis summary
        """
        logger.info(f"Running full analysis for {symbol} ({period_days} days)")

        try:
            return asyncio.run(self._run_analysis(symbol.upper(), period_days))
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
        from src.data.fundamental import FundamentalDataFetcher
        from src.data.market import MarketDataFetcher
        from src.data.news import NewsFetcher
        from src.models.llm import LLMClient
        from src.models.sentiment import FinBERTSentiment
        from src.workflows.trading import TradingWorkflow

        llm = LLMClient()
        market_fetcher = MarketDataFetcher()
        news_fetcher = NewsFetcher()
        finbert = FinBERTSentiment()
        fundamental_fetcher = FundamentalDataFetcher()

        workflow = TradingWorkflow(
            llm_client=llm,
            market_fetcher=market_fetcher,
            news_fetcher=news_fetcher,
            finbert=finbert,
            fundamental_fetcher=fundamental_fetcher,
        )

        result = await workflow.analyze(symbol, period_days)

        return self._format_result(result)

    def _format_result(self, result: "TradingWorkflowResult") -> str:
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
            f"- Sentiment: {result.sentiment.sentiment}",
            f"- Score: {result.sentiment.score:.2f}",
            f"- Confidence: {result.sentiment.confidence:.0%}",
            "",
            "## News Analysis",
            f"- Overall Sentiment: {result.news.overall_sentiment}",
            f"- Key Themes: {', '.join(result.news.key_themes)}",
            "",
            "## Decision Rationale",
            result.decision.rationale,
        ]

        if result.warnings:
            lines.extend(["", "## Warnings", *[f"- {w}" for w in result.warnings]])

        return "\n".join(lines)

    def __repr__(self) -> str:
        """String representation."""
        return "AnalyzeStockTool()"
