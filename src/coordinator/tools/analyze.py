"""Analyze symbol tool for coordinator."""

import asyncio
import concurrent.futures
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from loguru import logger
from pydantic import BaseModel

from src.tools.base import BaseTool
from src.tools.models import ToolDefinition, ToolFunction, ToolParameter, ToolParametersSchema
from src.workflows.types import WorkflowExtraContext

if TYPE_CHECKING:
    from src.coordinator.agent import TradingCoordinator
    from src.di.container import AppContainer
    from src.workflows.types import TradingWorkflowResult


class PositionContext(BaseModel):
    """Position context for analysis."""

    has_position: bool
    quantity: float
    avg_entry_price: float
    unrealized_pnl: float
    unrealized_pnl_percent: float


class AnalyzeSymbolTool(BaseTool):
    """Tool to run full trading analysis workflow."""

    def __init__(
        self,
        container: AppContainer,
        coordinator: TradingCoordinator | None = None,
    ) -> None:
        """Initialize tool with DI container.

        Args:
            container: DI container for workflow creation
            coordinator: Optional coordinator for result storage
        """
        self._container = container
        self._coordinator = coordinator

    @property
    def name(self) -> str:
        """Tool name."""
        return "analyze_symbol"

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
                    "This is an expensive operation that makes multiple API calls and LLM requests."
                ),
                parameters=ToolParametersSchema(
                    properties={
                        "symbol": ToolParameter(
                            type="string",
                            description="Stock ticker symbol (e.g., AAPL, TSLA, MSFT)",
                        ),
                        "period_days": ToolParameter(
                            type="integer",
                            description=(
                                "Number of days of historical data to analyze (default: 90, range: 30-365)"
                            ),
                        ),
                        "include_position_context": ToolParameter(
                            type="boolean",
                            description="Include existing position context in analysis (default: false)",
                        ),
                    },
                    required=["symbol"],
                ),
            ),
        )

    async def aexecute(self, **kwargs: str | int | float | bool) -> str:
        """Execute full trading analysis asynchronously.

        Overrides BaseTool.aexecute() to use native async instead of
        thread offloading, avoiding event loop conflicts with DB sessions.

        Args:
            **kwargs: Tool arguments (symbol: str, period_days: int = 90,
                     include_position_context: bool = False)

        Returns:
            Formatted analysis summary
        """
        symbol = str(kwargs["symbol"]).upper()
        period_days = int(kwargs.get("period_days", 90))
        include_position_context = bool(kwargs.get("include_position_context", False))

        logger.info(f"Running full analysis for {symbol} ({period_days} days)")

        try:
            return await self._run_analysis(symbol, period_days, include_position_context)
        except Exception as e:
            logger.opt(exception=True).error(f"Analysis failed for {symbol}: {e}")
            return f"Analysis failed for {symbol}: {e}"

    def execute(self, **kwargs: str | int | float | bool) -> str:
        """Execute full trading analysis (sync wrapper for compatibility).

        Args:
            **kwargs: Tool arguments (symbol: str, period_days: int = 90,
                     include_position_context: bool = False)

        Returns:
            Formatted analysis summary
        """
        # Sync wrapper - should not be called in production (use aexecute)
        logger.warning("AnalyzeSymbolTool.execute() called - prefer aexecute() to avoid event loop issues")
        symbol = str(kwargs["symbol"]).upper()
        period_days = int(kwargs.get("period_days", 90))
        include_position_context = bool(kwargs.get("include_position_context", False))

        def run_in_thread() -> str:
            return asyncio.run(self._run_analysis(symbol, period_days, include_position_context))

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_in_thread)
                return future.result()
        except Exception as e:
            logger.opt(exception=True).error(f"Analysis failed for {symbol}: {e}")
            return f"Analysis failed for {symbol}: {e}"

    async def _get_cooldown_symbols(self) -> list[str]:
        """Fetch symbols under re-entry cooldown from trade history.

        Returns:
            List of recently closed symbols in cooldown period
        """
        try:
            from src.daemon.config import PositionManagementConfig

            daemon_config = self._container.daemon_config()
            pos_config = daemon_config.position_management
            if not isinstance(pos_config, PositionManagementConfig):
                return []
            if not pos_config.whipsaw_prevention_enabled:
                return []

            cutoff = datetime.now(UTC) - timedelta(hours=pos_config.re_entry_cooldown_hours)
            repo = self._container.trade_repository()
            async with repo:
                closed_trades = await repo.get_closed_since(cutoff)
            return list({t.symbol for t in closed_trades})
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to fetch cooldown symbols: {e}")
            return []

    async def _run_analysis(
        self,
        symbol: str,
        period_days: int,
        include_position_context: bool,
    ) -> str:
        """Run analysis workflow asynchronously.

        Args:
            symbol: Stock ticker symbol
            period_days: Days of historical data
            include_position_context: Whether to include position context

        Returns:
            Formatted analysis summary
        """
        # Create workflow from container
        workflow = self._container.workflow_meta(container=self._container)

        # Get position context if requested
        position_ctx = None
        if include_position_context:
            pos_ctx = self._get_position_context(symbol)
            if pos_ctx:
                position_ctx = pos_ctx.model_dump()

        # Build extra context with cooldown symbols
        cooldown_symbols = await self._get_cooldown_symbols()
        extra_context = WorkflowExtraContext(
            position_context=position_ctx,
            re_entry_cooldown_symbols=cooldown_symbols,
        )

        # Run analysis
        result = await workflow.analyze(symbol, period_days, extra_context=extra_context)

        # Store structured result in coordinator for reflection tool access
        if self._coordinator:
            self._coordinator._last_analysis_results[symbol] = result  # noqa: SLF001

        return self._format_result(result)

    def _get_position_context(self, symbol: str) -> PositionContext | None:
        """Get position context for symbol.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Position context or None
        """
        try:
            broker = self._container.alpaca_broker()
            account_info = broker.get_account_info()

            if symbol in account_info.positions:
                pos = account_info.positions[symbol]
                return PositionContext(
                    has_position=True,
                    quantity=pos.qty,
                    avg_entry_price=pos.avg_entry_price,
                    unrealized_pnl=pos.unrealized_pnl,
                    unrealized_pnl_percent=pos.unrealized_pnl_percent,
                )

            return None
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to get position context: {e}")
            return None

    def _format_result(self, result: TradingWorkflowResult) -> str:
        """Format workflow result as markdown summary.

        Args:
            result: TradingWorkflowResult

        Returns:
            Formatted markdown string
        """
        # Reuse exact formatting from AnalyzeStockTool
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
            *(
                [
                    f"- Sentiment: {result.sentiment.overall_sentiment}",
                    f"- Score: {result.sentiment.sentiment_score:.2f}",
                    f"- Positive: {result.sentiment.positive_ratio:.0%} | "
                    f"Negative: {result.sentiment.negative_ratio:.0%} | "
                    f"Neutral: {result.sentiment.neutral_ratio:.0%}",
                ]
                if result.sentiment
                else ["- Sentiment: Unavailable"]
            ),
            "",
            "## News Analysis",
            *(
                [
                    f"- Key Themes: {', '.join(result.news.key_themes)}",
                    f"- Impact: {result.news.impact_assessment}",
                    f"- Recommendation: {result.news.recommendation}",
                ]
                if result.news
                else ["- News: Unavailable"]
            ),
            "",
            "## Decision Rationale",
            "\n".join(f"- {r}" for r in result.decision.reasoning),
        ]

        if result.warnings:
            lines.extend(["", "## Warnings", *[f"- {w}" for w in result.warnings]])

        return "\n".join(lines)

    def __repr__(self) -> str:
        """String representation."""
        return "AnalyzeSymbolTool()"
