"""Decision history query tool for coordinator learning."""

import asyncio
from typing import TYPE_CHECKING, Final, cast

from loguru import logger

from src.tools.base import BaseTool
from src.tools.models import ToolDefinition, ToolFunction, ToolParameter, ToolParametersSchema
from src.v1.coordinator.memory import DecisionQueryParams

if TYPE_CHECKING:
    from src.v1.coordinator.memory import CoordinatorMemory

# Constants
_MAX_DISPLAYED_DECISIONS: Final[int] = 30


class QueryPastDecisionsTool(BaseTool):
    """Tool to query past trading decisions with outcomes for learning from historical patterns."""

    def __init__(self, memory: CoordinatorMemory) -> None:
        """Initialize tool with coordinator memory.

        Args:
            memory: Coordinator memory instance with signal outcome repository
        """
        self._memory = memory

    @property
    def name(self) -> str:
        """Tool name."""
        return "query_past_decisions"

    @property
    def requires_confirmation(self) -> bool:
        """Read-only tool, no confirmation needed."""
        return False

    def get_tool_definition(self) -> ToolDefinition:
        """Get tool definition in LiteLLM/OpenAI format.

        Returns:
            Tool definition for LLM function calling
        """
        return ToolDefinition(
            function=ToolFunction(
                name=self.name,
                description=(
                    "Query past trading decisions with outcomes to learn from historical patterns. "
                    "Returns decisions with 5-day returns, classified as HIT/MISS. "
                    "Use to check success rate, identify failure patterns, "
                    "validate current analysis against history, and avoid repeating mistakes. "
                    "Includes regime and strategy context."
                ),
                parameters=ToolParametersSchema(
                    properties={
                        "symbol": ToolParameter(
                            type="string",
                            description="Stock ticker symbol (optional, omit to query all symbols)",
                        ),
                        "signal": ToolParameter(
                            type="string",
                            description="Signal type filter (BUY/SELL/HOLD, optional)",
                            enum=["BUY", "SELL", "HOLD"],
                        ),
                        "lookback_days": ToolParameter(
                            type="integer",
                            description="Days to look back (default: 90, max: 365)",
                            minimum=7,
                            maximum=365,
                        ),
                        "min_confidence": ToolParameter(
                            type="number",
                            description="Minimum confidence filter (0.0-1.0, optional)",
                            minimum=0.0,
                            maximum=1.0,
                        ),
                        "horizon": ToolParameter(
                            type="string",
                            description="Outcome horizon for return calculation (default: 5d)",
                            enum=["1d", "5d", "20d"],
                        ),
                    },
                    required=[],
                ),
            ),
        )

    def execute(self, **kwargs: str | int | float | bool) -> str:
        """Execute decision history query with outcome analysis.

        Args:
            **kwargs: Tool arguments
                - symbol: str (optional)
                - signal: str (optional, BUY/SELL/HOLD)
                - lookback_days: int (default: 90, max: 365)
                - min_confidence: float (optional, 0.0-1.0)
                - horizon: str (default: 5d, options: 1d/5d/20d)

        Returns:
            Formatted markdown table with decisions and statistics
        """
        # Extract and validate parameters
        symbol = kwargs.get("symbol")
        if symbol and not isinstance(symbol, str):
            symbol = str(symbol)

        signal = kwargs.get("signal")
        if signal and not isinstance(signal, str):
            signal = str(signal)

        lookback_days = max(7, min(int(kwargs.get("lookback_days", 90)), 365))
        min_confidence = kwargs.get("min_confidence")
        if min_confidence is not None:
            min_confidence = float(min_confidence)

        horizon = kwargs.get("horizon", "5d")
        if not isinstance(horizon, str):
            horizon = str(horizon)
        if horizon not in ["1d", "5d", "20d"]:
            horizon = "5d"

        logger.info(f"Querying past decisions: symbol={symbol}, signal={signal}, lookback={lookback_days}d")

        try:
            # Run async query via memory layer
            return asyncio.run(
                self._async_execute(
                    symbol=symbol,
                    signal=signal,
                    lookback_days=lookback_days,
                    min_confidence=min_confidence,
                    horizon=horizon,
                )
            )

        except Exception as e:
            logger.opt(exception=True).error(f"Decision query failed: {e}")
            return f"Failed to query past decisions: {e}"

    async def _async_execute(
        self,
        symbol: str | int | float | bool | None,
        signal: str | int | float | bool | None,
        lookback_days: int,
        min_confidence: float | None,
        horizon: str,
    ) -> str:
        """Async execution of decision query.

        Args:
            symbol: Optional symbol filter
            signal: Optional signal filter
            lookback_days: Days to look back
            min_confidence: Optional min confidence
            horizon: Outcome horizon

        Returns:
            Formatted markdown result
        """
        # Type narrow to str | None
        symbol_str = str(symbol) if symbol and not isinstance(symbol, bool) else None
        signal_str = str(signal) if signal and not isinstance(signal, bool) else None

        # Query decisions from memory (which delegates to signal_outcome_repo)
        params = DecisionQueryParams(
            symbol=symbol_str,
            signal=signal_str,
            lookback_days=lookback_days,
            min_confidence=min_confidence,
            limit=50,
            horizon=horizon,
        )
        decisions = await self._memory.query_decisions(params)

        # Get success rate statistics
        stats = await self._memory.get_success_rate(
            signal=signal_str,
            lookback_days=lookback_days,
            horizon=horizon,
        )

        # Format results as markdown
        return self._format_results(decisions, stats, symbol_str, lookback_days, horizon)

    def _format_results(
        self,
        decisions: list,
        stats: dict[str, object],
        symbol: str | None,
        lookback_days: int,
        horizon: str,
    ) -> str:
        """Format query results as markdown table with statistics.

        Args:
            decisions: List of DecisionQueryResult instances
            stats: SuccessRateStats dict
            symbol: Symbol filter (optional)
            lookback_days: Lookback period
            horizon: Outcome horizon

        Returns:
            Formatted markdown string
        """
        # Build header
        title_parts = ["## Past Trading Decisions"]
        if symbol:
            title_parts.append(f"for {symbol}")
        title_parts.append(f"(Last {lookback_days} Days, {horizon} Horizon)")
        title = " ".join(title_parts)

        if not decisions:
            return f"{title}\n\nNo decisions found matching filters."

        # Build summary stats (cast from object to expected types)
        success_rate_pct = cast("float", stats.get("success_rate", 0.0)) * 100
        total = cast("int", stats.get("total_decisions", 0))
        hit_count = cast("int", stats.get("hit_count", 0))
        miss_count = cast("int", stats.get("miss_count", 0))
        pending = cast("int", stats.get("pending_count", 0))
        avg_return_raw = stats.get("avg_return")
        avg_return = cast("float", avg_return_raw) if avg_return_raw is not None else None
        avg_confidence = cast("float", stats.get("avg_confidence", 0.0)) * 100

        summary_lines = [
            "**Summary Statistics:**",
            f"- Success Rate: **{success_rate_pct:.1f}%** "
            f"({hit_count} hits / {miss_count} misses / {pending} pending)",
            f"- Total Decisions: {total}",
            f"- Average Confidence: {avg_confidence:.0f}%",
        ]

        if avg_return is not None:
            summary_lines.append(f"- Average Return ({horizon}): **{avg_return:+.2f}%**")

        summary = "\n".join(summary_lines)

        # Build table
        table_lines = [
            "",
            "| Date | Symbol | Signal | Conf | Entry | Outcome | Return | Result |",
            "|------|--------|--------|------|-------|---------|--------|--------|",
        ]

        for decision in decisions[:_MAX_DISPLAYED_DECISIONS]:
            date_str = decision.timestamp.strftime("%m/%d")
            conf_str = f"{decision.confidence * 100:.0f}%"
            entry_str = f"${decision.price_at_signal:.2f}"

            if decision.price_at_outcome is not None:
                outcome_str = f"${decision.price_at_outcome:.2f}"
            else:
                outcome_str = "—"

            return_str = f"{decision.return_pct:+.1f}%" if decision.return_pct is not None else "—"

            # Format result with emoji
            if decision.hit_miss == "HIT":
                result_str = "✅ HIT"
            elif decision.hit_miss == "MISS":
                result_str = "❌ MISS"
            else:
                result_str = "⏳ PENDING"

            table_lines.append(
                f"| {date_str} | {decision.symbol} | {decision.signal} | {conf_str} | "
                f"{entry_str} | {outcome_str} | {return_str} | {result_str} |"
            )

        if len(decisions) > _MAX_DISPLAYED_DECISIONS:
            table_lines.append(f"\n*Showing first {_MAX_DISPLAYED_DECISIONS} of {len(decisions)} decisions*")

        table = "\n".join(table_lines)

        return f"{title}\n\n{summary}\n{table}"

    def __repr__(self) -> str:
        """String representation."""
        return "QueryPastDecisionsTool()"
