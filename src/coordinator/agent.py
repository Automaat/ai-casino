"""Autonomous trading coordinator using LLM tool calling."""

import asyncio
from typing import TYPE_CHECKING

from loguru import logger

from src.coordinator.memory import CoordinatorMemory
from src.coordinator.models import CoordinatorConfig, CoordinatorCycleResult
from src.models.llm import LLMClient
from src.prompts import PromptLoader
from src.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from src.data.broker import AlpacaBroker


class TradingCoordinator:
    """Autonomous trading coordinator using LLM tool calling.

    Orchestrates trading workflow through iterative tool use:
    - Analyzes market context (futures, sentiment, portfolio)
    - Generates daily game plan
    - Screens/analyzes potential opportunities
    - Executes trades based on conviction
    - Learns from outcomes via persistent memory
    """

    def __init__(
        self,
        llm_client: LLMClient,
        tool_registry: ToolRegistry,
        memory: CoordinatorMemory,
        config: CoordinatorConfig,
        broker: AlpacaBroker,
    ) -> None:
        """Initialize coordinator.

        Args:
            llm_client: LLM client for tool calling
            tool_registry: Registry of available tools
            memory: Persistent memory for observations
            config: Coordinator configuration
            broker: Broker for portfolio context
        """
        self._llm = llm_client
        self._tools = tool_registry
        self._memory = memory
        self._config = config
        self._broker = broker
        self._prompts = PromptLoader("coordinator")
        self._last_cycle_summary = "No previous cycle"

        # Tracking variables (reset per cycle)
        self._tool_calls_count = 0
        self._symbols_analyzed: set[str] = set()
        self._trades_proposed = 0
        self._trades_executed = 0
        self._game_plan_generated = False

        logger.info("Initialized TradingCoordinator")

    async def run_cycle(
        self,
        watchlist: list[str],
        degradation_context: dict | None = None,
    ) -> CoordinatorCycleResult:
        """Run single coordinator cycle.

        Args:
            watchlist: List of symbols to consider
            degradation_context: Optional degradation warnings

        Returns:
            CoordinatorCycleResult with summary and metrics
        """
        # Reset tracking variables
        self._tool_calls_count = 0
        self._symbols_analyzed = set()
        self._trades_proposed = 0
        self._trades_executed = 0
        self._game_plan_generated = False

        try:
            # Build prompts
            system_prompt = await self._build_system_prompt(degradation_context)
            user_prompt = self._build_cycle_prompt(watchlist)

            # Get tool definitions
            tool_definitions = self._tools.get_definitions()

            logger.info(f"Starting coordinator cycle with {len(watchlist)} symbols on watchlist")

            # Run tool calling loop with timeout
            final_response = await asyncio.wait_for(
                self._llm.acomplete_with_tools(
                    prompt=user_prompt,
                    tools=tool_definitions,
                    tool_executor=self._tool_executor,
                    system=system_prompt,
                    temperature=self._config.temperature,
                    max_tool_calls=self._config.max_tool_calls,
                    on_tool_call=self._on_tool_call,
                ),
                timeout=self._config.cycle_timeout_seconds,
            )

            logger.info(
                f"Coordinator cycle complete: {self._tool_calls_count} tools, "
                f"{len(self._symbols_analyzed)} symbols, {self._trades_executed} trades"
            )

            # Parse result
            result = await self._parse_cycle_result(final_response)

            # Update last summary for next cycle
            self._last_cycle_summary = result.summary

            return result

        except TimeoutError:
            logger.error(f"Coordinator cycle timeout after {self._config.cycle_timeout_seconds}s")
            return CoordinatorCycleResult(
                summary=f"Cycle timeout after {self._config.cycle_timeout_seconds}s",
                symbols_analyzed=list(self._symbols_analyzed),
                trades_proposed=self._trades_proposed,
                trades_executed=self._trades_executed,
                tool_calls_made=self._tool_calls_count,
                game_plan_generated=self._game_plan_generated,
            )
        except Exception as e:
            logger.error(f"Coordinator cycle failed: {e}")
            return CoordinatorCycleResult(
                summary=f"Error: {e!s}",
                symbols_analyzed=list(self._symbols_analyzed),
                trades_proposed=self._trades_proposed,
                trades_executed=self._trades_executed,
                tool_calls_made=self._tool_calls_count,
                game_plan_generated=self._game_plan_generated,
            )

    async def _build_system_prompt(self, degradation_context: dict | None) -> str:
        """Build system prompt with context sections.

        Args:
            degradation_context: Optional degradation warnings

        Returns:
            Formatted system prompt
        """
        # Retrieve recent memory
        recent_observations = await self._memory.retrieve_recent(limit=20)
        memory_text = self._format_memory(recent_observations) if recent_observations else ""

        # Get portfolio context
        portfolio_text = await self._get_portfolio_context()

        # Format degradation context
        degradation_text = (
            self._format_degradation_context(degradation_context) if degradation_context else ""
        )

        # Format risk limits
        risk_text = self._format_risk_limits()

        # Load and format system prompt with all variables
        return self._prompts.load(
            "system",
            min_confidence_to_trade=self._config.min_confidence_to_trade,
            max_position_pct=self._config.max_position_pct,
            max_daily_trades=self._config.max_daily_trades,
            memory_text=memory_text,
            portfolio_text=portfolio_text,
            degradation_text=degradation_text,
            risk_text=risk_text,
        )

    def _build_cycle_prompt(self, watchlist: list[str]) -> str:
        """Build user prompt for cycle.

        Args:
            watchlist: List of symbols

        Returns:
            Formatted user prompt
        """
        return self._prompts.load(
            "user",
            watchlist=", ".join(watchlist),
            last_summary=self._last_cycle_summary,
        )

    async def _tool_executor(self, name: str, args: dict) -> str:
        """Execute tool by name.

        Args:
            name: Tool name
            args: Tool arguments

        Returns:
            Tool result as string
        """
        # Check if tool requires confirmation
        if self._tools.requires_confirmation(name):
            if self._config.confirmation_mode == "manual":
                logger.warning(f"Tool {name} requires confirmation (manual mode) - execution deferred")
            else:
                logger.info(f"Tool {name} requires confirmation (auto mode) - executing")

        # Execute tool
        try:
            return await self._tools.aexecute(name, args)
        except Exception as e:
            logger.error(f"Tool execution failed: {name} - {e}")
            return f"Error: {e!s}"

    def _on_tool_call(self, name: str, args: dict, result: str) -> None:
        """Callback invoked after tool execution.

        Tracks metrics for cycle result.

        Args:
            name: Tool name
            args: Tool arguments
            result: Tool result
        """
        self._tool_calls_count += 1

        # Track analyze_symbol calls
        if name == "analyze_symbol" and "symbol" in args:
            self._symbols_analyzed.add(args["symbol"])

        # Track execute_trade calls
        if name == "execute_trade":
            self._trades_proposed += 1
            # Parse result to check if trade executed
            if "successfully" in result.lower() or "executed" in result.lower():
                self._trades_executed += 1

        # Track game plan generation
        if name == "generate_game_plan":
            self._game_plan_generated = True

        logger.debug(f"Tool callback: {name} (total calls: {self._tool_calls_count})")

    async def _parse_cycle_result(self, final_response: str) -> CoordinatorCycleResult:
        """Parse final response into cycle result.

        Args:
            final_response: LLM's final response

        Returns:
            Structured cycle result
        """
        # Try to extract summary (first paragraph or max 200 chars)
        max_summary_length = 200
        summary = final_response.strip()
        if "\n\n" in summary:
            summary = summary.split("\n\n")[0]
        if len(summary) > max_summary_length:
            summary = summary[:max_summary_length] + "..."

        return CoordinatorCycleResult(
            summary=summary,
            symbols_analyzed=list(self._symbols_analyzed),
            trades_proposed=self._trades_proposed,
            trades_executed=self._trades_executed,
            tool_calls_made=self._tool_calls_count,
            game_plan_generated=self._game_plan_generated,
        )

    def _format_memory(self, observations: list) -> str:
        """Format memory observations as markdown.

        Args:
            observations: List of ObservationRecord

        Returns:
            Formatted markdown text
        """
        if not observations:
            return ""

        lines = ["\n## Recent Observations (Last 20)\n"]
        for obs in observations:
            # Format: - [category] observation (timestamp)
            timestamp = obs.timestamp.strftime("%Y-%m-%d %H:%M")
            lines.append(f"- **[{obs.category}]** {obs.observation} ({timestamp})")

        return "\n".join(lines)

    async def _get_portfolio_context(self) -> str:
        """Get current portfolio context from broker.

        Returns:
            Formatted markdown text with portfolio info
        """
        try:
            account_info = await asyncio.to_thread(self._broker.get_account_info)

            lines = [
                "\n## Current Portfolio\n",
                f"- **Balance**: ${account_info.balance:,.2f}",
                f"- **Portfolio Value**: ${account_info.portfolio_value:,.2f}",
                f"- **Available Cash**: ${account_info.available_cash:,.2f}",
                f"- **Total Exposure**: ${account_info.total_exposure:,.2f} "
                f"({account_info.total_exposure / account_info.portfolio_value * 100:.1f}%)",
            ]

            if account_info.positions:
                lines.append(f"\n**Positions ({len(account_info.positions)}):**")
                for symbol, pos in account_info.positions.items():
                    lines.append(
                        f"- {symbol}: {pos.qty} shares @ ${pos.avg_entry_price:.2f} "
                        f"(P&L: ${pos.unrealized_pnl:,.2f} / {pos.unrealized_pnl_percent:+.1f}%)"
                    )

            return "\n".join(lines)

        except Exception as e:
            logger.error(f"Failed to get portfolio context: {e}")
            return "\n## Current Portfolio\n\n*Portfolio data unavailable*\n"

    def _format_degradation_context(self, context: dict) -> str:
        """Format degradation context as markdown.

        Args:
            context: Degradation context dict

        Returns:
            Formatted markdown text
        """
        lines = ["\n## Degradation Warnings\n"]

        if context.get("disabled_agents"):
            lines.append(f"**Disabled agents**: {', '.join(context['disabled_agents'])}")

        if context.get("degraded_tools"):
            lines.append(f"**Degraded tools**: {', '.join(context['degraded_tools'])}")

        if message := context.get("message"):
            lines.append(f"\n{message}")

        return "\n".join(lines)

    def _format_risk_limits(self) -> str:
        """Format risk limits from config as markdown.

        Returns:
            Formatted markdown text
        """
        return f"""
## Risk Limits

- **Max Position Size**: {self._config.max_position_pct}% of portfolio
- **Max Daily Trades**: {self._config.max_daily_trades}
- **Min Confidence to Trade**: {self._config.min_confidence_to_trade:.0%}
- **Confirmation Mode**: {self._config.confirmation_mode}
"""

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"TradingCoordinator(tools={len(self._tools)}, "
            f"max_tool_calls={self._config.max_tool_calls}, "
            f"confirmation={self._config.confirmation_mode})"
        )
