"""Autonomous trading coordinator using LLM tool calling."""

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from loguru import logger
from result import Err

from src.models.llm import LLMClient
from src.prompts import PromptLoader
from src.strategies.session import TradingSession
from src.tools.registry import ToolRegistry
from src.v1.coordinator.event_prompt import EventCycleContext, EventCyclePromptBuilder, extract_symbols
from src.v1.coordinator.memory import CoordinatorMemory, DecisionQueryParams
from src.v1.coordinator.models import EVENT_CYCLE_TYPE, CoordinatorConfig, CoordinatorCycleResult

if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.agents.critic import CriticAgent
    from src.daemon.threshold_adapter import AdaptiveThresholdManager
    from src.v1.event_queue.models import QueuedMarketEvent
    from src.v1.trades.brokers import Broker
    from src.workflows.types import TradingWorkflowResult


class TradingCoordinator:
    """Autonomous trading coordinator using LLM tool calling.

    Orchestrates trading workflow through iterative tool use:
    - Analyzes market context (futures, sentiment, portfolio)
    - Generates daily game plan
    - Screens/analyzes potential opportunities
    - Executes trades based on conviction
    - Learns from outcomes via persistent memory
    """

    def __init__(  # noqa: PLR0913
        self,
        llm_client: LLMClient,
        tool_registry: ToolRegistry,
        memory: CoordinatorMemory,
        config: CoordinatorConfig,
        broker: Broker,
        critic_agent: CriticAgent,
        adaptive_threshold_manager: AdaptiveThresholdManager | None = None,
    ) -> None:
        """Initialize coordinator.

        Args:
            llm_client: LLM client for tool calling
            tool_registry: Registry of available tools
            memory: Persistent memory for observations
            config: Coordinator configuration
            broker: Broker for portfolio context
            critic_agent: Critic agent for decision evaluation
            adaptive_threshold_manager: Optional adaptive threshold manager
        """
        self._llm = llm_client
        self._tools = tool_registry
        self._memory = memory
        self._config = config
        self._broker = broker
        self._critic_agent = critic_agent  # Used by ReflectOnDecisionTool
        self._adaptive_threshold_manager = adaptive_threshold_manager
        self._prompts = PromptLoader("coordinator")
        self._last_cycle_summary = "No previous cycle"

        # Tracking variables (reset per cycle)
        self._tool_calls_count = 0
        self._symbols_analyzed: set[str] = set()
        self._trades_proposed = 0
        self._trades_executed = 0
        self._cycle_counter = 0

        # Reflection tracking (reset per cycle)
        self._reflection_counters: dict[str, int] = {}
        self._last_analysis_results: dict[str, TradingWorkflowResult] = {}

        logger.info("Initialized TradingCoordinator")

    @property
    def memory(self) -> CoordinatorMemory:
        """Access coordinator memory for saving observations.

        Returns:
            CoordinatorMemory instance
        """
        return self._memory

    @property
    def config(self) -> CoordinatorConfig:
        """Coordinator configuration.

        Returns:
            CoordinatorConfig instance
        """
        return self._config

    @property
    def reflection_counters(self) -> dict[str, int]:
        """Per-symbol reflection counters for current cycle.

        Returns:
            Mutable dict of symbol → reflection count
        """
        return self._reflection_counters

    @property
    def last_analysis_results(self) -> dict[str, TradingWorkflowResult]:
        """Last analysis results per symbol for current cycle.

        Returns:
            Mutable dict of symbol → TradingWorkflowResult
        """
        return self._last_analysis_results

    async def run_cycle(
        self,
        watchlist: list[str],
        degradation_context: dict | None = None,
        trading_session: TradingSession = TradingSession.REGULAR,
    ) -> CoordinatorCycleResult:
        """Run single coordinator cycle.

        Args:
            watchlist: List of symbols to consider
            degradation_context: Optional degradation warnings
            trading_session: Trading session type (REGULAR or PRE_MARKET)

        Returns:
            CoordinatorCycleResult with summary and metrics
        """
        import time

        # Reset tracking variables
        self._tool_calls_count = 0
        self._symbols_analyzed = set()
        self._trades_proposed = 0
        self._trades_executed = 0

        # Reset reflection tracking
        self._reflection_counters.clear()
        self._last_analysis_results.clear()

        # Track cycle duration
        cycle_start = time.time()

        # Increment cycle counter
        self._cycle_counter += 1

        try:
            # Update adaptive thresholds if needed
            if (
                self._adaptive_threshold_manager
                and self._cycle_counter % self._config.adaptive_thresholds.adaptation_interval_cycles == 0
            ):
                await self._update_adaptive_thresholds()

            # Build prompts
            system_prompt = await self._build_system_prompt(watchlist, degradation_context)
            user_prompt = await self._build_cycle_prompt(watchlist, trading_session)

            # Get tool definitions
            tool_definitions = self._tools.get_definitions()

            logger.info(f"Starting coordinator cycle with {len(watchlist)} symbols on watchlist")

            # Run tool calling loop with timeout
            from src.models.llm import ToolCallingParams

            final_response = await asyncio.wait_for(
                self._llm.acomplete_with_tools(
                    ToolCallingParams(
                        prompt=user_prompt,
                        tools=tool_definitions,
                        tool_executor=self._tool_executor,
                        system=system_prompt,
                        temperature=self._config.temperature,
                        max_tool_calls=self._config.max_tool_calls,
                        max_tokens=2048,
                        on_tool_call=self._on_tool_call,
                    )
                ),
                timeout=self._config.cycle_timeout_seconds,
            )

            logger.info(
                f"Coordinator cycle complete: {self._tool_calls_count} tools, "
                f"{len(self._symbols_analyzed)} symbols, {self._trades_executed} trades"
            )

            # Parse result
            result = await self._parse_cycle_result(final_response)

            # Add cycle duration
            result.cycle_duration_seconds = time.time() - cycle_start

            # Update last summary for next cycle
            self._last_cycle_summary = result.summary

            return result

        except TimeoutError:
            logger.opt(exception=True).error(
                f"Coordinator cycle timeout after {self._config.cycle_timeout_seconds}s"
            )
            return CoordinatorCycleResult(
                summary=f"Cycle timeout after {self._config.cycle_timeout_seconds}s",
                symbols_analyzed=list(self._symbols_analyzed),
                trades_proposed=self._trades_proposed,
                trades_executed=self._trades_executed,
                tool_calls_made=self._tool_calls_count,
                cycle_duration_seconds=time.time() - cycle_start,
            )
        except Exception as e:
            logger.opt(exception=True).error(f"Coordinator cycle failed: {e}")
            return CoordinatorCycleResult(
                summary=f"Error: {e!s}",
                symbols_analyzed=list(self._symbols_analyzed),
                trades_proposed=self._trades_proposed,
                trades_executed=self._trades_executed,
                tool_calls_made=self._tool_calls_count,
                cycle_duration_seconds=time.time() - cycle_start,
            )

    async def run_event_cycle(
        self,
        events: Sequence[QueuedMarketEvent],
        degradation_context: dict | None = None,
        trading_session: TradingSession = TradingSession.REGULAR,
        market_open: bool = True,
    ) -> CoordinatorCycleResult:
        """Run event-driven coordinator cycle for dequeued market events.

        Args:
            events: Dequeued market events to process
            degradation_context: Optional degradation warnings
            trading_session: Trading session type
            market_open: Whether market is currently open

        Returns:
            CoordinatorCycleResult with event-specific metrics
        """
        import time

        # Reset tracking variables (same pattern as run_cycle)
        self._tool_calls_count = 0
        self._symbols_analyzed = set()
        self._trades_proposed = 0
        self._trades_executed = 0
        self._reflection_counters.clear()
        self._last_analysis_results.clear()

        cycle_start = time.time()
        event_ids = [ev.event_id for ev in events]
        affected_symbols = sorted(extract_symbols(events))

        logger.info(
            f"Starting event cycle: {len(events)} events, symbols={affected_symbols}, ids={event_ids}"
        )

        try:
            # Build prompts with narrowed watchlist
            watchlist = affected_symbols or []
            system_prompt = await self._build_system_prompt(
                watchlist, degradation_context, max_tool_calls=self._config.event_max_tool_calls
            )

            positions_summary = await self._get_positions_summary()
            game_plan = await self._memory.get_today_game_plan(max_tokens=500)
            prompt_builder = EventCyclePromptBuilder()
            user_prompt = prompt_builder.build(
                events=events,
                context=EventCycleContext(
                    positions_summary=positions_summary,
                    session=trading_session,
                    market_open=market_open,
                    game_plan=game_plan,
                ),
                config=self._config,
            )

            tool_definitions = self._tools.get_definitions()

            from src.models.llm import ToolCallingParams

            final_response = await asyncio.wait_for(
                self._llm.acomplete_with_tools(
                    ToolCallingParams(
                        prompt=user_prompt,
                        tools=tool_definitions,
                        tool_executor=self._tool_executor,
                        system=system_prompt,
                        temperature=self._config.temperature,
                        max_tool_calls=self._config.event_max_tool_calls,
                        max_tokens=2048,
                        on_tool_call=self._on_tool_call,
                    )
                ),
                timeout=self._config.cycle_timeout_seconds,
            )

            logger.info(
                f"Event cycle complete: {self._tool_calls_count} tools, "
                f"{len(self._symbols_analyzed)} symbols, {self._trades_executed} trades"
            )

            result = await self._parse_cycle_result(final_response)
            result.cycle_duration_seconds = time.time() - cycle_start
            result.cycle_type = EVENT_CYCLE_TYPE
            result.event_ids = event_ids
            return result

        except TimeoutError:
            logger.opt(exception=True).error(
                f"Event cycle timeout after {self._config.cycle_timeout_seconds}s"
            )
            return CoordinatorCycleResult(
                summary=f"Event cycle timeout after {self._config.cycle_timeout_seconds}s",
                symbols_analyzed=list(self._symbols_analyzed),
                trades_proposed=self._trades_proposed,
                trades_executed=self._trades_executed,
                tool_calls_made=self._tool_calls_count,
                cycle_duration_seconds=time.time() - cycle_start,
                cycle_type=EVENT_CYCLE_TYPE,
                event_ids=event_ids,
            )
        except Exception as e:
            logger.opt(exception=True).error(f"Event cycle failed: {e}")
            return CoordinatorCycleResult(
                summary=f"Error: {e!s}",
                symbols_analyzed=list(self._symbols_analyzed),
                trades_proposed=self._trades_proposed,
                trades_executed=self._trades_executed,
                tool_calls_made=self._tool_calls_count,
                cycle_duration_seconds=time.time() - cycle_start,
                cycle_type=EVENT_CYCLE_TYPE,
                event_ids=event_ids,
            )

    async def _build_system_prompt(
        self,
        watchlist: list[str],
        degradation_context: dict | None,
        max_tool_calls: int | None = None,
    ) -> str:
        """Build system prompt with context sections.

        Args:
            watchlist: List of symbols to consider
            degradation_context: Optional degradation warnings
            max_tool_calls: Override max tool calls shown in prompt

        Returns:
            Formatted system prompt
        """
        # Retrieve recent memory
        recent_observations = await self._memory.retrieve_recent(limit=20)
        memory_section = self._format_memory(recent_observations) if recent_observations else ""

        # Get medium-term memory context from memory layer
        today_summary_section = await self._memory.get_today_summary(max_tokens=500)
        today_game_plan_section = await self._memory.get_today_game_plan(max_tokens=300)
        portfolio_section = await self._memory.get_portfolio_summary(max_tokens=400)

        # Format degradation context
        degradation_section = (
            self._format_degradation_context(degradation_context) if degradation_context else ""
        )

        # Get trading mode
        trading_mode = self._get_trading_mode()

        # Load and format system prompt with all variables
        return self._prompts.load(
            "system",
            min_confidence_to_trade=self._config.min_confidence_to_trade,
            max_position_pct=self._config.max_position_pct,
            max_daily_trades=self._config.max_daily_trades,
            max_tool_calls=max_tool_calls if max_tool_calls is not None else self._config.max_tool_calls,
            trading_mode=trading_mode,
            watchlist=", ".join(watchlist),
            memory_section=memory_section,
            today_summary_section=today_summary_section,
            today_game_plan_section=today_game_plan_section,
            portfolio_section=portfolio_section,
            degradation_section=degradation_section,
        )

    async def _build_cycle_prompt(
        self,
        watchlist: list[str],
        trading_session: TradingSession = TradingSession.REGULAR,
    ) -> str:
        """Build cycle prompt with current context.

        Args:
            watchlist: List of symbols
            trading_session: Trading session type (REGULAR or PRE_MARKET)

        Returns:
            Formatted cycle prompt
        """
        positions_summary = await self._get_positions_summary()
        recent_outcomes_section = await self._get_recent_outcomes_summary()
        current_date = datetime.now(UTC).strftime("%Y-%m-%d")
        session_name = trading_session.value

        return self._prompts.load(
            "cycle",
            watchlist=", ".join(watchlist),
            last_summary=self._last_cycle_summary,
            positions_summary=positions_summary,
            recent_outcomes_section=recent_outcomes_section,
            date=current_date,
            session=session_name,
            min_confidence_to_trade=self._config.min_confidence_to_trade,
            max_position_pct=self._config.max_position_pct,
            max_daily_trades=self._config.max_daily_trades,
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
                # In manual confirmation mode, do not execute the tool automatically.
                # Return an explicit status string so callers can surface this to users.
                return f"Skipped: awaiting manual confirmation for tool '{name}'."
            logger.info(f"Tool {name} requires confirmation (auto mode) - executing")

        # Execute tool
        try:
            return await self._tools.aexecute(name, args)
        except Exception as e:
            logger.opt(exception=True).error(f"Tool execution failed: {name} - {e}")
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

    def _get_trading_mode(self) -> str:
        """Derive trading mode from config.

        Returns:
            Trading mode description string
        """
        if self._config.confirmation_mode == "manual":
            return "MANUAL (requires confirmation for trades)"
        return "AUTO (trades execute automatically)"

    async def _get_positions_summary(self) -> str:
        """Get current positions summary for cycle prompt.

        Returns:
            Formatted positions summary string
        """
        result = await asyncio.to_thread(self._broker.get_account_info)
        if isinstance(result, Err):
            logger.opt(exception=True).warning(f"Failed to get positions summary: {result.err_value}")
            return "Positions data unavailable"
        account_info = result.ok()
        if not account_info.positions:
            return "No open positions"

        count = len(account_info.positions)
        position_label = "position" if count == 1 else "positions"
        lines = [f"{count} open {position_label}:"]
        for symbol, pos in account_info.positions.items():
            pnl_pct = pos.unrealized_pnl_percent
            pnl_dollar = pos.unrealized_pnl
            status = "profit" if pnl_dollar > 0 else "loss" if pnl_dollar < 0 else "flat"

            lines.append(
                f"- {symbol}: {pos.qty} shares @ ${pos.avg_entry_price:.2f} "
                f"({status}: {pnl_pct:+.1f}% / ${pnl_dollar:+,.2f})"
            )

        return "\n".join(lines)

    async def _get_recent_outcomes_summary(self, lookback_days: int = 7, limit: int = 10) -> str:
        """Get recent trade outcomes for cycle prompt context.

        Args:
            lookback_days: Days to look back for outcomes
            limit: Maximum number of outcomes to include

        Returns:
            Formatted outcomes summary or empty string
        """
        try:
            decisions = await self._memory.query_decisions(
                DecisionQueryParams(lookback_days=lookback_days, limit=limit)
            )
            if not decisions:
                return ""

            lines = ["## Recent Trade Outcomes (Last 7 Days)\n"]
            hits = sum(1 for d in decisions if d.hit_miss == "HIT")
            misses = sum(1 for d in decisions if d.hit_miss == "MISS")
            total_decided = hits + misses
            rate = f"{hits}/{total_decided} ({hits / total_decided:.0%})" if total_decided else "N/A"
            lines.append(f"**Success Rate:** {rate}\n")

            for d in decisions:
                ret = f"{d.return_pct:+.1f}%" if d.return_pct is not None else "pending"
                lines.append(f"- {d.symbol} {d.signal} {d.confidence:.0%} → {d.hit_miss} ({ret})")

            return "\n".join(lines)
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to get recent outcomes: {e}")
            return ""

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

    async def _update_adaptive_thresholds(self) -> None:
        """Update adaptive thresholds based on recent accuracy."""
        if not self._adaptive_threshold_manager:
            return

        try:
            thresholds = await self._adaptive_threshold_manager.update_thresholds()
            logger.info(
                f"Adaptive thresholds: BUY={thresholds.buy_threshold:.2f}, "
                f"SELL={thresholds.sell_threshold:.2f} "
                f"(updated {thresholds.adaptation_count} times)"
            )
        except Exception as e:
            logger.opt(exception=True).error(f"Threshold adaptation failed: {e}")

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"TradingCoordinator(tools={len(self._tools)}, "
            f"max_tool_calls={self._config.max_tool_calls}, "
            f"confirmation={self._config.confirmation_mode})"
        )
