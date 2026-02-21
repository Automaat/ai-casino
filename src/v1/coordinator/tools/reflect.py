"""Reflect on decision tool for coordinator."""

import asyncio
from typing import TYPE_CHECKING

from loguru import logger

from src.agents.critic import CriticAgent, CriticAnalysis, DecisionEvaluationRequest
from src.tools.base import BaseTool
from src.tools.models import ToolDefinition, ToolFunction, ToolParameter, ToolParametersSchema
from src.v1.coordinator.memory import DecisionQueryParams

if TYPE_CHECKING:
    from src.v1.coordinator.agent import TradingCoordinator
    from src.workflows.types import TradingWorkflowResult

# Max reflections per symbol per cycle
MAX_REFLECTIONS_PER_SYMBOL = 3


class ReflectOnDecisionTool(BaseTool):
    """Tool for LLM to request decision critique."""

    def __init__(self, coordinator: TradingCoordinator, critic_agent: CriticAgent) -> None:
        """Initialize reflection tool.

        Args:
            coordinator: Trading coordinator for state access
            critic_agent: Critic agent for evaluation
        """
        self._coordinator = coordinator
        self._critic_agent = critic_agent

    @property
    def name(self) -> str:
        """Tool name."""
        return "reflect_on_decision"

    @property
    def requires_confirmation(self) -> bool:
        """No confirmation needed - cheap operation."""
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
                    "Request a critical evaluation of a trading decision. Use when confidence is low "
                    "(<60%), signals are conflicting, or you're uncertain about decision quality. "
                    "The critic evaluates logical consistency, game plan alignment, and risk compliance. "
                    "Maximum 3 reflections per symbol per cycle."
                ),
                parameters=ToolParametersSchema(
                    properties={
                        "symbol": ToolParameter(
                            type="string",
                            description="Stock ticker symbol to reflect on (e.g., AAPL, TSLA)",
                        ),
                        "reason": ToolParameter(
                            type="string",
                            description=(
                                "Why requesting reflection (e.g., 'low confidence', 'conflicting signals')"
                            ),
                        ),
                    },
                    required=["symbol", "reason"],
                ),
            ),
        )

    def execute(self, **kwargs: str | int | float | bool) -> str:
        """Sync fallback - delegates to aexecute.

        Args:
            **kwargs: Tool arguments (symbol: str, reason: str)

        Returns:
            Formatted critique result
        """
        return asyncio.run(self.aexecute(**kwargs))

    async def aexecute(self, **kwargs: str | int | float | bool) -> str:
        """Execute reflection on decision asynchronously.

        Args:
            **kwargs: Tool arguments (symbol: str, reason: str)

        Returns:
            Formatted critique result
        """
        symbol = str(kwargs["symbol"]).upper()
        reason = str(kwargs["reason"])

        logger.info(f"Reflecting on {symbol} decision (reason: {reason})")

        # Check iteration limit
        current_count = self._coordinator.reflection_counters.get(symbol, 0)
        if current_count >= MAX_REFLECTIONS_PER_SYMBOL:
            return (
                f"⚠️ **Reflection limit reached for {symbol}** "
                f"({MAX_REFLECTIONS_PER_SYMBOL}/{MAX_REFLECTIONS_PER_SYMBOL} iterations used)\n\n"
                f"Cannot reflect further this cycle. Proceed with final decision or choose HOLD if uncertain."
            )

        # Retrieve last analysis result
        result = self._coordinator.last_analysis_results.get(symbol)
        if result is None:
            return (
                f"❌ **Cannot reflect on {symbol}** - no analysis found.\n\n"
                f"Run `analyze_symbol` first before requesting reflection."
            )

        # Build evaluation request
        request = await self._build_evaluation_request(symbol, result, reason)

        try:
            critique = await self._critic_agent.evaluate(request)
        except Exception as e:
            logger.opt(exception=True).error(f"Critique failed for {symbol}: {e}")
            return f"❌ **Critique failed for {symbol}:** {e}"

        # Increment reflection counter
        self._coordinator.reflection_counters[symbol] = current_count + 1

        # Format result
        return self._format_critique(symbol, reason, critique, current_count + 1)

    async def _build_evaluation_request(
        self,
        symbol: str,
        result: TradingWorkflowResult,
        reason: str,
    ) -> DecisionEvaluationRequest:
        """Build evaluation request from workflow result.

        Args:
            symbol: Stock symbol
            result: Trading workflow result
            reason: Reflection reason (unused in request body)

        Returns:
            DecisionEvaluationRequest
        """
        _ = reason  # contextual — logged at call site, not embedded in request
        constraints = {
            "min_confidence": self._coordinator.config.min_confidence_to_trade,
        }

        recent_outcomes = await self._get_recent_outcomes(symbol)

        return DecisionEvaluationRequest(
            symbol=symbol,
            decision=result.decision,
            technical=result.technical,
            sentiment=result.sentiment,
            news=result.news,
            fundamental=result.fundamental,
            risk=result.risk,
            game_plan_context=None,
            portfolio_constraints=constraints,
            recent_outcomes=recent_outcomes,
        )

    async def _get_recent_outcomes(self, symbol: str) -> list[str]:
        """Get recent execution outcomes for symbol from coordinator memory.

        Args:
            symbol: Stock symbol

        Returns:
            List of formatted outcome strings for critic context
        """
        try:
            params = DecisionQueryParams(symbol=symbol, lookback_days=30, limit=5)
            results = await self._coordinator.memory.query_decisions(params)
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to fetch recent outcomes for {symbol}: {e}")
            return []

        outcomes = []
        for r in results:
            date = r.timestamp.strftime("%m/%d")
            return_str = f"{r.return_pct:+.1f}%" if r.return_pct is not None else "pending"
            outcomes.append(f"{date} {r.signal} conf={r.confidence:.0%} → {r.hit_miss} ({return_str})")
        return outcomes

    def _format_critique(
        self,
        symbol: str,
        reason: str,
        critique: CriticAnalysis,
        iteration: int,
    ) -> str:
        """Format critique result as markdown.

        Args:
            symbol: Stock symbol
            reason: Reflection reason
            critique: Critique analysis
            iteration: Current iteration count

        Returns:
            Formatted markdown
        """
        status = "✅ PASSED" if critique.passed else "⚠️ FAILED"

        output = f"# Reflection on {symbol} Decision (Iteration {iteration}/3)\n\n"
        output += f"**Reason:** {reason}\n\n"
        output += f"**Quality Assessment:** {status}\n"
        output += f"**Overall Score:** {critique.overall_score:.0%}\n\n"

        output += "**Detailed Scores:**\n"
        output += f"- Logical Consistency: {critique.logical_consistency_score:.0%}\n"
        output += f"- Game Plan Alignment: {critique.game_plan_alignment_score:.0%}\n"
        output += f"- Risk Constraint Compliance: {critique.risk_constraint_score:.0%}\n\n"

        if critique.critical_issues:
            output += "## ⚠️ Critical Issues\n\n"
            for issue in critique.critical_issues:
                output += f"- {issue}\n"
            output += "\n"

        if critique.suggestions:
            output += "## 💡 Suggestions\n\n"
            for suggestion in critique.suggestions:
                output += f"- {suggestion}\n"
            output += "\n"

        output += "## 📝 Reflection\n\n"
        output += critique.reflection + "\n\n"

        if critique.passed:
            output += "**Next Steps:** Decision quality validated. Proceed with confidence.\n"
        else:
            output += (
                "**Next Steps:** Consider re-analyzing with revised approach, "
                "reducing position size, or choosing HOLD.\n"
            )

        return output
