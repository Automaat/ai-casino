"""Reflect on decision tool for coordinator."""

from typing import TYPE_CHECKING

from loguru import logger

from src.agents.critic import CriticAgent, CriticAnalysis, DecisionEvaluationRequest
from src.tools.base import BaseTool
from src.tools.models import ToolDefinition, ToolFunction, ToolParameter, ToolParametersSchema

if TYPE_CHECKING:
    from src.coordinator.agent import TradingCoordinator
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
        """Execute reflection on decision.

        Args:
            **kwargs: Tool arguments (symbol: str, reason: str)

        Returns:
            Formatted critique result
        """
        symbol = str(kwargs["symbol"]).upper()
        reason = str(kwargs["reason"])

        logger.info(f"Reflecting on {symbol} decision (reason: {reason})")

        # Check iteration limit
        current_count = self._coordinator._reflection_counters.get(symbol, 0)  # noqa: SLF001
        if current_count >= MAX_REFLECTIONS_PER_SYMBOL:
            return (
                f"⚠️ **Reflection limit reached for {symbol}** "
                f"({MAX_REFLECTIONS_PER_SYMBOL}/{MAX_REFLECTIONS_PER_SYMBOL} iterations used)\n\n"
                f"Cannot reflect further this cycle. Proceed with final decision or choose HOLD if uncertain."
            )

        # Retrieve last analysis result
        result = self._coordinator._last_analysis_results.get(symbol)  # noqa: SLF001
        if result is None:
            return (
                f"❌ **Cannot reflect on {symbol}** - no analysis found.\n\n"
                f"Run `analyze_symbol` first before requesting reflection."
            )

        # Build evaluation request
        request = self._build_evaluation_request(symbol, result, reason)

        # Run critique
        import asyncio
        import concurrent.futures

        def run_in_thread() -> CriticAnalysis:
            return asyncio.run(self._critic_agent.evaluate(request))

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_in_thread)
                critique = future.result()
        except Exception as e:
            logger.error(f"Critique failed for {symbol}: {e}")
            return f"❌ **Critique failed for {symbol}:** {e}"

        # Increment reflection counter
        self._coordinator._reflection_counters[symbol] = current_count + 1  # noqa: SLF001

        # Format result
        return self._format_critique(symbol, reason, critique, current_count + 1)

    def _build_evaluation_request(
        self,
        symbol: str,
        result: TradingWorkflowResult,
        reason: str,  # noqa: ARG002
    ) -> DecisionEvaluationRequest:
        """Build evaluation request from workflow result.

        Args:
            symbol: Stock symbol
            result: Trading workflow result
            reason: Reflection reason (logged but not used in request)

        Returns:
            DecisionEvaluationRequest
        """
        # Get portfolio constraints from config
        constraints = None
        if hasattr(self._coordinator, "_config"):
            constraints = {
                "min_confidence": self._coordinator._config.min_confidence_to_trade,  # noqa: SLF001
            }

        # Get recent outcomes from memory
        recent_outcomes = self._get_recent_outcomes(symbol)

        # Get game plan context
        game_plan = self._coordinator._game_plan_context  # noqa: SLF001

        return DecisionEvaluationRequest(
            symbol=symbol,
            decision=result.decision,
            technical=result.technical,
            sentiment=result.sentiment,
            news=result.news,
            fundamental=result.fundamental,
            risk=result.risk,
            game_plan_context=game_plan,
            portfolio_constraints=constraints,
            recent_outcomes=recent_outcomes,
        )

    def _get_recent_outcomes(self, _symbol: str) -> list[str]:
        """Get recent execution outcomes for symbol.

        Args:
            _symbol: Stock symbol (unused - async implementation pending)

        Returns:
            Empty list (placeholder for async memory retrieval)
        """
        # TODO: Make this async to properly retrieve memories from coordinator
        return []

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
