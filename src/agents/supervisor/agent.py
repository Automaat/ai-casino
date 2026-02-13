"""Trading Supervisor Agent."""

from loguru import logger

from src.agents.supervisor.models import (
    AnalysisRoutingDecision,
    AnalysisType,
    AnalysisWeights,
    PlanningContext,
    SynthesisContext,
)
from src.execution_tracking import track_agent
from src.models.llm import LLMClient
from src.models.providers.base import StructuredOutputError
from src.prompts import PromptLoader


class TradingSupervisor:
    """Intelligent analysis orchestrator with adaptive routing and result synthesis."""

    def __init__(self, llm_client: LLMClient) -> None:
        """Initialize trading supervisor.

        Args:
            llm_client: LLM client for planning and synthesis
        """
        self.llm = llm_client
        self._prompts = PromptLoader("supervisor")
        logger.info("Initialized TradingSupervisor")

    @track_agent
    async def plan_analyses(self, context: PlanningContext) -> AnalysisRoutingDecision:
        """Phase 1: Determine which analyses to run.

        Args:
            context: Planning context with market state and constraints

        Returns:
            AnalysisRoutingDecision with required/optional/skip lists
        """
        prompt = self._prompts.load(
            "plan",
            symbol=context.symbol,
            regime=context.regime.regime.value if context.regime else "unknown",
            session=context.trading_session.value,
            owns_position=context.owns_position,
            news_count=context.news_count,
            fundamental_available=context.fundamental_available,
            social_available=context.social_available,
            trump_count=context.trump_count,
            fundamental_rate_limit_status="limited" if context.fundamental_rate_limit else "available",
            time_budget_ms=context.time_budget_ms,
        )
        system = self._prompts.load("system")

        try:
            decision = await self.llm.astructured(
                prompt, AnalysisRoutingDecision, system=system, temperature=0.4
            )
        except StructuredOutputError as e:
            logger.opt(exception=True).warning(f"Structured output failed, using default: {e}")
            decision = self._default_routing(context)

        logger.info(
            f"Routing: {len(decision.required_analyses)} required, "
            f"{len(decision.optional_analyses)} optional, "
            f"{len(decision.skip_analyses)} skipped"
        )
        return decision

    @track_agent
    async def synthesize_results(
        self, context: SynthesisContext, completed: list[AnalysisType]
    ) -> AnalysisWeights:
        """Phase 2: Synthesize completed analyses.

        Args:
            context: Synthesis context with completed analysis summaries
            completed: List of completed analysis types

        Returns:
            AnalysisWeights with reliability scores and confidence adjustment
        """
        analyses_summary = self._format_analyses_summary(context, completed)

        prompt = self._prompts.load("synthesize", symbol=context.symbol, analyses_summary=analyses_summary)
        system = self._prompts.load("system")

        try:
            weights = await self.llm.astructured(prompt, AnalysisWeights, system=system, temperature=0.4)
        except StructuredOutputError as e:
            logger.opt(exception=True).warning(f"Structured output failed, uniform weights: {e}")
            weights = self._default_weights(completed)

        logger.info(
            f"Synthesis: {len(weights.conflicts)} conflicts, "
            f"{len(weights.consensus)} consensus, "
            f"confidence_adj={weights.confidence_adjustment:.2f}"
        )
        return weights

    def _default_routing(self, context: PlanningContext) -> AnalysisRoutingDecision:
        """Fallback when LLM fails.

        Args:
            context: Planning context

        Returns:
            Default routing decision
        """
        required = [
            AnalysisType.TECHNICAL,
            AnalysisType.SENTIMENT,
            AnalysisType.NEWS,
            AnalysisType.BULLISH_RESEARCH,
            AnalysisType.BEARISH_RESEARCH,
        ]
        optional = []
        skip = {}

        if context.fundamental_available and not context.fundamental_rate_limit:
            optional.append(AnalysisType.FUNDAMENTAL)
        else:
            skip[AnalysisType.FUNDAMENTAL] = "API rate limited or unavailable"

        if context.social_available:
            optional.append(AnalysisType.SOCIAL_SENTIMENT)

        if context.trump_count > 0:
            optional.append(AnalysisType.TRUMP)

        return AnalysisRoutingDecision(
            required_analyses=required,
            optional_analyses=optional,
            skip_analyses=skip,
            reasoning="Default routing (LLM fallback)",
            priority_order=required + optional,
        )

    def _default_weights(self, completed: list[AnalysisType]) -> AnalysisWeights:
        """Fallback uniform weights.

        Args:
            completed: List of completed analyses

        Returns:
            Uniform weights for all completed analyses
        """
        weights = dict.fromkeys(completed, 0.8)
        return AnalysisWeights(
            weights=weights,
            conflicts=[],
            consensus=[],
            confidence_adjustment=1.0,
            reasoning="Uniform weights (LLM fallback)",
        )

    def _format_analyses_summary(self, context: SynthesisContext, completed: list[AnalysisType]) -> str:
        """Format completed analyses for synthesis prompt.

        Args:
            context: Synthesis context with analysis summaries
            completed: List of completed analysis types

        Returns:
            Formatted summary string
        """
        summary_map = {
            AnalysisType.TECHNICAL: context.technical_summary,
            AnalysisType.SENTIMENT: context.sentiment_summary,
            AnalysisType.NEWS: context.news_summary,
            AnalysisType.FUNDAMENTAL: context.fundamental_summary,
            AnalysisType.COMPARATIVE: context.comparative_summary,
            AnalysisType.WEB_RESEARCH: context.web_research_summary,
            AnalysisType.SOCIAL_SENTIMENT: context.social_summary,
            AnalysisType.BULLISH_RESEARCH: context.bullish_summary,
            AnalysisType.BEARISH_RESEARCH: context.bearish_summary,
            AnalysisType.TRUMP: context.trump_summary,
        }

        lines = []
        for analysis_type in completed:
            summary = summary_map.get(analysis_type)
            if summary:
                lines.append(f"{analysis_type.value.upper()}: {summary}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        """String representation."""
        return f"TradingSupervisor(llm={self.llm.provider})"
