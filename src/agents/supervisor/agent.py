"""Trading Supervisor Agent."""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from src.agents.supervisor.models import (
    AnalysisRoutingDecision,
    AnalysisType,
    AnalysisWeights,
    CandidateEvaluationContext,
    CandidateRanking,
    PlanningContext,
    SynthesisContext,
    TradeApprovalContext,
    TradeApprovalDecision,
)
from src.execution_tracking import track_agent
from src.models.llm import LLMClient
from src.models.providers.base import StructuredOutputError
from src.prompts import PromptLoader

if TYPE_CHECKING:
    from src.di.container import AppContainer
    from src.metrics.execution import ExecutionMetricsCollector
    from src.strategies.ensemble import EnsembleStrategy
    from src.strategies.momentum import MomentumStrategy
    from src.strategies.session import TradingSession
    from src.workflows.config import WorkflowComponents, WorkflowConfig
    from src.workflows.types import TradingWorkflowResult, WorkflowExtraContext


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
    async def plan_analyses(
        self, context: PlanningContext, *, symbol: str | None = None
    ) -> AnalysisRoutingDecision:
        """Phase 1: Determine which analyses to run.

        Args:
            context: Planning context with market state and constraints
            symbol: Trading symbol for execution tracking; defaults to context.symbol

        Returns:
            AnalysisRoutingDecision with required/optional/skip lists
        """
        if symbol is None:
            symbol = context.symbol
        elif symbol != context.symbol:
            logger.warning(
                "plan_analyses called with mismatched symbol (%s) and context.symbol (%s)",
                symbol,
                context.symbol,
            )

        from src.strategies.regime import MIN_ROWS_FOR_REGIME

        # Format position P&L section
        position_pnl_section = ""
        if context.position_pnl:
            p = context.position_pnl
            position_pnl_section = (
                f"## Position P&L\n\n"
                f"- Entry Price: ${p.entry_price:.2f}\n"
                f"- Unrealized P&L: {p.unrealized_pnl_percent:+.2f}%\n"
                f"- Days Held: {p.days_held}\n"
                f"- Quantity: {p.current_qty}"
            )

        # Format portfolio summary section
        portfolio_summary_section = self._format_portfolio_summary(context)

        # Format health constraints section
        health_constraints_section = context.portfolio_health_constraints or "None"

        prompt = self._prompts.load(
            "plan",
            symbol=symbol,
            regime=context.regime.regime.value if context.regime else "unknown",
            session=context.trading_session.value,
            owns_position=context.owns_position,
            news_count=context.news_count,
            market_data_rows=context.market_data_rows,
            min_rows_required=MIN_ROWS_FOR_REGIME,
            fundamental_status="available" if context.fundamental_available else "unavailable",
            social_status="available" if context.social_available else "unavailable",
            trump_count=context.trump_count,
            is_high_volatility=context.is_high_volatility,
            fundamental_api_status="rate limited" if context.fundamental_rate_limit else "available",
            time_budget_ms=context.time_budget_ms,
            economic_risk=context.economic_risk or "None",
            position_pnl_section=position_pnl_section,
            portfolio_summary_section=portfolio_summary_section,
            health_constraints_section=health_constraints_section,
        )
        system = self._prompts.load("system")

        try:
            decision = await self.llm.astructured(
                prompt, AnalysisRoutingDecision, system=system, temperature=0.4, max_tokens=4096
            )
        except StructuredOutputError as e:
            logger.opt(exception=True).warning(f"Structured output failed, using default: {e}")
            decision = self.default_routing(context)

        # Ensure SENTIMENT/NEWS are never skipped - workers handle empty articles gracefully
        for analysis_type in [AnalysisType.SENTIMENT, AnalysisType.NEWS]:
            if analysis_type in decision.skip_analyses:
                reason = decision.skip_analyses.pop(analysis_type)
                logger.debug(f"Promoting {analysis_type.value} from skip to optional: {reason}")
                if (
                    analysis_type not in decision.optional_analyses
                    and analysis_type not in decision.required_analyses
                ):
                    decision.optional_analyses.append(analysis_type)

        # Log routing summary
        required_types = (
            ", ".join([a.value for a in decision.required_analyses]) if decision.required_analyses else "none"
        )
        optional_types = (
            ", ".join([a.value for a in decision.optional_analyses]) if decision.optional_analyses else "none"
        )
        logger.info(
            f"Routing: {len(decision.required_analyses)} required ({required_types}), "
            f"{len(decision.optional_analyses)} optional ({optional_types}), "
            f"{len(decision.skip_analyses)} skipped"
        )

        # Log skip reasons for each skipped analysis (debug to avoid duplicate info-level noise)
        for analysis_type, reason in decision.skip_analyses.items():
            logger.debug(f"Skip {analysis_type.value}: {reason}")

        # Debug log: full routing reasoning
        logger.debug(f"Routing reasoning: {decision.reasoning}")

        return decision

    def _format_portfolio_summary(self, context: PlanningContext) -> str:
        """Format portfolio summary section for the planning prompt.

        Args:
            context: Planning context with optional portfolio summary

        Returns:
            Formatted string for prompt injection
        """
        if not context.portfolio_summary:
            return "No portfolio data"
        ps = context.portfolio_summary
        if not ps.total_positions:
            return "No open positions"
        winner_str = (
            f"{ps.biggest_winner} ({ps.biggest_winner_pnl_percent:+.2f}%)" if ps.biggest_winner else "N/A"
        )
        loser_str = (
            f"{ps.biggest_loser} ({ps.biggest_loser_pnl_percent:+.2f}%)" if ps.biggest_loser else "N/A"
        )
        return (
            f"- Positions: {ps.total_positions}\n"
            f"- Exposure: {ps.total_exposure_percent:.1f}%\n"
            f"- Portfolio P&L: {ps.portfolio_pnl_percent:+.2f}%\n"
            f"- Biggest Winner: {winner_str}\n"
            f"- Biggest Loser: {loser_str}"
        )

    @track_agent
    async def synthesize_results(
        self, context: SynthesisContext, completed: list[AnalysisType], *, symbol: str | None = None
    ) -> AnalysisWeights:
        """Phase 2: Synthesize completed analyses.

        Args:
            context: Synthesis context with completed analysis summaries
            completed: List of completed analysis types
            symbol: Trading symbol for execution tracking; defaults to context.symbol

        Returns:
            AnalysisWeights with reliability scores and confidence adjustment
        """
        if symbol is None:
            symbol = context.symbol
        elif symbol != context.symbol:
            logger.warning(
                "synthesize_results called with mismatched symbol (%s) and context.symbol (%s)",
                symbol,
                context.symbol,
            )

        # Short-circuit when no analyses completed (avoid wasting LLM tokens)
        if not completed:
            logger.info("No analyses completed, returning default weights")
            return self._default_weights(completed)

        analyses_summary = self._format_analyses_summary(context, completed)

        prompt = self._prompts.load("synthesize", symbol=context.symbol, analyses_summary=analyses_summary)
        system = self._prompts.load("system")

        try:
            weights = await self.llm.astructured(
                prompt, AnalysisWeights, system=system, temperature=0.4, max_tokens=4096
            )
        except StructuredOutputError as e:
            logger.opt(exception=True).warning(f"Structured output failed, uniform weights: {e}")
            weights = self._default_weights(completed)

        # Log synthesis summary
        weighted_analyses = ", ".join([f"{t.value}={w:.2f}" for t, w in weights.weights.items()])
        conflict_pairs = ", ".join(weights.conflicts) if weights.conflicts else "none"
        consensus_items = ", ".join(weights.consensus) if weights.consensus else "none"
        logger.info(
            f"Synthesis: {len(completed)} analyses weighted ({weighted_analyses}), "
            f"{len(weights.conflicts)} conflicts ({conflict_pairs}), "
            f"{len(weights.consensus)} consensus ({consensus_items}), "
            f"confidence_adj={weights.confidence_adjustment:.2f}"
        )

        # Debug log: full synthesis reasoning
        logger.debug(f"Synthesis reasoning: {weights.reasoning}")

        return weights

    @track_agent
    async def evaluate_candidates(self, context: CandidateEvaluationContext) -> CandidateRanking:
        """Evaluate discovery candidates for watchlist addition.

        Scoring criteria:
        - Quality (30%): multi-source agreement, data completeness
        - Momentum (30%): earnings proximity, sector momentum, timing
        - Risk (20%): volatility, market cap, liquidity
        - Portfolio Fit (20%): sector diversification, no overlap

        Args:
            context: Evaluation context with candidates and portfolio state

        Returns:
            CandidateRanking with ADD/DEFER/SKIP recommendations
        """
        if not context.candidates:
            logger.info("No candidates to evaluate")
            return CandidateRanking(
                evaluations=[],
                add_watchlist=[],
                defer=[],
                skip=[],
                priority_order=[],
                overall_reasoning="No candidates provided",
                warnings=[],
            )

        prompt = self._build_evaluation_prompt(context)
        system = self._prompts.load("system")

        try:
            ranking = await self.llm.astructured(
                prompt, CandidateRanking, system=system, temperature=0.4, max_tokens=8192
            )
        except StructuredOutputError as e:
            logger.opt(exception=True).warning(f"Structured output failed, fallback: {e}")
            ranking = self._default_candidate_ranking(context)

        ranking = self._enforce_constraints(ranking, context)

        logger.info(
            f"Candidate evaluation: {len(ranking.add_watchlist)} add, "
            f"{len(ranking.defer)} defer, {len(ranking.skip)} skip"
        )

        return ranking

    def _build_evaluation_prompt(self, context: CandidateEvaluationContext) -> str:
        """Build evaluation prompt with formatted context."""
        sector_exposure_summary = self._format_sector_exposure(context.sector_exposure)
        candidate_summaries = self._format_candidates(context.candidates)
        recent_outcomes = self._format_recent_outcomes(context.recent_discovery_outcomes)

        regime_str = context.market_regime.regime.value if context.market_regime else "unknown"
        session_str = context.session.value

        return self._prompts.load(
            "evaluate_candidates",
            regime=regime_str,
            session=session_str,
            portfolio_size=len(context.portfolio_symbols),
            watchlist_size=len(context.watchlist_symbols),
            watchlist_max=len(context.watchlist_symbols) + context.watchlist_capacity,
            watchlist_capacity=context.watchlist_capacity,
            sector_exposure_summary=sector_exposure_summary,
            candidate_count=len(context.candidates),
            candidate_summaries=candidate_summaries,
            recent_outcomes=recent_outcomes,
        )

    def _format_sector_exposure(self, sector_exposure: dict[str, float]) -> str:
        """Format sector exposure for prompt."""
        if not sector_exposure:
            return "No sector exposure data"

        lines = []
        for sector, ratio in sorted(sector_exposure.items(), key=lambda x: -x[1]):
            lines.append(f"- {sector}: {ratio:.1%}")
        return "\n".join(lines)

    def _format_candidates(self, candidates: list) -> str:
        """Format candidates for prompt."""
        lines = []
        for idx, candidate in enumerate(candidates, 1):
            sources_str = ", ".join(str(s.value if hasattr(s, "value") else s) for s in candidate.sources)
            metadata_items = []
            if hasattr(candidate, "metadata") and candidate.metadata:
                for key, val in candidate.metadata.items():
                    if key in ["volume", "market_cap", "gap_percent", "volume_ratio"]:
                        metadata_items.append(f"{key}={val}")

            metadata_str = f" ({', '.join(metadata_items)})" if metadata_items else ""

            lines.append(
                f"{idx}. {candidate.symbol} - Score: {candidate.composite_score:.2f}, "
                f"Sources: [{sources_str}], "
                f"Sector: {getattr(candidate, 'sector', 'Unknown')}{metadata_str}"
            )
        return "\n".join(lines)

    def _format_recent_outcomes(self, outcomes: list[str] | None) -> str:
        """Format recent discovery outcomes for prompt."""
        if not outcomes:
            return "No recent outcome data available"
        return "\n".join(f"- {outcome}" for outcome in outcomes)

    def _default_candidate_ranking(self, context: CandidateEvaluationContext) -> CandidateRanking:
        """Fallback ranking when LLM unavailable - score-based heuristics.

        Rules:
        - ADD: composite_score >= 0.75 AND capacity available
        - DEFER: 0.60 <= composite_score < 0.75
        - SKIP: composite_score < 0.60 OR sector overconcentrated (>30%)
        """
        from src.agents.supervisor.models import CandidateEvaluation, CandidateRecommendation

        evaluations = []
        add_watchlist = []
        defer = []
        skip = []

        for candidate in context.candidates:
            score = candidate.composite_score
            symbol = candidate.symbol
            sector = getattr(candidate, "sector", "Unknown")

            sector_ratio = context.sector_exposure.get(sector, 0.0)
            sector_overweight = sector_ratio > 0.30

            if score >= 0.75 and not sector_overweight:
                recommendation = CandidateRecommendation.ADD_WATCHLIST
                reasoning = f"High score ({score:.2f}), sector not overweight"
                add_watchlist.append(symbol)
            elif 0.60 <= score < 0.75:
                recommendation = CandidateRecommendation.DEFER
                reasoning = f"Medium score ({score:.2f}), revisit later"
                defer.append(symbol)
            else:
                recommendation = CandidateRecommendation.SKIP
                if sector_overweight:
                    reasoning = f"Sector {sector} overweight ({sector_ratio:.1%})"
                else:
                    reasoning = f"Low score ({score:.2f})"
                skip.append(symbol)

            evaluations.append(
                CandidateEvaluation(
                    symbol=symbol,
                    quality_score=score,
                    momentum_score=score,
                    risk_score=1.0 - score,
                    portfolio_fit_score=0.0 if sector_overweight else 1.0,
                    recommendation=recommendation,
                    reasoning=reasoning,
                )
            )

        priority_order = sorted(
            add_watchlist,
            key=lambda s: next(c.composite_score for c in context.candidates if c.symbol == s),
            reverse=True,
        )

        return CandidateRanking(
            evaluations=evaluations,
            add_watchlist=add_watchlist,
            defer=defer,
            skip=skip,
            priority_order=priority_order,
            overall_reasoning="Heuristic fallback: score-based ranking with sector constraints",
            warnings=["LLM unavailable, using fallback heuristics"],
        )

    def _enforce_constraints(
        self, ranking: CandidateRanking, context: CandidateEvaluationContext
    ) -> CandidateRanking:
        """Enforce capacity limits, remove duplicates, warn on sector concentration."""
        add_watchlist = ranking.add_watchlist
        priority_order = ranking.priority_order
        warnings = list(ranking.warnings)

        all_symbols = context.portfolio_symbols + context.watchlist_symbols
        add_watchlist = [s for s in add_watchlist if s not in all_symbols]
        priority_order = [s for s in priority_order if s not in all_symbols]

        if len(add_watchlist) > context.watchlist_capacity:
            excess = len(add_watchlist) - context.watchlist_capacity
            warnings.append(f"Truncated {excess} symbols to fit watchlist capacity")
            add_watchlist = add_watchlist[: context.watchlist_capacity]
            priority_order = priority_order[: context.watchlist_capacity]

        for sector, ratio in context.sector_exposure.items():
            if ratio > 0.30:
                warnings.append(f"Sector {sector} currently overweight at {ratio:.1%}")

        return CandidateRanking(
            evaluations=ranking.evaluations,
            add_watchlist=add_watchlist,
            defer=ranking.defer,
            skip=ranking.skip,
            priority_order=priority_order,
            overall_reasoning=ranking.overall_reasoning,
            warnings=warnings,
        )

    def _route_optional_analyses(
        self, context: PlanningContext, optional: list[AnalysisType], skip: dict[AnalysisType, str]
    ) -> None:
        """Route optional analyses based on context.

        Args:
            context: Planning context
            optional: List to append optional analyses
            skip: Dict to record skipped analyses
        """
        # Fundamental: optional unless rate-limited
        if context.fundamental_rate_limit:
            skip[AnalysisType.FUNDAMENTAL] = "API rate limited"
        elif not context.fundamental_available:
            skip[AnalysisType.FUNDAMENTAL] = "Fundamental data unavailable"
        else:
            optional.append(AnalysisType.FUNDAMENTAL)

        # Comparative: optional (adds context)
        optional.append(AnalysisType.COMPARATIVE)

        # Web research: optional (adds context)
        optional.append(AnalysisType.WEB_RESEARCH)

        # Social sentiment: optional if available
        if context.social_available:
            optional.append(AnalysisType.SOCIAL_SENTIMENT)
        else:
            skip[AnalysisType.SOCIAL_SENTIMENT] = "Social sentiment data unavailable"

        # Trump: optional if posts exist
        if context.trump_count > 0:
            optional.append(AnalysisType.TRUMP)
        else:
            skip[AnalysisType.TRUMP] = "No Trump posts available"

    def default_routing(self, context: PlanningContext) -> AnalysisRoutingDecision:
        """Fallback routing when LLM unavailable - intelligent data-driven decisions.

        Args:
            context: Planning context

        Returns:
            Default routing decision
        """
        from src.strategies.regime import MIN_ROWS_FOR_REGIME

        required: list[AnalysisType] = []
        optional: list[AnalysisType] = []
        skip: dict[AnalysisType, str] = {}

        # Technical: skip if insufficient data
        if context.market_data_rows < MIN_ROWS_FOR_REGIME:
            skip[AnalysisType.TECHNICAL] = (
                f"Insufficient data ({context.market_data_rows} < {MIN_ROWS_FOR_REGIME} required)"
            )
        else:
            required.append(AnalysisType.TECHNICAL)

        # Sentiment + News: optional if no articles (workers return defaults for empty input)
        if context.news_count == 0:
            optional.extend([AnalysisType.SENTIMENT, AnalysisType.NEWS])
        else:
            required.extend([AnalysisType.SENTIMENT, AnalysisType.NEWS])

        # Route optional analyses
        self._route_optional_analyses(context, optional, skip)

        # Research: required only if technical not skipped
        if AnalysisType.TECHNICAL not in skip:
            required.extend([AnalysisType.BULLISH_RESEARCH, AnalysisType.BEARISH_RESEARCH])
        else:
            skip[AnalysisType.BULLISH_RESEARCH] = "Technical skipped (dependency)"
            skip[AnalysisType.BEARISH_RESEARCH] = "Technical skipped (dependency)"

        # Build priority order based on context
        priority_order = self._build_priority_order(context, required, optional)

        return AnalysisRoutingDecision(
            required_analyses=required,
            optional_analyses=optional,
            skip_analyses=skip,
            reasoning="Intelligent fallback routing based on data availability",
            priority_order=priority_order,
        )

    def _build_priority_order(
        self,
        context: PlanningContext,
        required: list[AnalysisType],
        optional: list[AnalysisType],
    ) -> list[AnalysisType]:
        """Build execution priority order based on session and conditions.

        Args:
            context: Planning context
            required: Required analyses
            optional: Optional analyses

        Returns:
            Priority-ordered list of analyses
        """
        from src.strategies.session import TradingSession

        if context.trading_session == TradingSession.PRE_MARKET:
            # Pre-market: prioritize news/sentiment for breaking developments
            priority = []
            for analysis_type in [AnalysisType.NEWS, AnalysisType.SENTIMENT]:
                if analysis_type in required:
                    priority.append(analysis_type)
            # Add remaining required analyses
            for analysis_type in required:
                if analysis_type not in priority:
                    priority.append(analysis_type)
            return priority + optional

        # Regular session: standard order (technical → sentiment → news → research)
        return required + optional

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

    @track_agent
    async def approve_trade(
        self, context: TradeApprovalContext, *, symbol: str | None = None
    ) -> TradeApprovalDecision:
        """Final gate: review full research and approve or reject trade.

        Args:
            context: Trade approval context with all research summaries
            symbol: Trading symbol for execution tracking; defaults to context.symbol

        Returns:
            TradeApprovalDecision with approved flag and reasoning
        """
        if symbol is None:
            symbol = context.symbol
        prompt = self._prompts.load(
            "approve_trade",
            symbol=symbol,
            action=context.action.value,
            confidence=context.confidence,
            risk_level=context.risk_level,
            risk_score=context.risk_score,
            current_price=context.current_price,
            recommended_shares=context.recommended_shares,
            position_value=context.position_value,
            stop_loss_price=context.stop_loss_price,
            reward_risk_ratio=context.reward_risk_ratio or "N/A",
            decision_reasoning="\n".join(f"- {r}" for r in context.decision_reasoning),
            technical_summary=context.technical_summary or "N/A",
            sentiment_summary=context.sentiment_summary or "N/A",
            news_summary=context.news_summary or "N/A",
            bullish_summary=context.bullish_summary or "N/A",
            bearish_summary=context.bearish_summary or "N/A",
            risk_warnings="\n".join(f"- {w}" for w in context.risk_warnings) or "None",
        )
        system = self._prompts.load("system")
        try:
            decision = await self.llm.astructured(
                prompt, TradeApprovalDecision, system=system, temperature=0.3
            )
        except StructuredOutputError as e:
            logger.opt(exception=True).warning(f"Structured output failed, using fallback: {e}")
            decision = self._default_approval(context)
        logger.info(
            f"Supervisor trade approval for {symbol}: "
            f"{'APPROVED' if decision.approved else 'REJECTED'} - {decision.reasoning}"
        )
        return decision

    def _default_approval(self, context: TradeApprovalContext) -> TradeApprovalDecision:
        """Fallback approval: approve if confidence >= 0.7 and risk LOW/MEDIUM."""
        approved = context.confidence >= 0.7 and context.risk_level in ("LOW", "MEDIUM")
        return TradeApprovalDecision(
            approved=approved,
            reasoning=f"Fallback heuristic: confidence={context.confidence:.2f}, risk={context.risk_level}",
            key_concerns=(
                [] if approved else ["LLM unavailable, heuristic rejected low-confidence/high-risk trade"]
            ),
        )

    async def coordinate(
        self,
        symbol: str,
        period_days: int,
        components: WorkflowComponents,
        config: WorkflowConfig,
        trading_session: TradingSession | None = None,
        collector: ExecutionMetricsCollector | None = None,
        target_allocations: dict[str, float] | None = None,
        extra_context: WorkflowExtraContext | None = None,
    ) -> TradingWorkflowResult:
        """Coordinate full trading workflow with adaptive stage execution.

        This method orchestrates all 8 workflow stages using workers instead of agents.

        Args:
            symbol: Stock ticker symbol
            period_days: Days of historical data
            components: Workflow components (fetchers, broker, etc.)
            config: Workflow configuration
            trading_session: Trading session type (defaults to REGULAR)
            collector: Optional metrics collector
            target_allocations: Optional target portfolio allocations
            extra_context: Optional workflow context

        Returns:
            TradingWorkflowResult with all analyses and final decision

        Note:
            This method delegates to existing stage functions from workflows/stages/
            for most stages, using workers for the analysis stage (Stage 5).
        """
        from src.strategies.session import TradingSession
        from src.workflows.stages.instrumented_analysis import (
            AnalysisRequest,
            AnalysisRequestParams,
            run_instrumented_analysis,
        )

        logger.info(
            f"Supervisor coordinating workflow for {symbol} (supervisor mode - using existing pipeline)"
        )

        # Create minimal workflow and delegate to existing pipeline
        workflow = SupervisorWorkflow(components, config, self, target_allocations)
        session = trading_session or TradingSession.REGULAR
        params = AnalysisRequestParams(period_days, session, extra_context)
        # SupervisorWorkflow duck-types as TradingWorkflow (structural compatibility)
        request = AnalysisRequest(workflow, symbol, params, collector)  # pyrefly: ignore[bad-argument-type]

        return await run_instrumented_analysis(request)

    def __repr__(self) -> str:
        """String representation."""
        return f"TradingSupervisor(llm={self.llm.provider})"


class SupervisorWorkflow:
    """Minimal workflow object for instrumented analysis delegation."""

    def __init__(
        self,
        components: WorkflowComponents,
        config: WorkflowConfig,
        supervisor: TradingSupervisor,
        target_allocations: dict[str, float] | None = None,
    ) -> None:
        """Initialize SupervisorWorkflow with components and configuration."""
        self._init_components(components)
        self._init_config(components, config)
        self._init_conditional_components(components)
        self._init_agents(components)
        self.supervisor = supervisor
        self._target_allocations = target_allocations

    def _init_components(self, components: WorkflowComponents) -> None:
        """Initialize core component references."""
        from src.daemon.config import AnalysisOrchestratorConfig

        self.market_fetcher = components.market_fetcher
        self.news_fetcher = components.news_fetcher
        self.finbert = components.finbert
        self.fundamental_fetcher = components.fundamental_fetcher
        self.broker = components.broker
        self.metrics_tracker = components.metrics_tracker
        self.snapshot_repository = components.snapshot_repository
        self.execution_metric_repository = components.execution_metric_repository
        self.notification_service = components.notification_service
        self._container = components.container
        # Always use supervisor routing - default config if not provided
        self.analysis_orchestrator_config = (
            components.analysis_orchestrator_config or AnalysisOrchestratorConfig()
        )
        self.event_bus = components.event_bus
        self.web_search_fetcher = components.web_search_fetcher
        self.trading_service = components.trading_service

    def _init_config(self, components: WorkflowComponents, config: WorkflowConfig) -> None:
        """Initialize configuration attributes."""
        self.use_ensemble = config.use_ensemble
        self.use_meta_agent = config.use_meta_agent
        self.trump_mode = config.trump_mode
        self.snapshot_on_trade = config.snapshot_on_trade or False
        self.execution_metrics_enabled = config.execution_metrics_enabled
        self.pre_trade_backtest_config = config.pre_trade_backtest_config
        self.risk_validation_config = components.risk_validation_config
        self.risk_validator = components.risk_validator

    def _init_conditional_components(self, components: WorkflowComponents) -> None:
        """Initialize conditional components (Trump, meta-agent, backtest)."""
        # Trump fetcher
        if self.trump_mode:
            from src.data.truth_social import TruthSocialFetcher

            self.trump_fetcher: TruthSocialFetcher | None = TruthSocialFetcher(
                historical_cache=components.historical_cache
            )
        else:
            self.trump_fetcher: TruthSocialFetcher | None = None

        # Meta-agent
        if self.use_meta_agent:
            self.meta_agent = self._container.meta_agent()
            if components.metrics_tracker:
                self.meta_agent.metrics_tracker = components.metrics_tracker
            if components.param_store:
                self.meta_agent.param_store = components.param_store
        else:
            self.meta_agent = None  # pyrefly: ignore[bad-assignment]

        # Default strategy
        from src.strategies.ensemble import EnsembleStrategy
        from src.strategies.momentum import MomentumStrategy

        self._default_strategy = EnsembleStrategy() if self.use_ensemble else MomentumStrategy()

        # Backtest runner
        if self.pre_trade_backtest_config and self.pre_trade_backtest_config.enabled:
            from src.backtesting import VectorBTRunner

            self.vectorbt_runner: VectorBTRunner | None = VectorBTRunner()
        else:
            self.vectorbt_runner: VectorBTRunner | None = None

    def _init_agents(self, components: WorkflowComponents) -> None:
        """Initialize workers and risk agents."""
        from src.agents.risk import RiskManagementAgent

        self.technical_worker = self._container.technical_worker()
        self.sentiment_worker = self._container.sentiment_worker()
        self.news_worker = self._container.news_worker()
        self.fundamental_worker = self._container.fundamental_worker()
        self.comparative_worker = self._container.comparative_worker()
        self.web_researcher = self._container.web_research_worker()
        self.social_worker = self._container.social_sentiment_worker()
        self.bullish_researcher = self._container.bullish_thesis_worker()
        self.bearish_researcher = self._container.bearish_thesis_worker()
        if self.trump_mode:
            self.trump_worker = self._container.trump_worker()
        else:
            self.trump_worker = None  # pyrefly: ignore[bad-assignment]
        self.trader = self._container.trader_agent()
        self.risk_manager = RiskManagementAgent(
            components.llm_client,
            portfolio_var_calculator=components.portfolio_var_calculator,
            portfolio_var_config=components.portfolio_var_config,
            position_sizing_config=components.position_sizing_config,
        )

    def get_default_strategy(self) -> EnsembleStrategy | MomentumStrategy:
        """Return the default trading strategy."""
        return self._default_strategy

    def get_container(self) -> AppContainer:
        """Return the DI container."""
        return self._container

    def get_target_allocation(self, symbol: str) -> float | None:
        """Return target allocation for symbol, or None if not configured."""
        return self._target_allocations.get(symbol) if self._target_allocations else None
