"""Instrumented analysis pipeline with metrics collection."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sized
from typing import TYPE_CHECKING, cast

from loguru import logger

from src.agents.supervisor.models import TradeApprovalContext
from src.metrics.execution import ExecutionMetricsCollector
from src.strategies.session import TradingSession
from src.strategies.signal import Signal
from src.workflows.models.account import AccountInfoOutput
from src.workflows.models.analysis import AnalysisInput, AnalysisOutput
from src.workflows.models.backtest import BacktestValidationOutput
from src.workflows.models.data_fetch import FetchDataOutput
from src.workflows.models.decision import DecisionContext, DecisionInput, DecisionOutput
from src.workflows.models.execution import TradeExecutionInput, TradeExecutionOutput
from src.workflows.models.risk import RiskAssessmentInput, RiskAssessmentOutput
from src.workflows.models.risk_validation import RiskValidationInput, RiskValidationOutput
from src.workflows.models.strategy import StrategySelectionInput, StrategySelectionOutput
from src.workflows.stages import (
    data_fetch,
    decision,
    execution,
    risk,
    strategy_selection,
    supervised_analysis,
)
from src.workflows.stages.risk_validation import validate_analyses_stage
from src.workflows.types import TradingWorkflowResult, WorkflowExtraContext

if TYPE_CHECKING:
    from src.agents.supervisor.agent import SupervisorWorkflow
    from src.daemon.degradation import DegradationContext
    from src.data.broker import BrokerPosition
    from src.metrics.execution import WorkflowExecutionMetrics
    from src.workflows.orchestrator import TradingWorkflow


class AnalysisRequestParams:
    """Bundle of analysis request parameters."""

    def __init__(
        self, period_days: int, trading_session: TradingSession, extra_context: WorkflowExtraContext | None
    ) -> None:
        """Initialize request parameters.

        Args:
            period_days: Days of historical data
            trading_session: Trading session type
            extra_context: Optional workflow context
        """
        self.period_days = period_days
        self.trading_session = trading_session
        self.extra_context = extra_context


class AnalysisRequest:
    """Request for instrumented analysis."""

    def __init__(
        self,
        workflow: TradingWorkflow,
        symbol: str,
        params: AnalysisRequestParams,
        collector: ExecutionMetricsCollector | None = None,
    ) -> None:
        """Initialize analysis request.

        Args:
            workflow: TradingWorkflow instance
            symbol: Stock ticker symbol
            params: Request parameters bundle
            collector: Optional metrics collector
        """
        self.workflow = workflow
        self.symbol = symbol
        self.period_days = params.period_days
        self.trading_session = params.trading_session
        self.extra_context = params.extra_context
        self.collector = collector


class _ExecutionContext:
    """Bundle of common execution parameters."""

    def __init__(
        self,
        workflow: TradingWorkflow,
        symbol: str,
        trading_session: TradingSession,
        collector: ExecutionMetricsCollector | None,
        extra_context: WorkflowExtraContext | None = None,
    ) -> None:
        self.workflow = workflow
        self.symbol = symbol
        self.trading_session = trading_session
        self.collector = collector
        self.extra_context = extra_context


class _ContextBundle:
    """Bundle of decision and degradation contexts."""

    def __init__(
        self, decision_context: DecisionContext, degradation_context: DegradationContext | None
    ) -> None:
        self.decision_context = decision_context
        self.degradation_context = degradation_context


class _PreparationResult:
    """Bundle of preparation stage outputs."""

    def __init__(
        self,
        data_output: FetchDataOutput,
        account_output: AccountInfoOutput,
        strategy_output: StrategySelectionOutput,
        backtest_output: BacktestValidationOutput,
    ) -> None:
        self.data_output = data_output
        self.account_output = account_output
        self.strategy_output = strategy_output
        self.backtest_output = backtest_output


class _AnalysisResult:
    """Bundle of analysis stage outputs."""

    def __init__(
        self,
        analysis_output: AnalysisOutput,
        validation_output: RiskValidationOutput | None,
        supervisor_routing: object | None = None,
    ) -> None:
        self.analysis_output = analysis_output
        self.validation_output = validation_output
        self.supervisor_routing = supervisor_routing


class _ExecutionResult:
    """Bundle of execution stage outputs."""

    def __init__(
        self,
        decision_output: DecisionOutput,
        risk_output: RiskAssessmentOutput,
        execution_output: TradeExecutionOutput | None,
    ) -> None:
        self.decision_output = decision_output
        self.risk_output = risk_output
        self.execution_output = execution_output


async def run_instrumented_analysis(request: AnalysisRequest) -> TradingWorkflowResult:
    """Run analysis pipeline with optional metrics instrumentation.

    Args:
        request: Analysis request with all parameters
    """
    from src.daemon.degradation import DegradationTier

    ctx = request.extra_context or WorkflowExtraContext()
    degradation_context = ctx.degradation_context

    # Check if halted
    if degradation_context and degradation_context.tier == DegradationTier.HALTED:
        msg = f"Analysis halted: {degradation_context.halt_reason}"
        raise RuntimeError(msg)

    enable_multi_timeframe = bool(ctx.enable_multi_timeframe)

    exec_ctx = _ExecutionContext(
        request.workflow, request.symbol, request.trading_session, request.collector, ctx
    )
    context_bundle = _ContextBundle(
        decision_context=DecisionContext(
            sector_rotation=ctx.sector_rotation_context,
            earnings=ctx.earnings_context,
            peer_analysis=ctx.peer_analysis_context,
            game_plan=ctx.game_plan_context,
            position=ctx.position_context,
        ),
        degradation_context=degradation_context,
    )

    # Stage 1-4: Fetch data and prepare strategy
    prep_result = await _fetch_and_prepare_strategy(exec_ctx, request.period_days, enable_multi_timeframe)

    # Stage 5-5.5: Run analyses and validation
    analysis_result = await _run_analyses_with_validation(
        exec_ctx, prep_result, enable_multi_timeframe, context_bundle.degradation_context
    )

    # Stage 6-8: Make decision, assess risk, and execute
    execution_result = await _make_decision_and_execute(
        exec_ctx, prep_result, analysis_result, context_bundle
    )

    # Log final result
    logger.info(
        f"Workflow complete: {execution_result.decision_output.final_decision.action.value} "
        f"(confidence={execution_result.decision_output.final_decision.confidence:.2f}, "
        f"risk_approved={execution_result.risk_output.risk_assessment.validation.approved})"
    )

    # Build result
    return await _build_and_persist_result(
        exec_ctx, prep_result, analysis_result, execution_result, context_bundle
    )


async def _fetch_and_prepare_strategy(
    ctx: _ExecutionContext, period_days: int, enable_multi_timeframe: bool
) -> _PreparationResult:
    """Fetch data and prepare strategy (stages 1-4).

    Args:
        ctx: Execution context
        period_days: Historical data period
        enable_multi_timeframe: Whether to enable multi-timeframe analysis

    Returns:
        Preparation result bundle
    """
    from src.workflows.stages.data_fetch import DataFetchConfig

    # Stage 1: Fetch data
    start = time.perf_counter()
    data_output = await data_fetch.fetch_data(
        ctx.symbol,
        period_days,
        ctx.trading_session,
        config=DataFetchConfig(
            market_fetcher=ctx.workflow.market_fetcher,
            news_fetcher=ctx.workflow.news_fetcher,
            enable_multi_timeframe=enable_multi_timeframe,
            trump_mode=ctx.workflow.trump_mode,
            trump_fetcher=ctx.workflow.trump_fetcher,
            web_search_fetcher=ctx.workflow.web_search_fetcher,
        ),
    )
    _record_stage(ctx.collector, "fetch_data", start)

    # Stage 2: Fetch account info
    start = time.perf_counter()
    account_output = await data_fetch.fetch_account_info(ctx.workflow.broker)
    _record_stage(ctx.collector, "fetch_account_info", start)

    # Stage 3: Select strategy
    start = time.perf_counter()
    strategy_input = StrategySelectionInput(symbol=ctx.symbol, market_data=data_output.market_data)
    strategy_output = await strategy_selection.select_strategy(
        strategy_input,
        ctx.workflow.meta_agent,
        ctx.workflow.get_default_strategy(),
        ctx.workflow.use_ensemble,
        ctx.collector,
    )
    _record_stage(ctx.collector, "strategy_selection", start)

    # Stage 4: Validate strategy with backtest
    start = time.perf_counter()
    backtest_output = await strategy_selection.validate_strategy_with_backtest(
        ctx.symbol,
        strategy_output.strategy_instance,
        strategy_output.strategy_name,
        strategy_input,
        ctx.workflow.pre_trade_backtest_config,
        ctx.workflow.vectorbt_runner,
        ctx.collector,
    )
    _record_stage(ctx.collector, "backtest_validation", start)

    return _PreparationResult(data_output, account_output, strategy_output, backtest_output)


def _check_owns_position(positions: dict[str, BrokerPosition] | None, symbol: str) -> bool:
    """Check if currently own position in symbol."""
    if not positions:
        return False
    return symbol in positions


def _get_market_data_length(market_data: object) -> int:
    """Get length of market data for routing decisions.

    Args:
        market_data: OHLCV DataFrame or multi-timeframe data

    Returns:
        Number of rows in market data, 0 if None
    """
    from src.strategies.timeframe import MultiTimeframeData

    if market_data is None:
        return 0
    if isinstance(market_data, MultiTimeframeData):
        if not market_data.timeframes:
            return 0
        return min(len(df) for df in market_data.timeframes.values())
    if isinstance(market_data, Sized):
        return len(market_data)
    return 0


def _build_position_pnl_context(extra_context: WorkflowExtraContext | None) -> object | None:
    """Build PositionPnLContext from extra context position data.

    Args:
        extra_context: Workflow extra context

    Returns:
        PositionPnLContext or None
    """
    from src.agents.supervisor.models import PositionPnLContext

    if not extra_context or not extra_context.position_context:
        return None

    pos = extra_context.position_context
    if not pos.get("has_position"):
        return None

    return PositionPnLContext(
        entry_price=float(str(pos.get("entry_price", 0.0))),
        unrealized_pnl_percent=float(str(pos.get("unrealized_pnl_percent", 0.0))),
        days_held=int(str(pos.get("days_held", 0))),
        current_qty=float(str(pos.get("current_qty", 0.0))),
    )


def _build_portfolio_summary(
    broker_positions: dict[str, BrokerPosition] | None,
    portfolio_value: float | None,
) -> object | None:
    """Build PortfolioSummary from broker positions.

    Args:
        broker_positions: Dict of symbol -> BrokerPosition
        portfolio_value: Total portfolio value

    Returns:
        PortfolioSummary or None
    """
    from src.agents.supervisor.models import PortfolioSummary

    if not broker_positions or not portfolio_value or portfolio_value <= 0:
        return None

    total_exposure = 0.0
    total_unrealized_pnl = 0.0
    biggest_winner: str | None = None
    biggest_winner_pnl = -float("inf")
    biggest_loser: str | None = None
    biggest_loser_pnl = float("inf")

    for symbol, pos in broker_positions.items():
        total_exposure += pos.market_value
        total_unrealized_pnl += pos.unrealized_pnl

        cost_basis = pos.avg_entry_price * pos.qty
        pnl_pct = (pos.unrealized_pnl / cost_basis * 100) if cost_basis > 0 else 0.0
        if pnl_pct > biggest_winner_pnl:
            biggest_winner_pnl = pnl_pct
            biggest_winner = symbol
        if pnl_pct < biggest_loser_pnl:
            biggest_loser_pnl = pnl_pct
            biggest_loser = symbol

    num_positions = len(broker_positions)
    # With only one position it would be both winner and loser — avoid misleading data
    if num_positions <= 1:
        biggest_winner = None
        biggest_winner_pnl = 0.0
        biggest_loser = None
        biggest_loser_pnl = 0.0

    return PortfolioSummary(
        total_positions=num_positions,
        total_exposure_percent=(total_exposure / portfolio_value) * 100,
        portfolio_pnl_percent=(total_unrealized_pnl / portfolio_value) * 100,
        biggest_winner=biggest_winner,
        biggest_winner_pnl_percent=biggest_winner_pnl if biggest_winner else 0.0,
        biggest_loser=biggest_loser,
        biggest_loser_pnl_percent=biggest_loser_pnl if biggest_loser else 0.0,
    )


async def _run_analyses_with_validation(
    ctx: _ExecutionContext,
    prep_result: _PreparationResult,
    enable_multi_timeframe: bool,
    degradation_context: DegradationContext | None,
) -> _AnalysisResult:
    """Run analyses and validation (stages 5-5.5).

    Args:
        ctx: Execution context
        prep_result: Preparation result from previous stages
        enable_multi_timeframe: Whether multi-timeframe is enabled
        degradation_context: Degradation context

    Returns:
        Analysis result bundle
    """
    from src.agents.supervisor.models import PlanningContext

    # Stage 5: Run analyses
    start = time.perf_counter()
    analysis_input = AnalysisInput(
        symbol=ctx.symbol,
        market_data=prep_result.data_output.market_data,
        news_articles=prep_result.data_output.news_articles,
        trump_posts=prep_result.data_output.trump_posts,
        enable_multi_timeframe=enable_multi_timeframe,
        strategy=prep_result.strategy_output.strategy_instance,
    )

    routing_decision = None
    config = ctx.workflow.analysis_orchestrator_config

    if not (config and config.enable_supervisor_routing):
        msg = (
            "Supervisor routing is required for analysis, but either "
            "`workflow.analysis_orchestrator_config` is missing or "
            "`analysis_orchestrator_config.enable_supervisor_routing` is False. "
            "Configure `workflow.analysis_orchestrator_config` and set "
            "`enable_supervisor_routing = True`; non-supervisor routing is no longer supported."
        )
        raise RuntimeError(msg)

    # Supervisor-driven conditional execution
    start_planning = time.perf_counter()

    from src.strategies.regime import MarketRegime

    market_data_rows = _get_market_data_length(prep_result.data_output.market_data)
    is_high_volatility = (
        prep_result.strategy_output.regime_analysis.regime == MarketRegime.HIGH_VOLATILITY
        if prep_result.strategy_output.regime_analysis
        else False
    )

    planning_context = PlanningContext(
        symbol=ctx.symbol,
        regime=prep_result.strategy_output.regime_analysis,
        trading_session=ctx.trading_session,
        owns_position=_check_owns_position(prep_result.account_output.broker_positions, ctx.symbol),
        news_count=len(prep_result.data_output.news_articles or []),
        fundamental_available=True,
        social_available=True,
        trump_count=len(prep_result.data_output.trump_posts or []),
        fundamental_rate_limit=False,
        time_budget_ms=config.worker_execution_timeout_ms,
        market_data_rows=market_data_rows,
        is_high_volatility=is_high_volatility,
        economic_risk=ctx.extra_context.economic_calendar_context if ctx.extra_context else None,
        options_flow=ctx.extra_context.options_flow_context if ctx.extra_context else None,
        position_pnl=_build_position_pnl_context(ctx.extra_context),
        portfolio_summary=_build_portfolio_summary(
            prep_result.account_output.broker_positions,
            prep_result.account_output.portfolio_value,
        ),
        portfolio_health_constraints=(
            ctx.extra_context.portfolio_health_context if ctx.extra_context else None
        ),
        social_sentiment=ctx.extra_context.social_sentiment_context if ctx.extra_context else None,
    )

    planning_fallback_used = False
    try:
        routing_decision = await asyncio.wait_for(
            ctx.workflow.supervisor.plan_analyses(planning_context, symbol=ctx.symbol),
            timeout=config.supervisor_planning_timeout_ms / 1000,
        )
    except TimeoutError:
        logger.warning("Supervisor planning timed out, using default routing")
        routing_decision = ctx.workflow.supervisor.default_routing(planning_context)
        planning_fallback_used = True

    _record_stage(ctx.collector, "supervisor_planning", start_planning)

    # Run supervised analyses
    supervisor_workflow = cast("SupervisorWorkflow", ctx.workflow)
    workflow_id = ctx.collector.workflow_id if ctx.collector else None
    analysis_output = await supervised_analysis.run_supervised_analyses(
        analysis_input,
        routing_decision,
        supervisor_workflow.technical_worker,
        supervisor_workflow.sentiment_worker,
        supervisor_workflow.news_worker,
        supervisor_workflow.fundamental_worker,
        supervisor_workflow.comparative_worker,
        supervisor_workflow.web_researcher,
        supervisor_workflow.social_worker,
        supervisor_workflow.bullish_researcher,
        supervisor_workflow.bearish_researcher,
        supervisor_workflow.trump_mode,
        supervisor_workflow.trump_worker,
        ctx.collector,
        timeout_ms=config.worker_execution_timeout_ms,
        workflow_id=workflow_id,
        event_bus=supervisor_workflow.event_bus,
        planning_fallback_used=planning_fallback_used,
    )

    _record_stage(ctx.collector, "analyses", start)

    # Stage 5.5: Validate analyses
    start_validation = time.perf_counter()
    validation_output: RiskValidationOutput | None = None
    if ctx.workflow.risk_validation_config.enabled:
        validation_input = RiskValidationInput(
            symbol=ctx.symbol,
            trading_session=ctx.trading_session,
            technical_analysis=analysis_output.technical_analysis,
            sentiment_analysis=analysis_output.sentiment_analysis,
            news_analysis=analysis_output.news_analysis,
            fundamental_analysis=analysis_output.fundamental_analysis,
            bullish_research=analysis_output.bullish_research,
            bearish_research=analysis_output.bearish_research,
            market_data=prep_result.data_output.market_data,
            degradation_context=degradation_context,
        )
        validation_output = validate_analyses_stage(validation_input, ctx.workflow.risk_validator)
        _record_stage(ctx.collector, "risk_validation", start_validation)

        if not validation_output.validation_result.approved:
            logger.warning(
                f"Risk validation WARNING for {ctx.symbol}: {validation_output.validation_result.warnings}"
            )

    return _AnalysisResult(analysis_output, validation_output, routing_decision)


def _build_approval_context(
    symbol: str,
    decision_output: DecisionOutput,
    risk_output: RiskAssessmentOutput,
    analysis_result: _AnalysisResult,
) -> TradeApprovalContext:
    """Build TradeApprovalContext from workflow outputs."""
    ao = analysis_result.analysis_output
    ra = risk_output.risk_assessment
    td = decision_output.final_decision
    tech = ao.technical_analysis
    sent = ao.sentiment_analysis
    news = ao.news_analysis
    bull = ao.bullish_research
    bear = ao.bearish_research

    return TradeApprovalContext(
        symbol=symbol,
        action=td.action,
        confidence=td.confidence,
        risk_level=ra.validation.risk_level,
        risk_score=ra.validation.risk_score,
        current_price=ra.current_price,
        recommended_shares=ra.position_sizing.recommended_shares,
        position_value=ra.position_sizing.position_value,
        stop_loss_price=ra.stop_loss.stop_loss_price,
        reward_risk_ratio=ra.reward_risk_ratio,
        decision_reasoning=td.reasoning,
        risk_warnings=ra.validation.warnings,
        technical_summary=(
            f"{tech.signal.value} | "
            f"RSI={tech.rsi if tech.rsi is not None else 'N/A'} | "
            f"MACD={tech.macd_hist if tech.macd_hist is not None else 'N/A'} | "
            f"{tech.interpretation}"
            if tech
            else None
        ),
        sentiment_summary=(
            f"{sent.overall_sentiment} (score={sent.sentiment_score:.2f}) | {sent.summary}" if sent else None
        ),
        news_summary=(f"{news.recommendation} | {news.impact_assessment}" if news else None),
        bullish_summary=(f"{bull.thesis} | " + "; ".join(bull.key_points[:2]) if bull else None),
        bearish_summary=(f"{bear.thesis} | " + "; ".join(bear.key_points[:2]) if bear else None),
    )


async def _make_decision_and_execute(
    ctx: _ExecutionContext,
    prep_result: _PreparationResult,
    analysis_result: _AnalysisResult,
    context_bundle: _ContextBundle,
) -> _ExecutionResult:
    """Make decision, assess risk, and execute trade (stages 6-8).

    Args:
        ctx: Execution context
        prep_result: Preparation result from earlier stages
        analysis_result: Analysis result from earlier stages
        context_bundle: Decision and degradation contexts

    Returns:
        Execution result bundle
    """
    # Stage 6: Make decision
    start = time.perf_counter()
    decision_input = DecisionInput(
        symbol=ctx.symbol,
        technical=analysis_result.analysis_output.technical_analysis,
        sentiment=analysis_result.analysis_output.sentiment_analysis,
        news=analysis_result.analysis_output.news_analysis,
        bullish=analysis_result.analysis_output.bullish_research,
        bearish=analysis_result.analysis_output.bearish_research,
        fundamental=analysis_result.analysis_output.fundamental_analysis,
        comparative=analysis_result.analysis_output.comparative_analysis,
        trump=analysis_result.analysis_output.trump_analysis,
        account_info=prep_result.account_output.account_info,
        context=context_bundle.decision_context,
        backtest_validation=prep_result.backtest_output.backtest_validation,
        degradation_context=context_bundle.degradation_context,
        validation_context=analysis_result.validation_output,
    )
    decision_output = await decision.make_decision(decision_input, ctx.workflow.trader, ctx.collector)
    _record_stage(ctx.collector, "decision", start)

    # Stage 7: Assess risk
    start = time.perf_counter()
    target_weight = ctx.workflow.get_target_allocation(ctx.symbol)
    risk_input = RiskAssessmentInput(
        symbol=ctx.symbol,
        market_data=prep_result.data_output.market_data,
        final_decision=decision_output.final_decision,
        account_info=prep_result.account_output.account_info,
        broker_positions=prep_result.account_output.broker_positions,
        portfolio_value=prep_result.account_output.portfolio_value,
        target_portfolio_weight=target_weight,
        backtest_validation=prep_result.backtest_output.backtest_validation,
        degradation_context=context_bundle.degradation_context,
        broker_api_failed=prep_result.account_output.broker_api_failed,
    )
    risk_output = await risk.assess_risk(risk_input, ctx.workflow.risk_manager)
    _record_stage(ctx.collector, "risk_assessment", start)

    # Notify if trade rejected (only during regular hours)
    if (
        risk_output.risk_assessment
        and decision_output.final_decision
        and not risk_output.risk_assessment.validation.approved
        and decision_output.final_decision.action != Signal.HOLD
        and ctx.workflow.notification_service
        and ctx.trading_session == TradingSession.REGULAR
    ):
        await execution.notify_trade_execution(
            ctx.symbol,
            decision_output.final_decision,
            risk_output.risk_assessment,
            ctx.workflow.notification_service,
        )

    # Stage 8: Execute trade
    execution_output = None
    if (
        ctx.workflow.broker
        and risk_output.risk_assessment
        and decision_output.final_decision
        and risk_output.risk_assessment.validation.approved
        and decision_output.final_decision.action != Signal.HOLD
    ):
        # Supervisor approval gate
        supervisor = getattr(ctx.workflow, "supervisor", None)
        if supervisor is not None:
            start = time.perf_counter()
            approval_ctx = _build_approval_context(ctx.symbol, decision_output, risk_output, analysis_result)
            approval = await supervisor.approve_trade(approval_ctx, symbol=ctx.symbol)
            _record_stage(ctx.collector, "supervisor_approval", start)
            if not approval.approved:
                logger.info(
                    f"Supervisor rejected {ctx.symbol} trade: {approval.reasoning} "
                    f"concerns={approval.key_concerns}"
                )
                if ctx.workflow.notification_service and ctx.trading_session == TradingSession.REGULAR:
                    await execution.notify_trade_execution(
                        ctx.symbol,
                        decision_output.final_decision,
                        risk_output.risk_assessment,
                        ctx.workflow.notification_service,
                    )
                return _ExecutionResult(decision_output, risk_output, None)

        execution_input = TradeExecutionInput(
            symbol=ctx.symbol,
            final_decision=decision_output.final_decision,
            risk_assessment=risk_output.risk_assessment,
            trading_session=ctx.trading_session,
        )
        execution_output = await execution.execute_trade(execution_input, ctx.workflow.broker)

    return _ExecutionResult(decision_output, risk_output, execution_output)


async def _persist_execution_metrics(execution_metrics: WorkflowExecutionMetrics) -> None:
    """Persist execution metrics to database with JSONL fallback.

    Args:
        execution_metrics: Metrics to persist
    """
    from src.database.connection import get_session
    from src.database.engine import MissingDatabaseURLError
    from src.database.repositories.workflow_execution_metrics import WorkflowExecutionMetricsRepository
    from src.metrics.execution import persist_jsonl

    try:
        async with get_session() as session:
            repo = WorkflowExecutionMetricsRepository(session)
            await repo.create(execution_metrics)
            logger.debug(f"Persisted execution metrics to database: {execution_metrics.workflow_id}")
    except MissingDatabaseURLError:
        logger.debug("Database not configured, falling back to JSONL for execution metrics")
        try:
            persist_jsonl(execution_metrics)
        except Exception as e:
            logger.opt(exception=True).error(f"Failed to persist execution metrics to JSONL: {e}")
    except Exception as e:
        logger.opt(exception=True).warning(f"Failed to persist execution metrics to database: {e}")
        try:
            persist_jsonl(execution_metrics)
            logger.debug(f"Persisted execution metrics to JSONL fallback: {execution_metrics.workflow_id}")
        except Exception as jsonl_error:
            logger.opt(exception=True).error(
                f"Failed to persist execution metrics to JSONL fallback: {jsonl_error}"
            )


async def _build_and_persist_result(
    ctx: _ExecutionContext,
    prep_result: _PreparationResult,
    analysis_result: _AnalysisResult,
    execution_result: _ExecutionResult,
    context_bundle: _ContextBundle,
) -> TradingWorkflowResult:
    """Build workflow result and persist metrics/snapshots.

    Args:
        ctx: Execution context
        prep_result: Preparation result from earlier stages
        analysis_result: Analysis result from earlier stages
        execution_result: Execution result from earlier stages
        context_bundle: Decision and degradation contexts
    """
    from src.metrics.db_tracker import DatabaseMetricsTracker

    execution_metrics = ctx.collector.finalize() if ctx.collector else None

    degradation_tier = (
        context_bundle.degradation_context.tier.value if context_bundle.degradation_context else None
    )
    degradation_confidence_penalty = (
        (1 - context_bundle.degradation_context.confidence_adjustment)
        if context_bundle.degradation_context
        else None
    )

    all_warnings = []
    all_warnings.extend(prep_result.data_output.warnings)
    all_warnings.extend(prep_result.account_output.warnings)
    all_warnings.extend(prep_result.backtest_output.warnings)
    all_warnings.extend(analysis_result.analysis_output.warnings)
    if execution_result.execution_output:
        all_warnings.extend(execution_result.execution_output.warnings)

    result = TradingWorkflowResult(
        symbol=ctx.symbol,
        trading_session=ctx.trading_session,
        technical=analysis_result.analysis_output.technical_analysis,
        sentiment=analysis_result.analysis_output.sentiment_analysis,
        news=analysis_result.analysis_output.news_analysis,
        trump=analysis_result.analysis_output.trump_analysis,
        fundamental=analysis_result.analysis_output.fundamental_analysis,
        comparative=analysis_result.analysis_output.comparative_analysis,
        web_research=analysis_result.analysis_output.web_research,
        social_sentiment=analysis_result.analysis_output.social_sentiment_analysis,
        bullish=analysis_result.analysis_output.bullish_research,
        bearish=analysis_result.analysis_output.bearish_research,
        decision=execution_result.decision_output.final_decision,
        risk=execution_result.risk_output.risk_assessment,
        order=execution_result.execution_output.order_status if execution_result.execution_output else None,
        regime=prep_result.strategy_output.regime_analysis,
        strategy_used=prep_result.strategy_output.strategy_name,
        warnings=all_warnings,
        earnings_context=context_bundle.decision_context.earnings,
        peer_analysis_context=context_bundle.decision_context.peer_analysis,
        execution_metrics=execution_metrics,
        backtest_validation=prep_result.backtest_output.backtest_validation,
        degradation_tier=degradation_tier,
        degradation_confidence_penalty=degradation_confidence_penalty,
        supervisor_routing=analysis_result.supervisor_routing,
    )

    if execution_metrics:
        await _persist_execution_metrics(execution_metrics)

    if ctx.workflow.metrics_tracker:
        try:
            is_paper = ctx.workflow.broker.paper if ctx.workflow.broker else True
            if isinstance(ctx.workflow.metrics_tracker, DatabaseMetricsTracker):
                await ctx.workflow.metrics_tracker.record_decision_async(
                    result, strategy_name=prep_result.strategy_output.strategy_name, is_paper_trade=is_paper
                )
            else:
                ctx.workflow.metrics_tracker.record_decision(
                    result, strategy_name=prep_result.strategy_output.strategy_name, is_paper_trade=is_paper
                )
        except Exception as e:
            logger.opt(exception=True).error(f"Failed to record metrics (continuing): {e}")

    if (
        ctx.workflow.snapshot_on_trade
        and ctx.workflow.snapshot_repository
        and execution_result.risk_output.risk_assessment
        and execution_result.decision_output.final_decision
        and execution_result.risk_output.risk_assessment.validation.approved
        and execution_result.decision_output.final_decision.action != Signal.HOLD
    ):
        await execution.create_portfolio_snapshot(
            ctx.symbol,
            prep_result.account_output.account_info,
            ctx.workflow.snapshot_repository,
        )

    return result


def _record_stage(
    collector: ExecutionMetricsCollector | None,
    stage: str,
    start: float,
) -> None:
    """Record pipeline stage timing if collector is active.

    Args:
        collector: Optional metrics collector
        stage: Stage name
        start: perf_counter start time
    """
    if collector:
        collector.record_pipeline_stage(stage, (time.perf_counter() - start) * 1000)
