"""Instrumented analysis pipeline with metrics collection."""

import time
from typing import Any

from loguru import logger

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
from src.workflows.models.strategy import StrategySelectionInput, StrategySelectionOutput
from src.workflows.stages import analysis, data_fetch, decision, execution, risk, strategy_selection
from src.workflows.types import TradingWorkflowResult, WorkflowExtraContext


async def run_instrumented_analysis(  # noqa: PLR0913
    workflow: Any,  # noqa: ANN401
    symbol: str,
    period_days: int,
    trading_session: TradingSession,
    collector: ExecutionMetricsCollector | None,
    extra_context: WorkflowExtraContext | None = None,
) -> TradingWorkflowResult:
    """Run analysis pipeline with optional metrics instrumentation.

    Args:
        workflow: TradingWorkflow instance
        symbol: Stock ticker symbol
        period_days: Days of historical data
        trading_session: Trading session type (REGULAR or PRE_MARKET)
        collector: Optional metrics collector
        extra_context: Optional context with degradation_context, enable_multi_timeframe, etc
    """
    from src.daemon.degradation import DegradationTier

    ctx = extra_context or WorkflowExtraContext()
    degradation_context = ctx.degradation_context

    # Check if halted
    if degradation_context and degradation_context.tier == DegradationTier.HALTED:
        msg = f"Analysis halted: {degradation_context.halt_reason}"
        raise RuntimeError(msg)

    enable_multi_timeframe = bool(ctx.enable_multi_timeframe)

    # Stage 1: Fetch data
    start = time.perf_counter()
    data_output = await data_fetch.fetch_data(
        symbol,
        period_days,
        trading_session,
        workflow.market_fetcher,
        workflow.news_fetcher,
        enable_multi_timeframe=enable_multi_timeframe,
        trump_mode=workflow.trump_mode,
        trump_fetcher=workflow.trump_fetcher,
    )
    _record_stage(collector, "fetch_data", start)

    # Stage 2: Fetch account info
    start = time.perf_counter()
    account_output = await data_fetch.fetch_account_info(workflow.broker)
    _record_stage(collector, "fetch_account_info", start)

    # Stage 3: Select strategy
    start = time.perf_counter()
    strategy_input = StrategySelectionInput(symbol=symbol, market_data=data_output.market_data)
    strategy_output = await strategy_selection.select_strategy(
        strategy_input,
        workflow.meta_agent,
        workflow._default_strategy,  # noqa: SLF001
        workflow.use_ensemble,
        collector,
    )
    _record_stage(collector, "strategy_selection", start)

    # Stage 4: Validate strategy with backtest
    start = time.perf_counter()
    backtest_output = await strategy_selection.validate_strategy_with_backtest(
        symbol,
        strategy_output.strategy_instance,
        strategy_output.strategy_name,
        strategy_input,
        workflow.pre_trade_backtest_config,
        workflow.vectorbt_runner,
        collector,
    )
    _record_stage(collector, "backtest_validation", start)

    # Create TechnicalAnalyst with selected strategy
    if workflow._container:  # noqa: SLF001
        technical_analyst = workflow._container.technical_analyst()(  # noqa: SLF001
            strategy_output.strategy_instance
        )
    else:
        from src.agents.technical import TechnicalAnalyst

        technical_analyst = TechnicalAnalyst(workflow.llm_client, strategy_output.strategy_instance)

    # Stage 5: Run analyses
    start = time.perf_counter()
    analysis_input = AnalysisInput(
        symbol=symbol,
        market_data=data_output.market_data,
        news_articles=data_output.news_articles,
        trump_posts=data_output.trump_posts,
        enable_multi_timeframe=enable_multi_timeframe,
    )
    analysis_output = await analysis.run_analyses(
        analysis_input,
        technical_analyst,
        workflow.sentiment_analyst,
        workflow.news_analyst,
        workflow.fundamental_analyst,
        workflow.comparative_analyst,
        workflow.web_researcher,
        workflow.social_analyst,
        workflow.bullish_researcher,
        workflow.bearish_researcher,
        workflow.trump_mode,
        workflow.trump_analyst,
        collector,
    )
    _record_stage(collector, "analyses", start)

    # Stage 6: Make decision
    start = time.perf_counter()
    decision_context = DecisionContext(
        sector_rotation=ctx.sector_rotation_context,
        earnings=ctx.earnings_context,
        peer_analysis=ctx.peer_analysis_context,
        game_plan=ctx.game_plan_context,
        position=ctx.position_context,
    )
    decision_input = DecisionInput(
        symbol=symbol,
        technical=analysis_output.technical_analysis,
        sentiment=analysis_output.sentiment_analysis,
        news=analysis_output.news_analysis,
        bullish=analysis_output.bullish_research,
        bearish=analysis_output.bearish_research,
        fundamental=analysis_output.fundamental_analysis,
        comparative=analysis_output.comparative_analysis,
        trump=analysis_output.trump_analysis,
        account_info=account_output.account_info,
        context=decision_context,
        backtest_validation=backtest_output.backtest_validation,
        degradation_context=degradation_context,
    )
    decision_output = await decision.make_decision(decision_input, workflow.trader, collector)
    _record_stage(collector, "decision", start)

    # Stage 7: Assess risk
    start = time.perf_counter()
    # Get target weight from allocations if available
    target_weight = (
        workflow._target_allocations.get(symbol)  # noqa: SLF001
        if workflow._target_allocations  # noqa: SLF001
        else None
    )
    risk_input = RiskAssessmentInput(
        symbol=symbol,
        market_data=data_output.market_data,
        final_decision=decision_output.final_decision,
        account_info=account_output.account_info,
        broker_positions=account_output.broker_positions,
        portfolio_value=account_output.portfolio_value,
        target_portfolio_weight=target_weight,
        backtest_validation=backtest_output.backtest_validation,
        degradation_context=degradation_context,
        broker_api_failed=account_output.broker_api_failed,
    )
    risk_output = await risk.assess_risk(risk_input, workflow.risk_manager)
    _record_stage(collector, "risk_assessment", start)

    # Notify if trade rejected by risk gate (only during regular hours when trades can execute)
    if (
        risk_output.risk_assessment
        and decision_output.final_decision
        and not risk_output.risk_assessment.validation.approved
        and decision_output.final_decision.action != Signal.HOLD
        and workflow.notification_service
        and trading_session == TradingSession.REGULAR
    ):
        await execution.notify_trade_execution(
            symbol,
            decision_output.final_decision,
            risk_output.risk_assessment,
            workflow.notification_service,
        )

    # Stage 8: Execute trade
    execution_output = None
    if (
        workflow.broker
        and risk_output.risk_assessment
        and decision_output.final_decision
        and risk_output.risk_assessment.validation.approved
        and decision_output.final_decision.action != Signal.HOLD
    ):
        execution_input = TradeExecutionInput(
            symbol=symbol,
            final_decision=decision_output.final_decision,
            risk_assessment=risk_output.risk_assessment,
            trading_session=trading_session,
        )
        execution_output = await execution.execute_trade(execution_input, workflow.broker)

    # Log final result
    logger.info(
        f"Workflow complete: {decision_output.final_decision.action.value} "
        f"(confidence={decision_output.final_decision.confidence:.2f}, "
        f"risk_approved={risk_output.risk_assessment.validation.approved})"
    )

    # Build result
    return await _build_and_persist_result(
        workflow,
        symbol,
        data_output,
        account_output,
        strategy_output,
        backtest_output,
        analysis_output,
        decision_output,
        risk_output,
        execution_output,
        decision_context,
        degradation_context,
        target_weight,
        trading_session,
        collector,
    )


async def _build_and_persist_result(  # noqa: PLR0913
    workflow: Any,  # noqa: ANN401
    symbol: str,
    data_output: FetchDataOutput,
    account_output: AccountInfoOutput,
    strategy_output: StrategySelectionOutput,
    backtest_output: BacktestValidationOutput,
    analysis_output: AnalysisOutput,
    decision_output: DecisionOutput,
    risk_output: RiskAssessmentOutput,
    execution_output: TradeExecutionOutput | None,
    decision_context: DecisionContext,
    degradation_context: Any,  # noqa: ANN401
    target_weight: float | None,  # noqa: ARG001
    trading_session: TradingSession,
    collector: ExecutionMetricsCollector | None,
) -> TradingWorkflowResult:
    """Build workflow result and persist metrics/snapshots.

    Args:
        workflow: TradingWorkflow instance
        symbol: Stock ticker
        data_output: Data fetch output
        account_output: Account info output
        strategy_output: Strategy selection output
        backtest_output: Backtest validation output
        analysis_output: Analysis output
        decision_output: Decision output
        risk_output: Risk assessment output
        execution_output: Execution output
        decision_context: Decision context
        degradation_context: Degradation context
        target_weight: Target portfolio weight
        trading_session: Trading session type
        collector: Optional metrics collector
    """
    from src.metrics.execution import persist_jsonl
    from src.metrics.tracker import DatabaseMetricsTracker

    execution_metrics = collector.finalize() if collector else None

    # Extract degradation fields
    degradation_tier = degradation_context.tier.value if degradation_context else None
    degradation_confidence_penalty = (
        (1 - degradation_context.confidence_adjustment) if degradation_context else None
    )

    # Aggregate warnings
    all_warnings = []
    all_warnings.extend(data_output.warnings)
    all_warnings.extend(account_output.warnings)
    all_warnings.extend(backtest_output.warnings)
    all_warnings.extend(analysis_output.warnings)
    if execution_output:
        all_warnings.extend(execution_output.warnings)

    result = TradingWorkflowResult(
        symbol=symbol,
        trading_session=trading_session,
        technical=analysis_output.technical_analysis,
        sentiment=analysis_output.sentiment_analysis,
        news=analysis_output.news_analysis,
        trump=analysis_output.trump_analysis,
        fundamental=analysis_output.fundamental_analysis,
        comparative=analysis_output.comparative_analysis,
        web_research=analysis_output.web_research,
        social_sentiment=analysis_output.social_sentiment_analysis,
        bullish=analysis_output.bullish_research,
        bearish=analysis_output.bearish_research,
        decision=decision_output.final_decision,
        risk=risk_output.risk_assessment,
        order=execution_output.order_status if execution_output else None,
        regime=strategy_output.regime_analysis,
        strategy_used=strategy_output.strategy_name,
        warnings=all_warnings,
        earnings_context=decision_context.earnings,
        peer_analysis_context=decision_context.peer_analysis,
        execution_metrics=execution_metrics,
        backtest_validation=backtest_output.backtest_validation,
        degradation_tier=degradation_tier,
        degradation_confidence_penalty=degradation_confidence_penalty,
    )

    if execution_metrics:
        try:
            persist_jsonl(execution_metrics)
        except Exception as e:
            logger.error(f"Failed to persist execution metrics (continuing): {e}")

    if workflow.metrics_tracker:
        try:
            is_paper = workflow.broker.paper if workflow.broker else True
            if isinstance(workflow.metrics_tracker, DatabaseMetricsTracker):
                await workflow.metrics_tracker.record_decision_async(
                    result, strategy_name=strategy_output.strategy_name, is_paper_trade=is_paper
                )
            else:
                workflow.metrics_tracker.record_decision(
                    result, strategy_name=strategy_output.strategy_name, is_paper_trade=is_paper
                )
        except Exception as e:
            logger.error(f"Failed to record metrics (continuing): {e}")

    if (
        workflow.snapshot_on_trade
        and workflow.snapshot_repository
        and risk_output.risk_assessment
        and decision_output.final_decision
        and risk_output.risk_assessment.validation.approved
        and decision_output.final_decision.action != Signal.HOLD
    ):
        await execution.create_portfolio_snapshot(
            symbol,
            account_output.account_info,
            workflow.snapshot_repository,
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
