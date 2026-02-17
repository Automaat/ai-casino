"""Supervised analysis stage implementation with conditional execution."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Coroutine
from typing import TYPE_CHECKING, Any, TypeVar, cast

from loguru import logger

from src.agents.supervisor.metrics import SupervisorMetricsCollector

if TYPE_CHECKING:
    import pandas as pd

    from src.data.news import NewsArticle
    from src.metrics.execution import ExecutionMetricsCollector
    from src.strategies.timeframe import MultiTimeframeData
    from src.workers.comparative import ComparativeWorker
    from src.workers.fundamental import FundamentalWorker
    from src.workers.news import NewsWorker
    from src.workers.sentiment import SentimentWorker
    from src.workers.social import SocialSentimentWorker
    from src.workers.technical import TechnicalWorker
    from src.workers.thesis_research import ThesisResearchWorker
    from src.workers.trump import TrumpWorker
    from src.workers.web_research import WebResearchWorker

from src.agents.supervisor.models import AnalysisRoutingDecision, AnalysisType
from src.workers.thesis_research import AnalysisInputs
from src.workflows.models.analysis import AnalysisInput, AnalysisOutput
from src.workflows.stages.strategy_selection import _timed_agent_call

T = TypeVar("T")


def _record_worker_starts(
    tasks: list[_WorkerTask], metrics_collector: SupervisorMetricsCollector | None
) -> None:
    """Record worker starts in metrics collector and attach completion callbacks."""
    if not metrics_collector:
        return
    for worker_task in tasks:
        is_required = worker_task.category == "required"
        metrics_collector.record_worker_start(worker_task.analysis_type.value, is_required)

        # Attach callback to record end time when task completes
        # Capture collector in closure to ensure it's not None
        def record_end_time(
            _task: asyncio.Task,
            worker_name: str = worker_task.analysis_type.value,
            collector: SupervisorMetricsCollector = metrics_collector,
        ) -> None:
            collector.record_worker_end_time(worker_name)

        worker_task.task.add_done_callback(record_end_time)


def _record_worker_result(
    worker_name: str,
    result: object,
    *,
    is_error: bool,
    metrics_collector: SupervisorMetricsCollector | None,
) -> None:
    """Record worker completion or error in metrics collector."""
    if not metrics_collector:
        return
    if is_error:
        metrics_collector.record_worker_error(worker_name, str(result))
    else:
        metrics_collector.record_worker_complete(worker_name, result)


class _WorkerTask:
    """Wrapper for worker task with metadata."""

    def __init__(
        self,
        analysis_type: AnalysisType,
        task: asyncio.Task[Any],
        category: str,
    ) -> None:
        self.analysis_type = analysis_type
        self.task = task
        self.category = category


def _create_worker_if_needed(
    analysis_type: AnalysisType,
    required: set[AnalysisType],
    optional: set[AnalysisType],
    coro: Coroutine[Any, Any, Any],
) -> _WorkerTask | None:
    """Create worker task if analysis is required or optional.

    Args:
        analysis_type: Type of analysis
        required: Set of required analysis types
        optional: Set of optional analysis types
        coro: Coroutine to execute

    Returns:
        WorkerTask if needed, None otherwise
    """
    if analysis_type not in required and analysis_type not in optional:
        return None
    category = "required" if analysis_type in required else "optional"
    task = asyncio.create_task(coro)
    return _WorkerTask(analysis_type, task, category)


def _validate_input_data(input_data: AnalysisInput) -> None:
    """Validate input data, raise on critical errors."""
    if input_data.market_data is None:
        msg = "market_data is None, cannot run analyses"
        raise ValueError(msg)
    if input_data.news_articles is None:
        logger.warning("news_articles is None; sentiment/news analyses may be skipped")


async def _execute_workers_with_gather(
    tasks: list[_WorkerTask],
    timeout_ms: int,
    metrics_collector: SupervisorMetricsCollector | None = None,
) -> dict[AnalysisType, Any]:
    """Execute workers using asyncio.gather with timeout and error handling.

    Args:
        tasks: List of worker tasks with metadata
        timeout_ms: Timeout in milliseconds
        metrics_collector: Optional supervisor metrics collector

    Returns:
        Dict mapping analysis type to result (None for failures)

    Raises:
        Exception: If a required worker fails
    """
    output: dict[AnalysisType, Any] = {}

    if not tasks:
        return output

    _record_worker_starts(tasks, metrics_collector)
    task_list = [t.task for t in tasks]

    # NOTE:
    # We intentionally use asyncio.gather(..., return_exceptions=True) here
    # instead of asyncio.TaskGroup (as used in analysis.py). This allows us
    # to collect exceptions as results so we can:
    #   * handle failures on a per-worker basis, and
    #   * distinguish between required and optional workers when deciding
    #     whether to propagate or log-and-suppress an error.
    # This per-task exception handling depends on the return_exceptions
    # behaviour and is the reason this function deviates from the TaskGroup
    # pattern used elsewhere in the codebase.
    start_time = time.perf_counter()
    success_count = 0
    error_count = 0

    try:
        gather_coro = asyncio.gather(*task_list, return_exceptions=True)
        results = await asyncio.wait_for(gather_coro, timeout=timeout_ms / 1000)
    except TimeoutError:
        logger.warning(f"Worker execution timed out after {timeout_ms}ms, cancelling tasks")
        if metrics_collector:
            metrics_collector.record_timeout()
        for worker_task in tasks:
            if not worker_task.task.done():
                worker_task.task.cancel()
        await asyncio.gather(*task_list, return_exceptions=True)
        raise
    finally:
        # Log worker completion summary even on timeout for observability
        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            f"Workers completed: {success_count}/{len(tasks)} successful, "
            f"{error_count} errors, duration {duration_ms:.0f}ms"
        )

    # Record worker completions and errors
    for worker_task, result in zip(tasks, results, strict=True):
        analysis_type = worker_task.analysis_type
        category = worker_task.category

        # Control-flow exceptions - propagate immediately
        if isinstance(result, (asyncio.CancelledError, KeyboardInterrupt)):
            raise result

        # Worker failures
        if isinstance(result, Exception):
            _record_worker_result(
                analysis_type.value, result, is_error=True, metrics_collector=metrics_collector
            )
            error_count += 1
            if category == "required":
                logger.opt(exception=result).error(f"Required worker {analysis_type.value} failed")
                raise result
            logger.opt(exception=result).warning(f"Optional worker {analysis_type.value} failed")
            output[analysis_type] = None
        else:
            _record_worker_result(
                analysis_type.value, result, is_error=False, metrics_collector=metrics_collector
            )
            success_count += 1
            output[analysis_type] = result

    return output


def _setup_workers_group1(
    input_data: AnalysisInput,
    required: set[AnalysisType],
    optional: set[AnalysisType],
    technical_worker: TechnicalWorker,
    sentiment_worker: SentimentWorker,
    news_worker: NewsWorker,
    fundamental_worker: FundamentalWorker,
    comparative_worker: ComparativeWorker,
    web_researcher: WebResearchWorker,
    social_worker: SocialSentimentWorker,
    trump_mode: bool,
    trump_worker: TrumpWorker | None,
    collector: ExecutionMetricsCollector | None,
) -> list[_WorkerTask]:
    """Setup worker tasks for group 1 analyses."""
    tasks: list[_WorkerTask] = []

    # Cast after validation - we already validated in run_supervised_analyses
    market_data = cast("pd.DataFrame | MultiTimeframeData", input_data.market_data)
    news_articles = cast("list[NewsArticle]", input_data.news_articles)
    trump_posts = input_data.trump_posts or None

    # Technical analysis
    worker = _create_worker_if_needed(
        AnalysisType.TECHNICAL,
        required,
        optional,
        _timed_agent_call(
            "technical",
            technical_worker.analyze(
                input_data.symbol,
                market_data,
                strategy=input_data.strategy,
                enable_multi_timeframe=input_data.enable_multi_timeframe,
            ),
            collector,
        ),
    )
    if worker:
        tasks.append(worker)

    # Sentiment analysis
    worker = _create_worker_if_needed(
        AnalysisType.SENTIMENT,
        required,
        optional,
        _timed_agent_call("sentiment", sentiment_worker.analyze(input_data.symbol, news_articles), collector),
    )
    if worker:
        tasks.append(worker)

    # News analysis
    worker = _create_worker_if_needed(
        AnalysisType.NEWS,
        required,
        optional,
        _timed_agent_call("news", news_worker.analyze(input_data.symbol, news_articles), collector),
    )
    if worker:
        tasks.append(worker)

    # Fundamental analysis
    worker = _create_worker_if_needed(
        AnalysisType.FUNDAMENTAL,
        required,
        optional,
        _timed_agent_call(
            "fundamental",
            fundamental_worker.analyze(input_data.symbol, input_data.get_current_price()),
            collector,
        ),
    )
    if worker:
        tasks.append(worker)

    # Comparative analysis
    worker = _create_worker_if_needed(
        AnalysisType.COMPARATIVE,
        required,
        optional,
        _timed_agent_call("comparative", comparative_worker.analyze(input_data.symbol), collector),
    )
    if worker:
        tasks.append(worker)

    # Web research
    worker = _create_worker_if_needed(
        AnalysisType.WEB_RESEARCH,
        required,
        optional,
        _timed_agent_call("web_research", web_researcher.research(input_data.symbol), collector),
    )
    if worker:
        tasks.append(worker)

    # Social sentiment
    worker = _create_worker_if_needed(
        AnalysisType.SOCIAL_SENTIMENT,
        required,
        optional,
        _timed_agent_call("social", social_worker.analyze(input_data.symbol), collector),
    )
    if worker:
        tasks.append(worker)

    # Trump analysis
    if trump_mode and trump_worker and trump_posts:
        worker = _create_worker_if_needed(
            AnalysisType.TRUMP,
            required,
            optional,
            _timed_agent_call("trump", trump_worker.analyze(trump_posts), collector),
        )
        if worker:
            tasks.append(worker)

    return tasks


async def _run_supervised_group1(
    input_data: AnalysisInput,
    routing_decision: AnalysisRoutingDecision,
    technical_worker: TechnicalWorker,
    sentiment_worker: SentimentWorker,
    news_worker: NewsWorker,
    fundamental_worker: FundamentalWorker,
    comparative_worker: ComparativeWorker,
    web_researcher: WebResearchWorker,
    social_worker: SocialSentimentWorker,
    trump_mode: bool,
    trump_worker: TrumpWorker | None,
    collector: ExecutionMetricsCollector | None,
    timeout_ms: int,
    metrics_collector: SupervisorMetricsCollector | None = None,
) -> dict[AnalysisType, Any]:
    """Run first group of analyses based on routing decision.

    Args:
        input_data: Analysis input
        routing_decision: Supervisor routing decision
        technical_worker: Technical worker
        sentiment_worker: Sentiment worker
        news_worker: News worker
        fundamental_worker: Fundamental worker
        comparative_worker: Comparative worker
        web_researcher: Web research worker
        social_worker: Social sentiment worker
        trump_mode: Enable Trump analysis
        trump_worker: Trump worker
        collector: Optional metrics collector
        timeout_ms: Timeout in milliseconds
        metrics_collector: Optional supervisor metrics collector

    Returns:
        Dict mapping analysis type to result
    """
    required = set(routing_decision.required_analyses)
    optional = set(routing_decision.optional_analyses)

    # Log skipped analyses
    for analysis_type, reason in routing_decision.skip_analyses.items():
        logger.info(f"Skipped {analysis_type.value}: {reason}")

    # Setup all worker tasks
    tasks = _setup_workers_group1(
        input_data,
        required,
        optional,
        technical_worker,
        sentiment_worker,
        news_worker,
        fundamental_worker,
        comparative_worker,
        web_researcher,
        social_worker,
        trump_mode,
        trump_worker,
        collector,
    )

    # Execute workers
    required_tasks = [t for t in tasks if t.category == "required"]
    optional_tasks = [t for t in tasks if t.category == "optional"]
    required_types = ", ".join([t.analysis_type.value for t in required_tasks]) if required_tasks else "none"
    optional_types = ", ".join([t.analysis_type.value for t in optional_tasks]) if optional_tasks else "none"

    logger.info(
        f"Launching group 1: {len(tasks)} workers "
        f"({len(required_tasks)} required: {required_types}, "
        f"{len(optional_tasks)} optional: {optional_types})"
    )

    return await _execute_workers_with_gather(tasks, timeout_ms, metrics_collector)


async def _run_supervised_research(
    symbol: str,
    results: dict[AnalysisType, Any],
    routing_decision: AnalysisRoutingDecision,
    bullish_researcher: ThesisResearchWorker,
    bearish_researcher: ThesisResearchWorker,
    collector: ExecutionMetricsCollector | None,
    timeout_ms: int,
    metrics_collector: SupervisorMetricsCollector | None = None,
) -> dict[AnalysisType, Any]:
    """Run research analyses based on routing decision.

    Args:
        symbol: Stock ticker
        results: Results from group 1 analyses
        routing_decision: Supervisor routing decision
        bullish_researcher: Bullish thesis research worker
        bearish_researcher: Bearish thesis research worker
        collector: Optional metrics collector
        timeout_ms: Timeout in milliseconds
        metrics_collector: Optional supervisor metrics collector

    Returns:
        Dict mapping analysis type to result
    """
    required = set(routing_decision.required_analyses)
    optional = set(routing_decision.optional_analyses)

    # Extract results
    technical = results.get(AnalysisType.TECHNICAL)
    sentiment = results.get(AnalysisType.SENTIMENT)
    news = results.get(AnalysisType.NEWS)
    fundamental = results.get(AnalysisType.FUNDAMENTAL)
    comparative = results.get(AnalysisType.COMPARATIVE)
    trump = results.get(AnalysisType.TRUMP)

    # Research requires all three core analyses; skip if any are missing
    if technical is None or sentiment is None or news is None:
        logger.warning(
            f"Skipping research for {symbol}: "
            "required analyses (technical/sentiment/news) not available"
        )
        return {}

    tasks: list[_WorkerTask] = []

    # Bullish research
    if AnalysisType.BULLISH_RESEARCH in required or AnalysisType.BULLISH_RESEARCH in optional:
        category = "required" if AnalysisType.BULLISH_RESEARCH in required else "optional"
        inputs = AnalysisInputs(
            technical=technical,
            sentiment=sentiment,
            news=news,
            fundamental=fundamental,
            comparative=comparative,
            trump_analysis=trump,
        )
        coro = _timed_agent_call(
            "bullish_researcher",
            bullish_researcher.analyze(symbol, inputs),
            collector,
        )
        task = asyncio.create_task(coro)
        tasks.append(_WorkerTask(AnalysisType.BULLISH_RESEARCH, task, category))

    # Bearish research
    if AnalysisType.BEARISH_RESEARCH in required or AnalysisType.BEARISH_RESEARCH in optional:
        category = "required" if AnalysisType.BEARISH_RESEARCH in required else "optional"
        inputs = AnalysisInputs(
            technical=technical,
            sentiment=sentiment,
            news=news,
            fundamental=fundamental,
            comparative=comparative,
            trump_analysis=trump,
        )
        coro = _timed_agent_call(
            "bearish_researcher",
            bearish_researcher.analyze(symbol, inputs),
            collector,
        )
        task = asyncio.create_task(coro)
        tasks.append(_WorkerTask(AnalysisType.BEARISH_RESEARCH, task, category))

    if not tasks:
        return {}

    required_tasks = [t for t in tasks if t.category == "required"]
    optional_tasks = [t for t in tasks if t.category == "optional"]
    required_types = ", ".join([t.analysis_type.value for t in required_tasks]) if required_tasks else "none"
    optional_types = ", ".join([t.analysis_type.value for t in optional_tasks]) if optional_tasks else "none"

    logger.info(
        f"Launching research: {len(tasks)} workers "
        f"({len(required_tasks)} required: {required_types}, "
        f"{len(optional_tasks)} optional: {optional_types})"
    )

    return await _execute_workers_with_gather(tasks, timeout_ms, metrics_collector)


async def _publish_routing_event(
    event_bus: object | None,
    workflow_id: str,
    symbol: str,
    routing_decision: AnalysisRoutingDecision,
) -> None:
    """Publish supervisor routing complete event.

    Args:
        event_bus: Optional event bus for real-time updates
        workflow_id: Workflow identifier
        symbol: Stock ticker symbol
        routing_decision: Supervisor routing decision
    """
    if not event_bus or not workflow_id:
        return

    from src.daemon.event_bus import DashboardEvent, EventBus, EventType

    if isinstance(event_bus, EventBus):
        try:
            routing_event = DashboardEvent(
                event_type=EventType.SUPERVISOR_ROUTING_COMPLETE,
                data={
                    "workflow_id": workflow_id,
                    "symbol": symbol,
                    "required_analyses": [a.value for a in routing_decision.required_analyses],
                    "optional_analyses": [a.value for a in routing_decision.optional_analyses],
                    "skip_analyses": {k.value: v for k, v in routing_decision.skip_analyses.items()},
                    "reasoning": routing_decision.reasoning,
                },
            )
            await event_bus.publish(routing_event)
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to publish routing event: {e}")


async def _publish_metrics_event(
    event_bus: object | None,
    workflow_id: str,
    symbol: str,
    metrics_collector: SupervisorMetricsCollector,
) -> None:
    """Publish supervisor metrics updated event.

    Args:
        event_bus: Optional event bus for real-time updates
        workflow_id: Workflow identifier
        symbol: Stock ticker symbol
        metrics_collector: Supervisor metrics collector
    """
    if not event_bus or not workflow_id:
        return

    from src.daemon.event_bus import DashboardEvent, EventBus, EventType

    if isinstance(event_bus, EventBus):
        try:
            metrics_event = DashboardEvent(
                event_type=EventType.SUPERVISOR_METRICS_UPDATED,
                data={
                    "workflow_id": workflow_id,
                    "symbol": symbol,
                    "total_workers": metrics_collector.total_workers,
                    "successful_workers": metrics_collector.successful_workers,
                    "failed_workers": metrics_collector.failed_workers,
                    "routing_decision_ms": metrics_collector.routing_decision_ms,
                    "group1_execution_ms": metrics_collector.group1_execution_ms,
                    "research_execution_ms": metrics_collector.research_execution_ms,
                    "parallel_efficiency_percent": metrics_collector.parallel_efficiency_percent,
                    "timeout_triggered": metrics_collector.timeout_triggered,
                },
            )
            await event_bus.publish(metrics_event)
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to publish metrics event: {e}")


def _build_output(
    results: dict[AnalysisType, Any],
    research_results: dict[AnalysisType, Any],
    warnings: list[str],
) -> AnalysisOutput:
    """Build final analysis output from results.

    Args:
        results: Results from group 1 analyses
        research_results: Results from research analyses
        warnings: Accumulated warnings

    Returns:
        AnalysisOutput with all analyses
    """
    return AnalysisOutput(
        technical_analysis=results.get(AnalysisType.TECHNICAL),
        sentiment_analysis=results.get(AnalysisType.SENTIMENT),
        news_analysis=results.get(AnalysisType.NEWS),
        trump_analysis=results.get(AnalysisType.TRUMP),
        fundamental_analysis=results.get(AnalysisType.FUNDAMENTAL),
        comparative_analysis=results.get(AnalysisType.COMPARATIVE),
        web_research=results.get(AnalysisType.WEB_RESEARCH),
        social_sentiment_analysis=results.get(AnalysisType.SOCIAL_SENTIMENT),
        bullish_research=research_results.get(AnalysisType.BULLISH_RESEARCH),
        bearish_research=research_results.get(AnalysisType.BEARISH_RESEARCH),
        warnings=warnings,
    )


async def run_supervised_analyses(
    input_data: AnalysisInput,
    routing_decision: AnalysisRoutingDecision,
    technical_worker: TechnicalWorker,
    sentiment_worker: SentimentWorker,
    news_worker: NewsWorker,
    fundamental_worker: FundamentalWorker,
    comparative_worker: ComparativeWorker,
    web_researcher: WebResearchWorker,
    social_worker: SocialSentimentWorker,
    bullish_researcher: ThesisResearchWorker,
    bearish_researcher: ThesisResearchWorker,
    trump_mode: bool,
    trump_worker: TrumpWorker | None,
    collector: ExecutionMetricsCollector | None = None,
    timeout_ms: int = 30000,
    workflow_id: str | None = None,
    event_bus: object | None = None,
    planning_fallback_used: bool = False,
) -> AnalysisOutput:
    """Execute analyses based on supervisor routing decision.

    Three categories:
    - Required (fail-fast on error)
    - Optional (graceful degradation)
    - Skipped (don't execute, log reasoning)

    Args:
        input_data: Analysis input with symbol, market data, news, trump posts
        routing_decision: Supervisor routing decision
        technical_worker: Technical worker with strategy from analysis_input.strategy
        sentiment_worker: Sentiment worker
        news_worker: News worker
        fundamental_worker: Fundamental worker
        comparative_worker: Comparative worker
        web_researcher: Web research worker
        social_worker: Social sentiment worker
        bullish_researcher: Bullish thesis research worker
        bearish_researcher: Bearish thesis research worker
        trump_mode: Enable Trump analysis
        trump_worker: Trump worker (required if trump_mode=True)
        collector: Optional metrics collector
        timeout_ms: Timeout for worker execution in milliseconds
        workflow_id: Optional workflow ID for supervisor metrics
        event_bus: Optional event bus for real-time updates
        planning_fallback_used: Whether supervisor planning used fallback routing

    Returns:
        AnalysisOutput with all analyses
    """
    warnings: list[str] = []
    _validate_input_data(input_data)

    if not input_data.news_articles:
        logger.warning(f"No news articles available for {input_data.symbol}, analyses may be degraded")
        warnings.append("No news articles available - sentiment and news analyses degraded")

    # Initialize supervisor metrics collector
    metrics_collector = None
    if workflow_id:
        metrics_collector = SupervisorMetricsCollector(workflow_id, input_data.symbol)
        metrics_collector.record_planning_start()
        metrics_collector.record_planning(routing_decision, fallback_used=planning_fallback_used)

    # Publish routing complete event
    await _publish_routing_event(event_bus, workflow_id or "", input_data.symbol, routing_decision)

    # Run group 1 analyses
    if metrics_collector:
        metrics_collector.record_group1_start()
    start = time.perf_counter()
    group1_results = await _run_supervised_group1(
        input_data,
        routing_decision,
        technical_worker,
        sentiment_worker,
        news_worker,
        fundamental_worker,
        comparative_worker,
        web_researcher,
        social_worker,
        trump_mode,
        trump_worker,
        collector,
        timeout_ms,
        metrics_collector,
    )
    group1_ms = (time.perf_counter() - start) * 1000
    if metrics_collector:
        metrics_collector.record_group1_complete()
    logger.debug(f"Group 1 analyses completed in {group1_ms:.0f}ms")

    # Run group 2 research
    if metrics_collector:
        metrics_collector.record_research_start()
    start = time.perf_counter()
    research_results = await _run_supervised_research(
        input_data.symbol,
        group1_results,
        routing_decision,
        bullish_researcher,
        bearish_researcher,
        collector,
        timeout_ms,
        metrics_collector,
    )
    research_ms = (time.perf_counter() - start) * 1000
    if metrics_collector:
        metrics_collector.record_research_complete()
    logger.debug(f"Research analyses completed in {research_ms:.0f}ms")

    # Save supervisor metrics
    if metrics_collector:
        from src.database.connection import get_session
        from src.database.engine import MissingDatabaseURLError
        from src.database.repositories.supervisor_metrics import SupervisorMetricsRepository

        # Create session inline for metrics persistence (per session management pattern)
        try:
            async with get_session() as session:
                repo = SupervisorMetricsRepository(session)
                await metrics_collector.save(repo)
                await _publish_metrics_event(
                    event_bus, workflow_id or "", input_data.symbol, metrics_collector
                )
                logger.debug(f"Persisted supervisor metrics to database: workflow_id={workflow_id}")
        except MissingDatabaseURLError:
            logger.debug("Database not configured, skipping supervisor metrics persistence")
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to persist supervisor metrics: {e}")

    return _build_output(group1_results, research_results, warnings)
