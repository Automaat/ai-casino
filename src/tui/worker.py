"""Isolated thread workers for running analysis without blocking Textual TUI.

Architecture:
- Workers run in dedicated threads with their own event loops
- Communication via thread-safe callbacks (Textual's post_message)
- No awaiting from Textual side - true fire-and-forget with message-based results
"""

import asyncio
import contextlib
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger

from src.tui.types import ProgressCallback

if TYPE_CHECKING:
    from src.agents.technical import TechnicalAnalyst
    from src.screening.analyzer import ScreeningAnalysis
    from src.screening.screener import ScreeningOutput
    from src.workflows.trading import TradingWorkflow

from src.models.torch_config import configure_torch_env

# Configure environment BEFORE any imports (critical for fd safety)
configure_torch_env()


# --- Type Definitions ---


@dataclass
class AnalysisJob:
    """Represents a running analysis job."""

    thread: threading.Thread
    cancelled: threading.Event
    symbol: str


@dataclass
class AnalysisParams:
    """Parameters for analysis thread."""

    symbol: str
    period_days: int
    progress_callback: "ProgressCallback | None"
    result_callback: "ResultCallback"
    error_callback: "ErrorCallback"
    cancelled_event: threading.Event


@dataclass
class ScreeningParams:
    """Parameters for screening thread."""

    criteria: str
    universe: str
    top_n: int
    save_to_watchlist: bool
    progress_callback: "ProgressCallback | None"
    result_callback: "ResultCallback"
    error_callback: "ErrorCallback"
    cancelled_event: threading.Event


# Active jobs registry (thread-safe via GIL for dict operations)
_active_jobs: dict[str, AnalysisJob] = {}


# --- Callbacks Type Aliases ---

ResultCallback = Callable[[dict], None]  # (result_dict)
ErrorCallback = Callable[[str], None]  # (error_message)


# --- Helpers ---


def _validate_symbol(symbol: str) -> str:
    """Validate and sanitize stock symbol."""
    if not symbol or not re.match(r"^[A-Z]{1,5}$", symbol.upper()):
        msg = f"Invalid stock symbol: {symbol}"
        raise ValueError(msg)
    return symbol.upper()


def _update_progress(step: str, detail: str, callback: ProgressCallback | None) -> None:
    """Update progress via callback (thread-safe when using Textual post_message)."""
    from src.tui.log_capture import set_active_step

    set_active_step(step)  # Associate logs with this step
    if callback:
        callback(step, "active", detail)


def _check_cancelled(cancelled_event: threading.Event | None) -> None:
    """Check if operation was cancelled and raise if so."""
    if cancelled_event and cancelled_event.is_set():
        msg = "Operation cancelled"
        raise asyncio.CancelledError(msg)


def _create_workflow_with_progress(progress_callback: ProgressCallback | None) -> "TradingWorkflow":
    """Create workflow components with progress tracking."""
    # Configure torch NOW (lazy - only when analysis starts)
    from src.models.torch_config import configure_torch_runtime

    configure_torch_runtime()

    from src.cache.historical import HistoricalCache
    from src.data.fundamental import FundamentalDataFetcher
    from src.data.market import MarketDataFetcher
    from src.data.news import NewsFetcher
    from src.models.llm import LLMClient
    from src.models.sentiment import get_finbert_sentiment
    from src.workflows.trading import TradingWorkflow

    historical_cache = HistoricalCache()

    llm_client = LLMClient()
    market_fetcher = MarketDataFetcher(use_alpha_vantage=False, historical_cache=historical_cache)
    news_fetcher = NewsFetcher(historical_cache=historical_cache)

    _update_progress("fetch_data", "Loading FinBERT model...", progress_callback)
    finbert = get_finbert_sentiment()
    fundamental_fetcher = FundamentalDataFetcher(historical_cache=historical_cache)

    return TradingWorkflow(
        llm_client,
        market_fetcher,
        news_fetcher,
        finbert,
        fundamental_fetcher,
        broker=None,
        metrics_tracker=None,
        use_meta_agent=True,
        historical_cache=historical_cache,
    )


def _patch_workflow_progress(workflow: "TradingWorkflow", progress_callback: ProgressCallback | None) -> None:
    """Patch workflow methods to report progress."""
    from src.tui.log_capture import clear_active_step

    original_run_analyses = workflow.run_analyses

    async def patched_run_analyses(
        state: dict,
        technical_analyst: "TechnicalAnalyst",
        collector: object = None,
    ) -> dict:
        _update_progress("technical", "Running technical analysis...", progress_callback)
        result = await original_run_analyses(state, technical_analyst, collector)
        clear_active_step()  # Clear after analyses complete
        return result

    workflow.run_analyses = patched_run_analyses

    original_make_decision = workflow.make_decision

    async def patched_make_decision(state: dict) -> dict:
        _update_progress("decision", "Synthesizing trading decision...", progress_callback)
        result = await original_make_decision(state)
        clear_active_step()  # Clear after decision complete
        return result

    workflow.make_decision = patched_make_decision


def _setup_isolated_event_loop() -> asyncio.AbstractEventLoop:
    """Create fresh event loop for isolated thread execution.

    Each thread gets its own event loop and semaphore to avoid
    cross-loop async client issues.
    """
    from src.models.llm import _semaphore_holder

    # Clear our semaphore (bound to old event loop)
    _semaphore_holder.clear()

    # Create fresh event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    return loop


def _cleanup_event_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Clean up event loop: cancel tasks, close loop."""
    # Cancel pending tasks
    pending = asyncio.all_tasks(loop)
    for task in pending:
        task.cancel()

    if pending:
        with contextlib.suppress(Exception):
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))

    loop.close()


async def _run_analysis_async(
    symbol: str,
    period_days: int,
    progress_callback: ProgressCallback | None,
    cancelled_event: threading.Event | None,
) -> dict:
    """Internal async function that runs in isolated thread's event loop."""
    validated_symbol = _validate_symbol(symbol)

    try:
        _update_progress("fetch_data", "Initializing...", progress_callback)

        workflow = _create_workflow_with_progress(progress_callback)
        _patch_workflow_progress(workflow, progress_callback)

        # Cancellation check task
        async def check_cancellation() -> None:
            while True:
                await asyncio.sleep(0.5)
                _check_cancelled(cancelled_event)

        cancellation_task = asyncio.create_task(check_cancellation()) if cancelled_event else None

        try:
            result = await workflow.analyze(validated_symbol, period_days=period_days)
            _update_progress("decision", "Analysis complete", progress_callback)
            return result.model_dump()
        finally:
            if cancellation_task:
                cancellation_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await cancellation_task

    except asyncio.CancelledError:
        logger.info(f"Analysis cancelled for {validated_symbol}")
        raise
    except Exception as e:
        logger.error(f"Analysis failed for {validated_symbol}: {e}")
        msg = f"Analysis failed: {e}"
        raise RuntimeError(msg) from e


def _analysis_thread_target(params: AnalysisParams) -> None:
    """Thread entry point - creates isolated event loop and runs analysis."""
    from src.tui.log_capture import setup_log_capture, teardown_log_capture, worker_log_context

    loop = _setup_isolated_event_loop()
    handler_id = None

    try:
        # Set up log capture if progress callback provided
        if params.progress_callback:
            handler_id = setup_log_capture(params.progress_callback)

        # Wrap in worker_log_context so all logs get tui_worker=True
        with worker_log_context():
            coro = _run_analysis_async(
                params.symbol, params.period_days, params.progress_callback, params.cancelled_event
            )
            result = loop.run_until_complete(coro)
            params.result_callback(result)
    except asyncio.CancelledError:
        params.error_callback("Analysis cancelled")
    except Exception as e:
        params.error_callback(str(e))
    finally:
        if handler_id is not None:
            teardown_log_capture(handler_id)
        _cleanup_event_loop(loop)
        _active_jobs.pop(params.symbol, None)


def start_analysis(
    symbol: str,
    period_days: int = 90,
    progress_callback: ProgressCallback | None = None,
    result_callback: ResultCallback | None = None,
    error_callback: ErrorCallback | None = None,
) -> str:
    """Start analysis in isolated thread (fire-and-forget).

    Returns:
        Job ID (symbol) for cancellation
    """
    validated_symbol = _validate_symbol(symbol)

    # Remember previous job to wait for cleanup
    previous_job = _active_jobs.get(validated_symbol)

    cancel_analysis(validated_symbol)

    # Wait for previous thread to finish cleanup
    if previous_job is not None:
        thread = previous_job.thread
        if thread is not None and thread.is_alive():
            try:
                thread.join(timeout=5.0)
                if thread.is_alive():
                    logger.warning(
                        f"Previous analysis thread for {validated_symbol} "
                        "did not terminate within 5s after cancellation."
                    )
            except Exception as e:
                logger.warning(f"Error waiting for previous analysis thread for {validated_symbol}: {e}")

    def noop_result(_: dict) -> None:
        pass

    def noop_error(_: str) -> None:
        pass

    params = AnalysisParams(
        symbol=validated_symbol,
        period_days=period_days,
        progress_callback=progress_callback,
        result_callback=result_callback or noop_result,
        error_callback=error_callback or noop_error,
        cancelled_event=threading.Event(),
    )

    thread = threading.Thread(
        target=_analysis_thread_target,
        args=(params,),
        name=f"analysis-{validated_symbol}",
        daemon=True,
    )

    job = AnalysisJob(thread=thread, cancelled=params.cancelled_event, symbol=validated_symbol)
    _active_jobs[validated_symbol] = job

    thread.start()
    logger.info(f"Started analysis thread for {validated_symbol}")

    return validated_symbol


def cancel_analysis(symbol: str) -> bool:
    """Cancel running analysis for symbol.

    Returns:
        True if job was cancelled, False if no active job
    """
    job = _active_jobs.get(symbol)
    if job:
        job.cancelled.set()
        logger.info(f"Cancelled analysis for {symbol}")
        return True
    return False


def is_analysis_running(symbol: str) -> bool:
    """Check if analysis is running for symbol."""
    job = _active_jobs.get(symbol)
    return job is not None and job.thread.is_alive()


async def _run_screening_async(params: ScreeningParams) -> dict:
    """Internal async function that runs in isolated thread's event loop."""
    from src.tui.log_capture import clear_active_step

    try:
        _update_progress("fetch_universe", "Fetching stock universe...", params.progress_callback)

        from src.data.universe import StockUniverseFetcher
        from src.models.llm import LLMClient
        from src.screening.analyzer import ScreeningAnalyzer
        from src.screening.exporter import ScreeningExporter
        from src.screening.screener import ScreeningCriteria, StockScreener

        universe_fetcher = StockUniverseFetcher()

        _update_progress(
            "screening", f"Screening {params.universe} for {params.criteria}...", params.progress_callback
        )
        screener = StockScreener(universe_fetcher=universe_fetcher)
        screening_criteria = ScreeningCriteria(params.criteria)
        output = screener.screen(criteria=screening_criteria, universe=params.universe, top_n=params.top_n)
        clear_active_step()  # Clear after screening complete

        _check_cancelled(params.cancelled_event)

        _update_progress("analyzing", "Analyzing results with LLM...", params.progress_callback)
        llm = LLMClient()
        analyzer = ScreeningAnalyzer(llm_client=llm)

        analysis = await analyzer.analyze(output)
        clear_active_step()  # Clear after analysis complete

        _check_cancelled(params.cancelled_event)

        formatted = _format_screening_output(output, analysis)

        if params.save_to_watchlist and output.results:
            _update_progress("analyzing", "Saving to watchlist...", params.progress_callback)
            exporter = ScreeningExporter()
            exporter.save_to_watchlist(output.results, screening_criteria, "default")
            clear_active_step()  # Clear after save complete

        _update_progress("analyzing", "Screening complete", params.progress_callback)

        return {
            "screening_output": output.model_dump(),
            "analysis": analysis.model_dump(),
            "formatted_output": formatted,
        }

    except asyncio.CancelledError:
        logger.info(f"Screening cancelled for {params.criteria}")
        raise
    except Exception as e:
        logger.error(f"Screening failed: {e}")
        msg = f"Screening failed: {e}"
        raise RuntimeError(msg) from e


def _screening_thread_target(params: ScreeningParams) -> None:
    """Thread entry point for screening."""
    from src.tui.log_capture import setup_log_capture, teardown_log_capture, worker_log_context

    loop = _setup_isolated_event_loop()
    handler_id = None

    try:
        # Set up log capture if progress callback provided
        if params.progress_callback:
            handler_id = setup_log_capture(params.progress_callback)

        # Wrap in worker_log_context so all logs get tui_worker=True
        with worker_log_context():
            result = loop.run_until_complete(_run_screening_async(params))
            params.result_callback(result)
    except asyncio.CancelledError:
        params.error_callback("Screening cancelled")
    except Exception as e:
        params.error_callback(str(e))
    finally:
        if handler_id is not None:
            teardown_log_capture(handler_id)
        _cleanup_event_loop(loop)


@dataclass
class ScreeningCallbacks:
    """Callbacks for screening progress and results."""

    progress: ProgressCallback | None = None
    result: ResultCallback | None = None
    error: ErrorCallback | None = None


def start_screening(
    criteria: str,
    universe: str = "COMBINED",
    top_n: int = 10,
    save_to_watchlist: bool = False,
    callbacks: ScreeningCallbacks | None = None,
) -> threading.Event:
    """Start screening in isolated thread (fire-and-forget).

    Returns:
        Cancellation event - call .set() to cancel
    """
    cb = callbacks or ScreeningCallbacks()

    def noop_result(_: dict) -> None:
        pass

    def noop_error(_: str) -> None:
        pass

    params = ScreeningParams(
        criteria=criteria,
        universe=universe,
        top_n=top_n,
        save_to_watchlist=save_to_watchlist,
        progress_callback=cb.progress,
        result_callback=cb.result or noop_result,
        error_callback=cb.error or noop_error,
        cancelled_event=threading.Event(),
    )

    thread = threading.Thread(
        target=_screening_thread_target,
        args=(params,),
        name=f"screening-{criteria}",
        daemon=True,
    )
    thread.start()
    logger.info(f"Started screening thread for {criteria}")

    return params.cancelled_event


@dataclass
class LegacyScreeningOptions:
    """Options for legacy screening API."""

    universe: str = "COMBINED"
    top_n: int = 10
    save_to_watchlist: bool = False
    progress_callback: ProgressCallback | None = None
    is_cancelled: Callable[[], bool] | None = None


def _resolve_screening_options(
    options: LegacyScreeningOptions | None,
    legacy_kwargs: dict,
) -> LegacyScreeningOptions:
    """Resolve options from explicit options or legacy kwargs."""
    if options is not None and legacy_kwargs:
        logger.warning(
            "run_screening_in_process received both 'options' and legacy kwargs; ignoring legacy kwargs."
        )
    if options is None:
        return LegacyScreeningOptions(**legacy_kwargs) if legacy_kwargs else LegacyScreeningOptions()
    return options


def _handle_screening_result(error_holder: list[str], result_holder: dict) -> dict:
    """Handle screening result or raise appropriate error."""
    if error_holder:
        if "cancelled" in error_holder[0].lower():
            raise asyncio.CancelledError(error_holder[0])
        raise RuntimeError(error_holder[0])
    return result_holder["data"]


async def run_screening_in_process(
    criteria: str,
    options: LegacyScreeningOptions | None = None,
    **legacy_kwargs: str | int | Callable[[], bool] | ProgressCallback,
) -> dict:
    """Run screening in isolated thread (legacy async API).

    DEPRECATED: Use start_screening() for fire-and-forget pattern.

    Accepts either explicit `options` instance or legacy keyword arguments
    matching LegacyScreeningOptions fields.
    """
    opts = _resolve_screening_options(options, legacy_kwargs)
    result_holder: dict = {}
    error_holder: list[str] = []
    done_event = threading.Event()
    cancelled_event = threading.Event()

    def on_result(result: dict) -> None:
        result_holder["data"] = result
        done_event.set()

    def on_error(error: str) -> None:
        error_holder.append(error)
        done_event.set()

    def progress_with_cancel_check(step_id: str, status: str, detail: str) -> None:
        if opts.is_cancelled and opts.is_cancelled():
            cancelled_event.set()
        if opts.progress_callback:
            opts.progress_callback(step_id, status, detail)

    params = ScreeningParams(
        criteria=criteria,
        universe=opts.universe,
        top_n=opts.top_n,
        save_to_watchlist=opts.save_to_watchlist,
        progress_callback=progress_with_cancel_check,
        result_callback=on_result,
        error_callback=on_error,
        cancelled_event=cancelled_event,
    )

    thread = threading.Thread(
        target=_screening_thread_target,
        args=(params,),
        name=f"screening-{criteria}",
        daemon=True,
    )
    thread.start()

    while not done_event.is_set():
        await asyncio.sleep(0.1)
        if opts.is_cancelled and opts.is_cancelled():
            cancelled_event.set()

    return _handle_screening_result(error_holder, result_holder)


def _format_screening_output(output: "ScreeningOutput", analysis: "ScreeningAnalysis") -> str:
    """Format screening output as markdown."""
    lines = [
        f"## {output.criteria.value.title()} Screening Results",
        f"**Universe:** {output.universe} | **Screened:** {output.total_screened} stocks",
        "",
        "### Analysis",
        analysis.summary,
        "",
        "**Top Picks:**",
    ]
    for pick in analysis.top_picks:
        lines.append(f"- {pick}")
    lines.extend(
        [
            "",
            f"**Sector Insights:** {analysis.sector_insights}",
            f"**Risk Factors:** {analysis.risk_factors}",
            f"**Next Steps:** {analysis.next_steps}",
            "",
            "### Results",
            "",
        ]
    )
    for i, result in enumerate(output.results, 1):
        metrics_str = ", ".join(f"{k}={v}" for k, v in result.metrics.items())
        lines.extend(
            [
                f"**{i}. {result.symbol}** - {result.name}",
                f"   Sector: {result.sector} | Score: {result.score:.2f} | Signal: {result.signal.value}",
                f"   Metrics: {metrics_str}",
                f"   *{result.reason}*",
                "",
            ]
        )
    if output.errors:
        lines.append(f"*Note: {len(output.errors)} symbols failed to screen.*")
    return "\n".join(lines)
