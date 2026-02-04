"""Worker process for running analysis outside Textual's fd context."""

import asyncio
import contextlib
import json
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path


def _raise_process_error(returncode: int, stderr_path: Path) -> None:
    """Raise RuntimeError with stderr content if available."""
    stderr_output = ""
    with contextlib.suppress(OSError):
        stderr_output = stderr_path.read_text().strip()
    msg = f"Analysis process failed with code {returncode}"
    if stderr_output:
        msg = f"{msg}\nstderr: {stderr_output}"
    raise RuntimeError(msg)


def _validate_symbol(symbol: str) -> str:
    """Validate and sanitize stock symbol.

    Args:
        symbol: Stock ticker symbol

    Returns:
        Validated symbol

    Raises:
        ValueError: If symbol is invalid
    """
    if not symbol or not re.match(r"^[A-Z]{1,5}$", symbol.upper()):
        msg = f"Invalid stock symbol: {symbol}"
        raise ValueError(msg)
    return symbol.upper()


_STEP_MAP = {
    "fetch_data": "fetch_data",
    "loading_model": "fetch_data",
    "technical": "technical",
    "decision": "decision",
    "complete": "decision",
}


def _parse_status_file(status_path: Path) -> tuple[str, str]:
    """Parse status file and return (step, detail)."""
    try:
        content = status_path.read_text().strip()
        if content:
            status_data = json.loads(content)
            return status_data.get("step", ""), status_data.get("detail", "")
    except (OSError, json.JSONDecodeError):
        # Best-effort parsing: on any read/parse error, treat status as empty
        pass
    return "", ""


def _terminate_process(process: subprocess.Popen) -> None:
    """Terminate process gracefully, escalate to kill if needed."""
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    except OSError:
        pass  # Process already exited


_WORKER_SCRIPT = """
import asyncio
import json
import os
import sys

from dotenv import load_dotenv
load_dotenv()

# Fix sniffio async library detection
import sniffio
sniffio.current_async_library_cvar.set("asyncio")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

_current_step = "fetch_data"
_status_file = None

def update_status(step, detail=""):
    global _current_step, _status_file
    _current_step = step
    if _status_file:
        with open(_status_file, "w") as f:
            json.dump({"step": step, "detail": detail[:80] if detail else ""}, f)

def update_detail(detail):
    global _current_step, _status_file
    if _status_file:
        with open(_status_file, "w") as f:
            json.dump({"step": _current_step, "detail": detail[:80] if detail else ""}, f)

_last_log_time = 0

def log_sink(message):
    global _last_log_time
    import time
    now = time.time()
    if now - _last_log_time < 0.3:
        return
    _last_log_time = now
    record = message.record
    msg = record["message"]
    if msg and len(msg) > 5:
        update_detail(msg)

def main():
    global _status_file
    symbol = sys.argv[1]
    period_days = int(sys.argv[2])
    output_file = sys.argv[3]
    _status_file = sys.argv[4]

    try:
        update_status("fetch_data")

        from loguru import logger
        logger.remove()
        logger.add(log_sink, format="{message}", level="INFO")
        # File logging for subprocess debugging
        log_file = os.path.expanduser("~/.ai-casino/worker.log")
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        logger.add(log_file, format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}", level="DEBUG")

        from src.data.fundamental import FundamentalDataFetcher
        from src.data.market import MarketDataFetcher
        from src.data.news import NewsFetcher
        from src.models.llm import LLMClient
        from src.models.sentiment import FinBERTSentiment
        from src.workflows.trading import TradingWorkflow

        llm_client = LLMClient()
        market_fetcher = MarketDataFetcher(use_alpha_vantage=False)
        news_fetcher = NewsFetcher()

        update_status("loading_model", "Loading FinBERT model...")
        finbert = FinBERTSentiment()
        fundamental_fetcher = FundamentalDataFetcher()

        workflow = TradingWorkflow(
            llm_client,
            market_fetcher,
            news_fetcher,
            finbert,
            fundamental_fetcher,
            broker=None,
            metrics_tracker=None,
            use_meta_agent=True,
        )

        # Patch workflow to report progress
        original_run_analyses = workflow._run_analyses
        async def patched_run_analyses(state, technical_analyst):
            update_status("technical", "Starting technical analysis...")
            return await original_run_analyses(state, technical_analyst)
        workflow._run_analyses = patched_run_analyses

        original_make_decision = workflow._make_decision
        async def patched_make_decision(state):
            update_status("decision", "Synthesizing trading decision...")
            return await original_make_decision(state)
        workflow._make_decision = patched_make_decision

        # Python 3.14 fix: Patch event loop for anyio compatibility
        # anyio has issues with task state tracking in worker threads
        import nest_asyncio
        nest_asyncio.apply()

        result = asyncio.run(workflow.analyze(symbol, period_days=period_days))

        update_status("complete", "Analysis complete")

        with open(output_file, "w") as f:
            json.dump({"status": "success", "data": result.model_dump()}, f)
    except Exception as e:
        update_status("error", str(e)[:80])
        with open(output_file, "w") as f:
            json.dump({"status": "error", "data": str(e)}, f)

if __name__ == "__main__":
    main()
"""


async def run_analysis_in_process(
    symbol: str,
    period_days: int = 90,
    progress_callback: Callable[[str, str, str], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> dict:
    """Run analysis in a separate process to avoid Textual fd conflicts.

    Args:
        symbol: Stock ticker symbol
        period_days: Days of historical data
        progress_callback: Optional callback(step_id, status, detail) for progress updates
        is_cancelled: Optional callback returning True if analysis should be cancelled

    Returns:
        Analysis result dict

    Raises:
        RuntimeError: If analysis fails
        asyncio.CancelledError: If cancelled via is_cancelled callback
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as script_file:
        script_file.write(_WORKER_SCRIPT)
        script_path = Path(script_file.name)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as output_file:
        output_path = Path(output_file.name)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".status", delete=False) as status_file:
        status_path = Path(status_file.name)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".stderr", delete=False) as stderr_file:
        stderr_path = Path(stderr_file.name)

    try:
        validated_symbol = _validate_symbol(symbol)
        cmd = [
            sys.executable,
            str(script_path),
            validated_symbol,
            str(period_days),
            str(output_path),
            str(status_path),
        ]
        with stderr_path.open("w") as stderr_fh:
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=stderr_fh,
                close_fds=True,
                cwd=Path(__file__).parent.parent.parent,
            )

        last_step, last_detail = "", ""

        cancelled = False
        try:
            while process.poll() is None:
                await asyncio.sleep(0.3)
                if is_cancelled and is_cancelled():
                    cancelled = True
                    break
                if progress_callback and status_path.exists():
                    current_step, current_detail = _parse_status_file(status_path)
                    if (current_step and current_step != last_step) or current_detail != last_detail:
                        step_id = _STEP_MAP.get(current_step, current_step)
                        progress_callback(step_id, "active", current_detail)
                        last_step, last_detail = current_step, current_detail
        except asyncio.CancelledError:
            cancelled = True

        if cancelled:
            _terminate_process(process)
            raise asyncio.CancelledError

        if process.returncode != 0:
            _raise_process_error(process.returncode, stderr_path)

        with output_path.open() as f:
            result = json.load(f)

        if result["status"] == "error":
            raise RuntimeError(result["data"])

        return result["data"]

    finally:
        script_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        status_path.unlink(missing_ok=True)
        stderr_path.unlink(missing_ok=True)


_SCREENING_WORKER_SCRIPT = """
import asyncio
import json
import os
import sys

from dotenv import load_dotenv
load_dotenv()

# Fix sniffio async library detection
import sniffio
sniffio.current_async_library_cvar.set("asyncio")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

_current_step = "fetch_universe"
_status_file = None

def update_status(step, detail=""):
    global _current_step, _status_file
    _current_step = step
    if _status_file:
        with open(_status_file, "w") as f:
            json.dump({"step": step, "detail": detail[:80] if detail else ""}, f)

def update_detail(detail):
    global _current_step, _status_file
    if _status_file:
        with open(_status_file, "w") as f:
            json.dump({"step": _current_step, "detail": detail[:80] if detail else ""}, f)

_last_log_time = 0

def log_sink(message):
    global _last_log_time
    import time
    now = time.time()
    if now - _last_log_time < 0.3:
        return
    _last_log_time = now
    record = message.record
    msg = record["message"]
    if msg and len(msg) > 5:
        update_detail(msg)

def main():
    global _status_file
    criteria = sys.argv[1]
    universe = sys.argv[2]
    top_n = int(sys.argv[3])
    save_to_watchlist = sys.argv[4] == "True"
    output_file = sys.argv[5]
    _status_file = sys.argv[6]

    try:
        update_status("fetch_universe", "Fetching stock universe...")

        from loguru import logger
        logger.remove()
        logger.add(log_sink, format="{message}", level="INFO")
        # File logging for subprocess debugging
        log_file = os.path.expanduser("~/.ai-casino/worker.log")
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        logger.add(log_file, format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}", level="DEBUG")

        from src.data.universe import StockUniverseFetcher
        from src.models.llm import LLMClient
        from src.screening.analyzer import ScreeningAnalyzer
        from src.screening.exporter import ScreeningExporter
        from src.screening.screener import ScreeningCriteria, StockScreener

        universe_fetcher = StockUniverseFetcher()

        update_status("screening", f"Screening {universe} for {criteria}...")
        screener = StockScreener(universe_fetcher=universe_fetcher)
        screening_criteria = ScreeningCriteria(criteria)
        output = screener.screen(criteria=screening_criteria, universe=universe, top_n=top_n)

        update_status("analyzing", "Analyzing results with LLM...")
        llm = LLMClient()
        analyzer = ScreeningAnalyzer(llm_client=llm)

        # Python 3.14 fix: Patch event loop for anyio compatibility
        import nest_asyncio
        nest_asyncio.apply()

        analysis = asyncio.run(analyzer.analyze(output))

        formatted = format_screening_output(output, analysis)

        if save_to_watchlist and output.results:
            update_status("saving", "Saving to watchlist...")
            exporter = ScreeningExporter()
            exporter.save_to_watchlist(output.results, screening_criteria, "default")

        update_status("complete", "Screening complete")

        with open(output_file, "w") as f:
            json.dump({
                "status": "success",
                "data": {
                    "screening_output": output.model_dump(),
                    "analysis": analysis.model_dump(),
                    "formatted_output": formatted,
                }
            }, f, default=str)

    except Exception as e:
        update_status("error", str(e)[:80])
        with open(output_file, "w") as f:
            json.dump({"status": "error", "data": str(e)}, f)

def format_screening_output(output, analysis):
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
    lines.extend([
        "",
        f"**Sector Insights:** {analysis.sector_insights}",
        f"**Risk Factors:** {analysis.risk_factors}",
        f"**Next Steps:** {analysis.next_steps}",
        "",
        "### Results",
        "",
    ])
    for i, result in enumerate(output.results, 1):
        metrics_str = ", ".join(f"{k}={v}" for k, v in result.metrics.items())
        lines.extend([
            f"**{i}. {result.symbol}** - {result.name}",
            f"   Sector: {result.sector} | Score: {result.score:.2f} | Signal: {result.signal.value}",
            f"   Metrics: {metrics_str}",
            f"   *{result.reason}*",
            "",
        ])
    if output.errors:
        lines.append(f"*Note: {len(output.errors)} symbols failed to screen.*")
    return "\\n".join(lines)

if __name__ == "__main__":
    main()
"""


_SCREENING_STEP_MAP = {
    "fetch_universe": "fetch_universe",
    "screening": "screening",
    "analyzing": "analyzing",
    "saving": "analyzing",
    "complete": "analyzing",
}


async def run_screening_in_process(  # noqa: PLR0913
    criteria: str,
    universe: str = "COMBINED",
    top_n: int = 10,
    save_to_watchlist: bool = False,
    progress_callback: Callable[[str, str, str], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> dict:
    """Run screening in a separate process to avoid Textual fd conflicts.

    Args:
        criteria: Screening criteria (momentum, value, breakout)
        universe: Stock universe (SP500, NASDAQ100, COMBINED)
        top_n: Number of results
        save_to_watchlist: Save results to default watchlist
        progress_callback: Optional callback(step_id, status, detail) for progress updates
        is_cancelled: Optional callback returning True if screening should be cancelled

    Returns:
        Screening result dict

    Raises:
        RuntimeError: If screening fails
        asyncio.CancelledError: If cancelled via is_cancelled callback
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as script_file:
        script_file.write(_SCREENING_WORKER_SCRIPT)
        script_path = Path(script_file.name)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as output_file:
        output_path = Path(output_file.name)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".status", delete=False) as status_file:
        status_path = Path(status_file.name)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".stderr", delete=False) as stderr_file:
        stderr_path = Path(stderr_file.name)

    try:
        cmd = [
            sys.executable,
            str(script_path),
            criteria,
            universe,
            str(top_n),
            str(save_to_watchlist),
            str(output_path),
            str(status_path),
        ]
        with stderr_path.open("w") as stderr_fh:
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=stderr_fh,
                close_fds=True,
                cwd=Path(__file__).parent.parent.parent,
            )

        last_step, last_detail = "", ""

        cancelled = False
        try:
            while process.poll() is None:
                await asyncio.sleep(0.3)
                if is_cancelled and is_cancelled():
                    cancelled = True
                    break
                if progress_callback and status_path.exists():
                    current_step, current_detail = _parse_status_file(status_path)
                    if (current_step and current_step != last_step) or current_detail != last_detail:
                        step_id = _SCREENING_STEP_MAP.get(current_step, current_step)
                        progress_callback(step_id, "active", current_detail)
                        last_step, last_detail = current_step, current_detail
        except asyncio.CancelledError:
            cancelled = True

        if cancelled:
            _terminate_process(process)
            raise asyncio.CancelledError

        if process.returncode != 0:
            _raise_process_error(process.returncode, stderr_path)

        with output_path.open() as f:
            result = json.load(f)

        if result["status"] == "error":
            raise RuntimeError(result["data"])

        return result["data"]

    finally:
        script_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        status_path.unlink(missing_ok=True)
        stderr_path.unlink(missing_ok=True)
