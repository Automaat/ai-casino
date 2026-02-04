"""Worker process for running analysis outside Textual's fd context."""

import asyncio
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path


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


_WORKER_SCRIPT = """
import asyncio
import json
import os
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

def update_status(status_file, step):
    with open(status_file, "w") as f:
        f.write(step)

def main():
    symbol = sys.argv[1]
    period_days = int(sys.argv[2])
    output_file = sys.argv[3]
    status_file = sys.argv[4]

    try:
        update_status(status_file, "fetch_data")

        from src.data.fundamental import FundamentalDataFetcher
        from src.data.market import MarketDataFetcher
        from src.data.news import NewsFetcher
        from src.models.llm import LLMClient
        from src.models.sentiment import FinBERTSentiment
        from src.workflows.trading import TradingWorkflow

        llm_client = LLMClient()
        market_fetcher = MarketDataFetcher(use_alpha_vantage=False)
        news_fetcher = NewsFetcher()

        update_status(status_file, "loading_model")
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
            update_status(status_file, "technical")
            return await original_run_analyses(state, technical_analyst)
        workflow._run_analyses = patched_run_analyses

        original_make_decision = workflow._make_decision
        async def patched_make_decision(state):
            update_status(status_file, "decision")
            return await original_make_decision(state)
        workflow._make_decision = patched_make_decision

        result = asyncio.run(workflow.analyze(symbol, period_days=period_days))
        update_status(status_file, "complete")

        with open(output_file, "w") as f:
            json.dump({"status": "success", "data": result.model_dump()}, f)
    except Exception as e:
        update_status(status_file, "error")
        with open(output_file, "w") as f:
            json.dump({"status": "error", "data": str(e)}, f)

if __name__ == "__main__":
    main()
"""


async def run_analysis_in_process(
    symbol: str,
    period_days: int = 90,
    progress_callback: callable | None = None,
) -> dict:
    """Run analysis in a separate process to avoid Textual fd conflicts.

    Args:
        symbol: Stock ticker symbol
        period_days: Days of historical data
        progress_callback: Optional callback(step_id, status) for progress updates

    Returns:
        Analysis result dict

    Raises:
        RuntimeError: If analysis fails
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as script_file:
        script_file.write(_WORKER_SCRIPT)
        script_path = Path(script_file.name)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as output_file:
        output_path = Path(output_file.name)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".status", delete=False) as status_file:
        status_path = Path(status_file.name)

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
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            cwd=Path(__file__).parent.parent.parent,
        )

        last_status = ""
        step_map = {
            "fetch_data": "fetch_data",
            "loading_model": "fetch_data",
            "technical": "technical",
            "decision": "decision",
            "complete": "decision",
        }

        while process.poll() is None:
            await asyncio.sleep(0.3)

            # Check for status updates
            if progress_callback and status_path.exists():
                try:
                    current_status = status_path.read_text().strip()
                    if current_status and current_status != last_status:
                        step_id = step_map.get(current_status, current_status)
                        progress_callback(step_id, "active")
                        last_status = current_status
                except OSError:
                    pass

        if process.returncode != 0:
            msg = f"Analysis process failed with code {process.returncode}"
            raise RuntimeError(msg)

        with output_path.open() as f:
            result = json.load(f)

        if result["status"] == "error":
            raise RuntimeError(result["data"])

        return result["data"]

    finally:
        script_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        status_path.unlink(missing_ok=True)
