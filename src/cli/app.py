"""Typer CLI application entry point."""

import os
import sys

# Suppress transformers/torch warnings BEFORE any imports cascade to them
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(
    name="aicasino",
    help="AI Casino - Multi-agent stock trading system",
    no_args_is_help=False,
)


@app.command()
def analyze(
    symbol: Annotated[str, typer.Argument(help="Stock ticker symbol")],
    period: Annotated[int, typer.Option("--period", "-p", help="Days of historical data")] = 90,
    trade: Annotated[bool, typer.Option("--trade", "-t", help="Enable paper trading")] = False,
    show_metrics: Annotated[bool, typer.Option("--show-metrics", "-m", help="Show metrics")] = False,
    no_meta_agent: Annotated[bool, typer.Option("--no-meta-agent", help="Use momentum only")] = False,
    trump_mode: Annotated[bool, typer.Option("--trump-mode", help="Trump social media analysis")] = False,
    metrics: Annotated[bool, typer.Option("--metrics", help="Show execution performance metrics")] = False,
) -> None:
    """Analyze a stock and generate trading recommendations."""
    from src.cli.analyze import analyze as analyze_impl

    return analyze_impl(symbol, period, trade, show_metrics, no_meta_agent, trump_mode, metrics)


@app.command()
def optimize(
    symbol: Annotated[str, typer.Argument(help="Stock ticker symbol")],
    strategy: Annotated[str, typer.Option("--strategy", "-s", help="Strategy")] = "momentum",
    trials: Annotated[int, typer.Option("--trials", "-n", help="Trials")] = 100,
    multi_objective: Annotated[bool, typer.Option("--multi-objective", help="Multi-objective")] = False,
    walk_forward: Annotated[bool, typer.Option("--walk-forward", help="Walk-forward")] = False,
    splits: Annotated[int, typer.Option("--splits", help="Validation splits")] = 5,
    start: Annotated[str | None, typer.Option("--start", help="Start date (YYYY-MM-DD)")] = None,
    end: Annotated[str | None, typer.Option("--end", help="End date (YYYY-MM-DD)")] = None,
) -> None:
    """Optimize trading strategy parameters."""
    from src.cli.optimize import optimize as optimize_impl

    return optimize_impl(symbol, strategy, trials, multi_objective, walk_forward, splits, start, end)


@app.command()
def daemon(
    config: Annotated[
        Path | None, typer.Option("--config", "-c", help="Path to daemon config file (TOML)")
    ] = None,
) -> None:
    """Run autonomous trading daemon (24/7 scheduled analysis)."""
    from src.cli.daemon import daemon as daemon_impl

    return daemon_impl(config)


@app.command(name="trump-daemon")
def trump_daemon(
    poll_interval: Annotated[int, typer.Option("--interval", "-i", help="Poll interval in seconds")] = 60,
    max_analyses: Annotated[
        int, typer.Option("--max-analyses", "-m", help="Max stocks to analyze per signal")
    ] = 5,
) -> None:
    """Run Trump social media watcher daemon.

    Monitors Trump's Truth Social posts and triggers stock analysis
    when market-relevant posts are detected.
    """
    from src.cli.daemon import trump_daemon as trump_daemon_impl

    return trump_daemon_impl(poll_interval, max_analyses)


@app.command(name="chat")
def chat() -> None:
    """Launch interactive TUI chat interface."""
    from src.cli.chat import chat as chat_impl

    return chat_impl()


@app.command()
def tearsheet(
    symbol: Annotated[str, typer.Argument(help="Stock ticker symbol")],
    period: Annotated[str, typer.Option("--period", "-p", help="Time period")] = "1y",
    benchmark: Annotated[str | None, typer.Option("--benchmark", "-b", help="Benchmark symbol")] = "SPY",
    no_benchmark: Annotated[bool, typer.Option("--no-benchmark", help="Disable benchmark")] = False,
) -> None:
    """Generate QuantStats performance tearsheet."""
    from src.cli.tearsheet import tearsheet as tearsheet_impl

    return tearsheet_impl(symbol, period, benchmark if not no_benchmark else None)


def main() -> None:
    """CLI entry point - defaults to chat mode."""
    if len(sys.argv) == 1:
        chat()
    else:
        app()


if __name__ == "__main__":
    main()
