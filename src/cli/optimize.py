"""Optimize subcommand for strategy optimization."""

import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Annotated

import typer
from loguru import logger
from rich.console import Console
from rich.table import Table

from src.optimization.optimizer import OptunaOptimizer
from src.optimization.results import OptimizationResult
from src.optimization.validation import WalkForwardValidator
from src.utils.logging import sanitize_log_record

console = Console()


def _print_optimization_result(result: OptimizationResult) -> None:
    """Print optimization results."""
    console.print(
        f"\n[bold cyan]Optimization Results for {result.symbol} ({result.strategy_name})[/bold cyan]"
    )
    console.print("=" * 50)
    console.print(f"Trials: {result.total_trials} | Time: {result.optimization_time_seconds:.1f}s\n")

    params_table = Table(title="Best Parameters", show_header=True)
    params_table.add_column("Parameter", style="cyan")
    params_table.add_column("Value", style="yellow")

    for key, value in result.best_params.items():
        if isinstance(value, float):
            params_table.add_row(key, f"{value:.4f}")
        else:
            params_table.add_row(key, str(value))

    console.print(params_table)

    metrics_table = Table(title="Performance", show_header=True)
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Value", style="yellow")

    sharpe = result.best_metrics.get("sharpe_ratio", 0)
    total_return = result.best_metrics.get("total_return", 0)
    max_dd = result.best_metrics.get("max_drawdown", 0)

    sharpe_color = "green" if sharpe > 1 else "yellow" if sharpe > 0 else "red"
    return_color = "green" if total_return > 0 else "red"

    metrics_table.add_row("Sharpe Ratio", f"[{sharpe_color}]{sharpe:.2f}[/{sharpe_color}]")
    metrics_table.add_row("Total Return", f"[{return_color}]{total_return * 100:.1f}%[/{return_color}]")
    metrics_table.add_row("Max Drawdown", f"[red]{max_dd * 100:.1f}%[/red]")

    console.print(metrics_table)

    if result.pareto_front:
        console.print(f"\n[dim]Pareto front contains {len(result.pareto_front)} solutions[/dim]")


@dataclass
class OptimizeConfig:
    """Configuration for strategy optimization."""

    symbol: str
    strategy: str
    trials: int
    multi_objective: bool
    walk_forward: bool
    splits: int
    start_date: str
    end_date: str


def _run_optimization(config: OptimizeConfig) -> None:
    """Run strategy optimization with config."""
    console.print(f"\n[bold]Running optimization for {config.symbol}...[/bold]")
    console.print(f"Strategy: {config.strategy} | Trials: {config.trials}")
    console.print(f"Period: {config.start_date} to {config.end_date}")

    if config.multi_objective:
        console.print("[dim]Multi-objective: Sharpe, Return, Drawdown[/dim]")
    if config.walk_forward:
        console.print(f"[dim]Walk-forward validation: {config.splits} splits[/dim]")

    console.print()

    validator = WalkForwardValidator(n_splits=config.splits) if config.walk_forward else None
    directions = ["maximize", "maximize", "minimize"] if config.multi_objective else ["maximize"]

    optimizer = OptunaOptimizer(
        n_trials=config.trials,
        directions=directions,
        validator=validator,
    )

    result = optimizer.optimize(
        symbol=config.symbol,
        start_date=config.start_date,
        end_date=config.end_date,
        strategy_name=config.strategy,
    )
    _print_optimization_result(result)


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
    from src.daemon.config import DaemonConfig

    daemon_config = DaemonConfig()

    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=daemon_config.logging.log_level,
        filter=sanitize_log_record,
    )

    end_date = end or datetime.now(tz=UTC).strftime("%Y-%m-%d")
    start_date = start or (datetime.now(tz=UTC) - timedelta(days=365 * 2)).strftime("%Y-%m-%d")

    config = OptimizeConfig(
        symbol=symbol.upper(),
        strategy=strategy,
        trials=trials,
        multi_objective=multi_objective,
        walk_forward=walk_forward,
        splits=splits,
        start_date=start_date,
        end_date=end_date,
    )

    try:
        _run_optimization(config)
    except Exception as e:
        console.print(f"\n[bold red]Optimization failed:[/bold red] {e}")
        logger.exception("Optimization failed")
        raise typer.Exit(1) from e
