"""CLI command for paper trading validation."""

import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from src.daemon.config import DaemonConfig
from src.daemon.paper_trading_validator import PaperTradingValidator, ReadinessReport
from src.daemon.state import DaemonState
from src.metrics.tracker import MetricsTracker

app = typer.Typer(help="Validate paper trading readiness for live promotion")
console = Console()


def _display_criteria_table(report: ReadinessReport) -> None:
    """Display validation criteria table."""
    criteria_table = Table(title="Validation Criteria", show_header=True, header_style="bold")
    criteria_table.add_column("Criterion", style="cyan")
    criteria_table.add_column("Status", justify="center")
    criteria_table.add_column("Current", justify="right")
    criteria_table.add_column("Threshold", justify="right")
    criteria_table.add_column("Details")

    for criterion in report.criteria:
        status = "[green]✓[/green]" if criterion.passed else "[red]✗[/red]"
        current = f"{criterion.current_value:.2f}"
        threshold = f"{criterion.threshold:.2f}"

        criteria_table.add_row(
            criterion.name,
            status,
            current,
            threshold,
            criterion.message,
        )

    console.print(criteria_table)
    console.print()


def _display_metrics(report: ReadinessReport) -> None:
    """Display paper trading metrics."""
    console.print("[bold]Paper Trading Metrics[/bold]")
    console.print(f"  Sharpe Ratio: {report.metrics.sharpe_ratio:.2f}")
    console.print(f"  Max Drawdown: {report.metrics.max_drawdown_percent:.1f}%")
    console.print(f"  Win Rate: {report.metrics.win_rate:.1%}")
    console.print(f"  Total PnL: ${report.metrics.total_pnl:.2f}")
    console.print(f"  Winning Trades: {report.metrics.winning_trades}")
    console.print(f"  Losing Trades: {report.metrics.losing_trades}")
    console.print()


def _display_simulated_live(report: ReadinessReport) -> None:
    """Display simulated live trading comparison."""
    if not report.simulated_live:
        return

    console.print("[bold]Simulated Live Trading (with fees/slippage)[/bold]")
    console.print(
        f"  Sharpe Ratio: {report.simulated_live.live_metrics.sharpe_ratio:.2f} "
        f"({report.simulated_live.sharpe_delta:+.2f})"
    )
    console.print(
        f"  Total PnL: ${report.simulated_live.live_metrics.total_pnl:.2f} "
        f"({report.simulated_live.total_pnl_delta:+.2f})"
    )
    console.print(
        f"  Win Rate: {report.simulated_live.live_metrics.win_rate:.1%} "
        f"({report.simulated_live.win_rate_delta:+.1%})"
    )
    console.print()


def _display_recommendations(report: ReadinessReport) -> None:
    """Display recommendations."""
    if not report.recommendations:
        return

    console.print("[bold]Recommendations[/bold]")
    for i, rec in enumerate(report.recommendations, 1):
        console.print(f"  {i}. {rec}")
    console.print()


def validate_paper_trading(config_path: str = "daemon.yaml") -> int:
    """Validate paper trading readiness and display detailed report.

    Args:
        config_path: Path to daemon config file

    Returns:
        Exit code (0=ready, 1=not ready, 2=error)
    """
    try:
        # NOTE: This CLI command requires async rewrite after JSON state elimination
        # See PR description: TUI/CLI are expected to be broken
        raise NotImplementedError(
            "validate_paper_trading requires async rewrite after JSON state elimination. "
            "Use daemon API endpoints or wait for CLI refactor."
        )

    except FileNotFoundError:
        console.print(f"[red]Error: Config file not found: {config_path}[/red]")
        return 2
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback

        traceback.print_exc()
        return 2


@app.command()
def main(
    config: str = typer.Option("daemon.yaml", "--config", "-c", help="Path to daemon config file"),
) -> None:
    """Validate paper trading readiness for live promotion."""
    exit_code = validate_paper_trading(config)
    sys.exit(exit_code)


if __name__ == "__main__":
    app()
