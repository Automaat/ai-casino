"""Backtest subcommand for strategy backtesting."""

import sys
from datetime import UTC, datetime, timedelta

import typer
from loguru import logger
from rich.console import Console
from rich.table import Table

from src.backtesting.runner import BacktestResult, BacktestRunner
from src.backtesting.vectorbt_runner import MultiAssetBacktest, VectorBTResult, VectorBTRunner
from src.utils.logging import sanitize_log_record

console = Console()


def _validate_legacy_engine(symbol: str, portfolio: bool) -> None:
    """Validate legacy backtesting engine constraints."""
    if "," in symbol:
        console.print(
            "[bold red]Error:[/bold red] Legacy 'backtesting' engine does not support "
            "multiple symbols. Use engine='vectorbt' or provide a single symbol."
        )
        raise typer.Exit(1)
    if portfolio:
        console.print(
            "[bold red]Error:[/bold red] Legacy 'backtesting' engine does not support "
            "portfolio mode. Use engine='vectorbt' or remove --portfolio flag."
        )
        raise typer.Exit(1)


def _print_vectorbt_result(result: VectorBTResult) -> None:
    """Print VectorBT backtest result."""
    console.print(f"\n[bold cyan]Vectorized Backtest: {result.symbol}[/bold cyan]")
    console.print(f"Period: {result.start_date:%Y-%m-%d} to {result.end_date:%Y-%m-%d}")
    console.print("=" * 50)

    table = Table(title="Performance Metrics", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="yellow")

    return_color = "green" if result.total_return > 0 else "red"
    sharpe_color = "green" if result.sharpe_ratio > 1 else "yellow" if result.sharpe_ratio > 0 else "red"

    table.add_row("Total Return", f"[{return_color}]{result.total_return:.2%}[/{return_color}]")
    table.add_row("Sharpe Ratio", f"[{sharpe_color}]{result.sharpe_ratio:.2f}[/{sharpe_color}]")
    table.add_row("Sortino Ratio", f"{result.sortino_ratio:.2f}")
    table.add_row("Max Drawdown", f"[red]{result.max_drawdown:.2%}[/red]")
    table.add_row("Calmar Ratio", f"{result.calmar_ratio:.2f}")
    table.add_row("Win Rate", f"{result.win_rate:.2%}")
    table.add_row("Profit Factor", f"{result.profit_factor:.2f}")
    table.add_row("Total Trades", str(result.total_trades))

    console.print(table)


def _print_backtest_result(result: BacktestResult) -> None:
    """Print legacy backtest result."""
    console.print(f"\n[bold cyan]Backtest: {result.symbol}[/bold cyan]")
    console.print(f"Period: {result.start_date:%Y-%m-%d} to {result.end_date:%Y-%m-%d}")
    console.print("=" * 50)

    table = Table(title="Performance Metrics", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="yellow")

    return_color = "green" if result.total_return > 0 else "red"
    sharpe_color = "green" if result.sharpe_ratio > 1 else "yellow" if result.sharpe_ratio > 0 else "red"

    table.add_row("Total Return", f"[{return_color}]{result.total_return:.2%}[/{return_color}]")
    table.add_row("Sharpe Ratio", f"[{sharpe_color}]{result.sharpe_ratio:.2f}[/{sharpe_color}]")
    table.add_row("Max Drawdown", f"[red]{result.max_drawdown:.2%}[/red]")
    table.add_row("Win Rate", f"{result.win_rate:.2%}")
    table.add_row("Total Trades", str(result.total_trades))
    table.add_row("Avg Return/Trade", f"{result.avg_return_per_trade:.2%}")

    console.print(table)


def _print_portfolio_result(result: MultiAssetBacktest) -> None:
    """Print multi-asset portfolio backtest result."""
    console.print(f"\n[bold cyan]Portfolio Backtest: {', '.join(result.symbols)}[/bold cyan]")
    console.print("=" * 50)

    summary = Table(title="Portfolio Summary", show_header=True)
    summary.add_column("Metric", style="cyan")
    summary.add_column("Value", style="yellow")

    return_color = "green" if result.portfolio_return > 0 else "red"
    if result.portfolio_sharpe > 1:
        sharpe_color = "green"
    elif result.portfolio_sharpe > 0:
        sharpe_color = "yellow"
    else:
        sharpe_color = "red"

    summary.add_row("Portfolio Return", f"[{return_color}]{result.portfolio_return:.2%}[/{return_color}]")
    summary.add_row("Portfolio Sharpe", f"[{sharpe_color}]{result.portfolio_sharpe:.2f}[/{sharpe_color}]")
    summary.add_row("Portfolio Max DD", f"[red]{result.portfolio_max_drawdown:.2%}[/red]")

    console.print(summary)

    for r in result.results:
        _print_vectorbt_result(r)

    corr_table = Table(title="Correlation Matrix", show_header=True)
    corr_table.add_column("", style="cyan")
    for sym in result.symbols:
        corr_table.add_column(sym, style="yellow")

    for sym in result.symbols:
        row = [sym]
        for other in result.symbols:
            val = result.correlation_matrix[sym][other]
            row.append(f"{val:.2f}")
        corr_table.add_row(*row)

    console.print(corr_table)


def backtest(
    symbol: str,
    engine: str = "vectorbt",
    start: str | None = None,
    end: str | None = None,
    cash: float = 100_000.0,
    portfolio: bool = False,
) -> None:
    """Run backtest for symbol(s)."""
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
    start_date = start or (datetime.now(tz=UTC) - timedelta(days=365)).strftime("%Y-%m-%d")

    if engine not in ("vectorbt", "backtesting"):
        console.print(f"[bold red]Unknown engine:[/bold red] {engine}. Use 'vectorbt' or 'backtesting'.")
        raise typer.Exit(1)

    try:
        if engine == "vectorbt":
            runner = VectorBTRunner(cash=cash)
            symbols = [s.strip().upper() for s in symbol.split(",") if s.strip()]
            symbols = list(dict.fromkeys(symbols))  # Deduplicate while preserving order

            if portfolio or len(symbols) > 1:
                result = runner.run_portfolio_backtest(symbols, start_date, end_date)
                _print_portfolio_result(result)
            else:
                result = runner.run_backtest(symbols[0], start_date, end_date)
                _print_vectorbt_result(result)
        else:
            _validate_legacy_engine(symbol, portfolio)
            bt_runner = BacktestRunner(cash=cash)
            result = bt_runner.run_backtest(symbol.upper(), start_date, end_date)
            _print_backtest_result(result)
    except Exception as e:
        console.print(f"\n[bold red]Backtest failed:[/bold red] {e}")
        logger.exception("Backtest failed")
        raise typer.Exit(1) from e
