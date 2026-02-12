"""CLI command for generating QuantStats tearsheets."""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from loguru import logger
from rich.console import Console
from rich.table import Table

from src.data.market import MarketDataFetcher
from src.metrics.quantstats_reporter import QuantStatsReporter
from src.metrics.tracker import create_metrics_tracker

if TYPE_CHECKING:
    import pandas as pd

    from src.metrics.tracker import TearSheet

console = Console()


def tearsheet(
    symbol: str,
    period: str = "1y",
    benchmark: str | None = "SPY",
) -> None:
    """Generate QuantStats performance tearsheet.

    Args:
        symbol: Stock ticker symbol
        period: Time period ("1m", "3m", "6m", "1y", "all", or integer days)
        benchmark: Benchmark symbol (default "SPY", None to disable)
    """
    asyncio.run(_tearsheet_async(symbol, period, benchmark))


async def _tearsheet_async(
    symbol: str,
    period: str,
    benchmark: str | None,
) -> None:
    """Async implementation of tearsheet generation.

    Args:
        symbol: Stock ticker symbol
        period: Time period specification
        benchmark: Benchmark symbol or None
    """
    console.print(f"\n[bold cyan]Generating tearsheet for {symbol}[/bold cyan]")

    period_days = _parse_period(period)
    console.print(f"Period: {period} ({period_days} days)")

    filtered_trades = await _load_and_filter_trades(symbol, period_days)
    if not filtered_trades:
        return

    console.print(f"Found {len(filtered_trades)} closed trades")

    benchmark_returns = await _fetch_benchmark_data(benchmark, period_days)

    await _generate_and_save_tearsheet(symbol, filtered_trades, benchmark, benchmark_returns)


async def _load_and_filter_trades(symbol: str, period_days: int) -> list:
    """Load all trades and filter by symbol and period.

    Args:
        symbol: Stock ticker symbol
        period_days: Number of days to filter (-1 for all)

    Returns:
        List of filtered closed trades
    """
    tracker = create_metrics_tracker()

    if hasattr(tracker, "trades"):
        all_trades = tracker.trades
    else:
        from src.database.connection import get_session
        from src.database.repositories.trade import TradeRepository

        async with get_session() as session:
            repo = TradeRepository(session)
            all_trades = await repo.get_all()

    symbol_trades = [t for t in all_trades if t.symbol == symbol and t.is_closed()]

    if not symbol_trades:
        console.print(f"[red]No closed trades found for {symbol}[/red]")
        return []

    if period_days != -1:
        cutoff = datetime.now(UTC) - timedelta(days=period_days)
        filtered_trades = [t for t in symbol_trades if t.timestamp >= cutoff]
    else:
        filtered_trades = symbol_trades

    if not filtered_trades:
        console.print(f"[red]No trades in period for {symbol}[/red]")
        return []

    return filtered_trades


async def _fetch_benchmark_data(benchmark: str | None, period_days: int) -> pd.Series | None:
    """Fetch benchmark returns data if benchmark specified.

    Args:
        benchmark: Benchmark symbol or None
        period_days: Number of days to fetch

    Returns:
        pandas Series with returns or None
    """
    if not benchmark:
        return None

    console.print(f"Fetching benchmark data for {benchmark}...")
    try:
        from src.di.container import create_container

        container = create_container()
        fetcher = container.yfinance_market_fetcher()
        benchmark_returns = await _fetch_benchmark_returns(benchmark, period_days, fetcher)
        console.print(f"[green]Benchmark data fetched ({len(benchmark_returns)} days)[/green]")
        return benchmark_returns
    except Exception as e:
        logger.opt(exception=True).warning(f"Failed to fetch benchmark data: {e}")
        console.print("[yellow]Warning: Could not fetch benchmark data, continuing without[/yellow]")
        return None


async def _generate_and_save_tearsheet(
    symbol: str,
    trades: list,
    benchmark: str | None,
    benchmark_returns: pd.Series | None,
) -> None:
    """Generate tearsheet and save to database.

    Args:
        symbol: Stock ticker symbol
        trades: List of closed trades
        benchmark: Benchmark symbol or None
        benchmark_returns: Benchmark returns series or None
    """
    from pathlib import Path

    from src.daemon.config import DaemonConfig

    # Load user's YAML config if exists, otherwise defaults
    default_config_path = Path.home() / ".ai-casino" / "daemon.yaml"
    try:
        daemon_config = (
            DaemonConfig.from_yaml(default_config_path) if default_config_path.exists() else DaemonConfig()
        )
    except Exception as e:
        logger.opt(exception=True).warning(f"Failed to load config, using defaults: {e}")
        daemon_config = DaemonConfig()

    reporter = QuantStatsReporter(daemon_config.metrics.risk_free_rate)

    try:
        tearsheet_obj = reporter.generate_tearsheet(
            symbol=symbol,
            trades=trades,
            benchmark_symbol=benchmark,
            benchmark_returns=benchmark_returns,
        )

        _print_summary(tearsheet_obj)

        console.print(f"\n[green]HTML report saved to:[/green] {tearsheet_obj.html_report_path}")

        database_url = os.getenv("DATABASE_URL")
        if database_url:
            await _save_to_database(tearsheet_obj)

    except Exception as e:
        logger.opt(exception=True).error(f"Tearsheet generation failed: {e}")
        console.print(f"[red]Error: {e}[/red]")
        raise


def _parse_period(period: str) -> int:
    """Parse period string to days.

    Args:
        period: Period specification ("1m", "3m", "6m", "1y", "all", or integer)

    Returns:
        Number of days (-1 for "all")
    """
    period_map = {
        "1m": 30,
        "3m": 90,
        "6m": 180,
        "1y": 365,
        "all": -1,
    }

    if period in period_map:
        return period_map[period]

    try:
        return int(period)
    except ValueError:
        logger.opt(exception=True).warning(f"Unknown period '{period}', using 1y")
        return 365


async def _fetch_benchmark_returns(benchmark: str, period_days: int, fetcher: MarketDataFetcher) -> pd.Series:
    """Fetch benchmark returns data.

    Args:
        benchmark: Benchmark ticker symbol
        period_days: Number of days (-1 for all available)
        fetcher: Market data fetcher for benchmark data

    Returns:
        pandas Series with daily returns
    """
    fetch_days = period_days if period_days != -1 else 365 * 5

    market_data = fetcher.fetch_daily(benchmark, period_days=fetch_days)

    if market_data.data.empty:
        msg = f"No market data available for benchmark {benchmark}"
        raise ValueError(msg)

    equity = market_data.data.get("close", market_data.data["Close"])

    returns = equity.pct_change().fillna(0.0)

    logger.debug(f"Fetched {len(returns)} days of benchmark returns for {benchmark}")
    return returns


def _add_performance_metrics(table: Table, tearsheet: TearSheet) -> None:
    """Add performance metrics to table."""
    table.add_row("[bold]Performance[/bold]", "")
    if tearsheet.cagr is not None:
        table.add_row("CAGR", f"{tearsheet.cagr * 100:.2f}%")
    if tearsheet.sharpe_ratio is not None:
        table.add_row("Sharpe Ratio", f"{tearsheet.sharpe_ratio:.2f}")
    if tearsheet.sortino_ratio is not None:
        table.add_row("Sortino Ratio", f"{tearsheet.sortino_ratio:.2f}")
    if tearsheet.calmar_ratio is not None:
        table.add_row("Calmar Ratio", f"{tearsheet.calmar_ratio:.2f}")


def _add_risk_metrics(table: Table, tearsheet: TearSheet) -> None:
    """Add risk metrics to table."""
    table.add_row("[bold]Risk[/bold]", "")
    if tearsheet.max_drawdown is not None:
        table.add_row("Max Drawdown", f"{tearsheet.max_drawdown * 100:.2f}%")
    if tearsheet.max_drawdown_duration_days is not None:
        table.add_row("Max DD Duration", f"{tearsheet.max_drawdown_duration_days} days")
    if tearsheet.volatility_annual is not None:
        table.add_row("Annual Volatility", f"{tearsheet.volatility_annual * 100:.2f}%")


def _add_winloss_metrics(table: Table, tearsheet: TearSheet) -> None:
    """Add win/loss metrics to table."""
    table.add_row("[bold]Win/Loss[/bold]", "")
    if tearsheet.win_rate is not None:
        table.add_row("Win Rate", f"{tearsheet.win_rate * 100:.2f}%")
    if tearsheet.profit_factor is not None:
        table.add_row("Profit Factor", f"{tearsheet.profit_factor:.2f}")
    if tearsheet.avg_win is not None:
        table.add_row("Avg Win", f"{tearsheet.avg_win * 100:.2f}%")
    if tearsheet.avg_loss is not None:
        table.add_row("Avg Loss", f"{tearsheet.avg_loss * 100:.2f}%")
    if tearsheet.best_day is not None:
        table.add_row("Best Day", f"{tearsheet.best_day * 100:.2f}%")
    if tearsheet.worst_day is not None:
        table.add_row("Worst Day", f"{tearsheet.worst_day * 100:.2f}%")


def _add_benchmark_metrics(table: Table, tearsheet: TearSheet) -> None:
    """Add benchmark metrics to table."""
    if not tearsheet.benchmark_symbol:
        return

    table.add_row(f"[bold]Benchmark ({tearsheet.benchmark_symbol})[/bold]", "")
    if tearsheet.benchmark_cagr is not None:
        table.add_row("Benchmark CAGR", f"{tearsheet.benchmark_cagr * 100:.2f}%")
    if tearsheet.benchmark_sharpe is not None:
        table.add_row("Benchmark Sharpe", f"{tearsheet.benchmark_sharpe:.2f}")
    if tearsheet.alpha is not None:
        table.add_row("Alpha", f"{tearsheet.alpha:.4f}")
    if tearsheet.beta is not None:
        table.add_row("Beta", f"{tearsheet.beta:.2f}")


def _print_summary(tearsheet: TearSheet) -> None:
    """Print tearsheet summary.

    Args:
        tearsheet: TearSheet object
    """
    table = Table(
        title=f"{tearsheet.symbol} Performance Summary", show_header=True, header_style="bold magenta"
    )
    table.add_column("Metric", style="cyan", width=30)
    table.add_column("Value", style="white", justify="right")

    table.add_row("Period", f"{tearsheet.start_date.date()} to {tearsheet.end_date.date()}")
    table.add_row("", "")

    _add_performance_metrics(table, tearsheet)
    table.add_row("", "")

    _add_risk_metrics(table, tearsheet)
    table.add_row("", "")

    _add_winloss_metrics(table, tearsheet)

    if tearsheet.benchmark_symbol:
        table.add_row("", "")
        _add_benchmark_metrics(table, tearsheet)

    console.print(table)


async def _save_to_database(tearsheet: TearSheet) -> None:
    """Save tearsheet to database.

    Args:
        tearsheet: TearSheet object to save
    """
    from src.database.connection import get_session
    from src.database.repositories.tearsheet import TearSheetRepository

    try:
        async with get_session() as session:
            repo = TearSheetRepository(session)
            await repo.create(tearsheet)
            console.print("[green]Tearsheet saved to database[/green]")
    except Exception as e:
        logger.opt(exception=True).warning(f"Failed to save tearsheet to database: {e}")
        console.print(f"[yellow]Warning: Could not save to database: {e}[/yellow]")
