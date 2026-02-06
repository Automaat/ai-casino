"""Portfolio optimization CLI command."""

import json
import os

import typer
from loguru import logger
from rich.console import Console
from rich.table import Table

from src.data.broker import AlpacaBroker
from src.data.market import MarketDataFetcher
from src.optimization.portfolio import OptimizedPortfolio, PortfolioOptimizer

console = Console()

VALID_METHODS = ["max_sharpe", "min_volatility", "hrp"]
MIN_SYMBOLS = 2


def _validate_inputs(symbols: str, method: str) -> list[str]:
    """Validate and parse inputs.

    Args:
        symbols: Comma-separated symbols
        method: Optimization method

    Returns:
        List of validated symbols

    Raises:
        typer.Exit: On validation failure
    """
    symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    # Deduplicate while preserving order
    seen = set()
    symbol_list = [s for s in symbol_list if s not in seen and not seen.add(s)]

    if len(symbol_list) < MIN_SYMBOLS:
        console.print("[bold red]Error:[/bold red] Portfolio optimization requires at least 2 symbols")
        raise typer.Exit(1)

    if method not in VALID_METHODS:
        console.print(
            f"[bold red]Error:[/bold red] Invalid method '{method}'. Must be one of: {VALID_METHODS}"
        )
        raise typer.Exit(1)

    return symbol_list


def _initialize_broker(rebalance: bool, rebalance_from: str | None) -> AlpacaBroker | None:
    """Initialize broker if rebalancing requested.

    Args:
        rebalance: Whether to rebalance
        rebalance_from: Custom rebalance weights

    Returns:
        Broker instance or None

    Raises:
        typer.Exit: If rebalance requested but API keys missing
    """
    if not (rebalance or rebalance_from):
        return None

    if os.getenv("ALPACA_API_KEY") and os.getenv("ALPACA_SECRET_KEY"):
        broker = AlpacaBroker(paper=True)
        logger.info("Alpaca broker initialized for rebalancing")
        return broker

    if rebalance:
        console.print("[bold red]Error:[/bold red] --rebalance requires ALPACA_API_KEY and ALPACA_SECRET_KEY")
        raise typer.Exit(1)

    return None


def optimize_portfolio(
    symbols: str,
    method: str = "max_sharpe",
    period: int = 365,
    rebalance: bool = False,
    rebalance_from: str | None = None,
) -> None:
    """Optimize portfolio allocation.

    Args:
        symbols: Comma-separated stock ticker symbols
        method: Optimization method (max_sharpe, min_volatility, hrp)
        period: Historical data period in days
        rebalance: Compare with current Alpaca portfolio
        rebalance_from: Custom current weights JSON
    """
    symbol_list = _validate_inputs(symbols, method)

    console.print("\n[bold cyan]Portfolio Optimization[/bold cyan]")
    console.print(f"Symbols: {', '.join(symbol_list)}")
    console.print(f"Method: {method}")
    console.print(f"Period: {period} days\n")

    try:
        market_fetcher = MarketDataFetcher(use_alpha_vantage=False)
        broker = _initialize_broker(rebalance, rebalance_from)
        optimizer = PortfolioOptimizer(market_fetcher, broker=broker, period_days=period)

        # Run optimization
        console.print("[bold]Fetching market data and optimizing...[/bold]")
        if method == "max_sharpe":
            result = optimizer.optimize_max_sharpe(symbol_list)
        elif method == "min_volatility":
            result = optimizer.optimize_min_volatility(symbol_list)
        else:  # hrp
            result = optimizer.optimize_hrp(symbol_list)

        _print_portfolio_result(result)

        # Handle rebalancing
        if rebalance or rebalance_from:
            current_weights = None
            if rebalance_from:
                current_weights = json.loads(rebalance_from)
                logger.info(f"Using custom current weights: {current_weights}")

            console.print("\n[bold cyan]Rebalancing Analysis[/bold cyan]\n")

            if rebalance and broker:
                _print_current_portfolio(broker)

            rebalance_instructions = optimizer.calculate_rebalance(result, current=current_weights)
            _print_rebalance_instructions(rebalance_instructions)

    except ValueError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1) from e
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        console.print(f"[bold red]Error:[/bold red] Optimization failed: {e}")
        raise typer.Exit(1) from e


def _print_portfolio_result(portfolio: OptimizedPortfolio) -> None:
    """Print optimized portfolio result."""
    # Portfolio allocations table
    table = Table(title="Optimized Portfolio", show_header=True, header_style="bold cyan")
    table.add_column("Symbol", style="cyan", width=10)
    table.add_column("Weight", justify="right", style="green")
    table.add_column("Expected Return", justify="right")

    for allocation in portfolio.allocations:
        weight_str = f"{allocation.weight * 100:.2f}%"
        return_str = f"{allocation.expected_return * 100:.2f}%" if allocation.expected_return else "N/A"
        table.add_row(allocation.symbol, weight_str, return_str)

    console.print(table)

    # Portfolio metrics
    console.print("\n[bold]Portfolio Metrics:[/bold]")
    console.print(f"  Expected Annual Return: [green]{portfolio.expected_return * 100:.2f}%[/green]")
    console.print(f"  Expected Volatility: [yellow]{portfolio.expected_volatility * 100:.2f}%[/yellow]")
    console.print(f"  Sharpe Ratio: [cyan]{portfolio.sharpe_ratio:.3f}[/cyan]")
    console.print(f"  Total Weight: {portfolio.total_weight * 100:.2f}%")
    console.print(f"  Method: {portfolio.method}")


def _print_current_portfolio(broker: AlpacaBroker) -> None:
    """Print current Alpaca portfolio positions."""
    account_info = broker.get_account_info()

    table = Table(title="Current Portfolio (Alpaca)", show_header=True, header_style="bold cyan")
    table.add_column("Symbol", style="cyan", width=10)
    table.add_column("Qty", justify="right")
    table.add_column("Market Value", justify="right", style="green")
    table.add_column("Weight", justify="right", style="yellow")

    portfolio_value = account_info.portfolio_value
    for symbol, position in account_info.positions.items():
        weight = position.market_value / portfolio_value if portfolio_value > 0 else 0
        table.add_row(symbol, f"{position.qty:.0f}", f"${position.market_value:,.2f}", f"{weight * 100:.2f}%")

    console.print(table)
    console.print(f"\n[bold]Total Portfolio Value:[/bold] ${portfolio_value:,.2f}\n")


def _print_rebalance_instructions(rebalances: list) -> None:
    """Print rebalancing instructions."""
    # Filter out HOLD actions
    actionable = [r for r in rebalances if r.action != "HOLD"]

    if not actionable:
        console.print("[green]Portfolio is already balanced (no rebalancing needed)[/green]")
        return

    table = Table(title="Rebalancing Instructions", show_header=True, header_style="bold cyan")
    table.add_column("Symbol", style="cyan", width=10)
    table.add_column("Current", justify="right")
    table.add_column("Target", justify="right")
    table.add_column("Action", justify="center")
    table.add_column("Delta", justify="right")
    table.add_column("Shares", justify="right")

    for rebalance in actionable:
        action_color = "green" if rebalance.action == "BUY" else "red"
        delta_str = f"{rebalance.delta * 100:+.2f}%"
        shares_str = f"{rebalance.shares_to_trade:+d}" if rebalance.shares_to_trade is not None else "N/A"

        table.add_row(
            rebalance.symbol,
            f"{rebalance.current_weight * 100:.2f}%",
            f"{rebalance.target_weight * 100:.2f}%",
            f"[{action_color}]{rebalance.action}[/{action_color}]",
            delta_str,
            shares_str,
        )

    console.print(table)
