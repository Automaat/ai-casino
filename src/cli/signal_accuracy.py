"""CLI command for displaying signal accuracy metrics."""

from rich.console import Console
from rich.table import Table

from src.cache.historical import HistoricalCache
from src.metrics.signal_accuracy import SignalAccuracyCalculator

console = Console()


def signal_accuracy(window: str = "30d") -> None:
    """Display signal accuracy metrics.

    Args:
        window: Time window (7d/30d/90d/all)
    """
    cache = HistoricalCache()
    calculator = SignalAccuracyCalculator(cache)
    metrics = calculator.calculate(window)

    console.print(f"\n[bold cyan]Signal Accuracy Metrics ({window})[/bold cyan]\n")

    if metrics.total_signals == 0:
        console.print("[yellow]No signals found for the specified window[/yellow]")
        return

    console.print(f"Total signals: {metrics.total_signals}\n")

    # Hit Rates Table
    hit_table = Table(title="Hit Rates by Horizon", show_header=True, header_style="bold magenta")
    hit_table.add_column("Signal Type", style="cyan")
    hit_table.add_column("1 Day", justify="right")
    hit_table.add_column("5 Days", justify="right")
    hit_table.add_column("20 Days", justify="right")

    hit_table.add_row(
        "BUY",
        f"{metrics.buy_hit_rate_1d:.1%}",
        f"{metrics.buy_hit_rate_5d:.1%}",
        f"{metrics.buy_hit_rate_20d:.1%}",
    )
    hit_table.add_row(
        "SELL",
        f"{metrics.sell_hit_rate_1d:.1%}",
        f"{metrics.sell_hit_rate_5d:.1%}",
        f"{metrics.sell_hit_rate_20d:.1%}",
    )

    console.print(hit_table)
    console.print()

    # Average Returns Table
    ret_table = Table(title="Average Returns", show_header=True, header_style="bold magenta")
    ret_table.add_column("Horizon", style="cyan")
    ret_table.add_column("Avg Return", justify="right")

    for horizon, avg_ret in [
        ("1 Day", metrics.avg_return_1d),
        ("5 Days", metrics.avg_return_5d),
        ("20 Days", metrics.avg_return_20d),
    ]:
        color = "green" if avg_ret > 0 else "red" if avg_ret < 0 else "white"
        ret_table.add_row(horizon, f"[{color}]{avg_ret:+.2f}%[/{color}]")

    console.print(ret_table)
    console.print()

    # Confidence Calibration Table
    if metrics.calibration_curve:
        cal_table = Table(title="Confidence Calibration (5d)", show_header=True, header_style="bold magenta")
        cal_table.add_column("Confidence Range", style="cyan")
        cal_table.add_column("Hit Rate", justify="right")

        for bucket, hit_rate in sorted(metrics.calibration_curve.items()):
            cal_table.add_row(bucket, f"{hit_rate:.1%}")

        console.print(cal_table)
        console.print()

    # Strategy Accuracy Table
    if metrics.strategy_accuracy:
        strat_table = Table(title="Strategy Accuracy (5d)", show_header=True, header_style="bold magenta")
        strat_table.add_column("Strategy", style="cyan")
        strat_table.add_column("Hit Rate", justify="right")

        for strategy, hit_rate in sorted(metrics.strategy_accuracy.items(), key=lambda x: x[1], reverse=True):
            strat_table.add_row(strategy, f"{hit_rate:.1%}")

        console.print(strat_table)
        console.print()

    # Regime Accuracy Table
    if metrics.regime_accuracy:
        regime_table = Table(title="Regime Accuracy (5d)", show_header=True, header_style="bold magenta")
        regime_table.add_column("Regime", style="cyan")
        regime_table.add_column("Hit Rate", justify="right")

        for regime, hit_rate in sorted(metrics.regime_accuracy.items(), key=lambda x: x[1], reverse=True):
            regime_table.add_row(regime, f"{hit_rate:.1%}")

        console.print(regime_table)
        console.print()
