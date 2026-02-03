"""CLI for agentic trading system."""

import asyncio
import os
import sys

from dotenv import load_dotenv
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.data.broker import AlpacaBroker
from src.data.fundamental import FundamentalDataFetcher
from src.data.market import MarketDataFetcher
from src.data.news import NewsFetcher
from src.metrics.tracker import MetricsTracker
from src.models.llm import LLMClient
from src.models.sentiment import FinBERTSentiment
from src.workflows.trading import TradingWorkflow

load_dotenv()

console = Console()


def _print_regime_analysis(result) -> None:  # noqa: ANN001
    """Print regime analysis if available.

    Args:
        result: TradingWorkflowResult
    """
    if not result.regime:
        return

    regime_table = Table(title="Market Regime Analysis", show_header=True)
    regime_table.add_column("Metric", style="cyan")
    regime_table.add_column("Value", style="yellow")

    regime_color = {
        "TRENDING_BULLISH": "green",
        "TRENDING_BEARISH": "red",
        "RANGING": "yellow",
        "HIGH_VOLATILITY": "magenta",
    }
    color = regime_color.get(result.regime.regime.value, "white")

    regime_table.add_row("Regime", f"[bold {color}]{result.regime.regime.value}[/bold {color}]")
    regime_table.add_row("Confidence", f"{result.regime.confidence:.2f}")
    regime_table.add_row("ADX", f"{result.regime.indicators.adx:.2f}")
    regime_table.add_row(
        "+DI / -DI", f"{result.regime.indicators.plus_di:.2f} / {result.regime.indicators.minus_di:.2f}"
    )
    regime_table.add_row("ATR Ratio", f"{result.regime.indicators.atr_ratio:.2f}")

    if result.strategy_used:
        regime_table.add_row("Strategy Selected", f"[bold]{result.strategy_used}[/bold]")

    console.print(regime_table)
    console.print(Panel(result.regime.reasoning, title="Regime Reasoning"))


def _print_momentum_technical(result) -> None:  # noqa: ANN001
    """Print momentum strategy technical analysis.

    Args:
        result: TradingWorkflowResult
    """
    tech_table = Table(title="Technical Analysis (Momentum)", show_header=True)
    tech_table.add_column("Metric", style="cyan")
    tech_table.add_column("Value", style="yellow")

    tech_table.add_row("Signal", f"[bold]{result.technical.signal.value}[/bold]")
    if result.technical.rsi is not None:
        tech_table.add_row("RSI", f"{result.technical.rsi:.2f}")
    if result.technical.macd_hist is not None:
        tech_table.add_row("MACD Histogram", f"{result.technical.macd_hist:.4f}")
    tech_table.add_row("Confidence", f"{result.technical.confidence:.2f}")

    console.print(tech_table)


def _print_ensemble_technical(result) -> None:  # noqa: ANN001
    """Print ensemble strategy technical analysis.

    Args:
        result: TradingWorkflowResult
    """
    ensemble = result.technical.ensemble_result

    tech_table = Table(title="Technical Analysis (Ensemble)", show_header=True)
    tech_table.add_column("Metric", style="cyan")
    tech_table.add_column("Value", style="yellow")

    tech_table.add_row("Final Signal", f"[bold]{result.technical.signal.value}[/bold]")
    tech_table.add_row("Confidence", f"{result.technical.confidence:.2f}")
    tech_table.add_row("Agreement Ratio", f"{ensemble.agreement_ratio:.2f}")
    conflict_str = "[yellow]Yes[/yellow]" if ensemble.conflict_resolved else "No"
    tech_table.add_row("Conflict Resolved", conflict_str)

    console.print(tech_table)

    strategy_table = Table(title="Strategy Breakdown", show_header=True)
    strategy_table.add_column("Strategy", style="cyan")
    strategy_table.add_column("Signal", style="yellow")
    strategy_table.add_column("Weight", style="magenta")

    for sr in ensemble.strategy_results:
        signal_color = {"BUY": "green", "SELL": "red", "HOLD": "yellow"}[sr.signal.value]
        strategy_table.add_row(
            sr.name.replace("_", " ").title(),
            f"[{signal_color}]{sr.signal.value}[/{signal_color}]",
            f"{sr.weight:.2f}",
        )

    console.print(strategy_table)


def setup_logging(level: str = "INFO") -> None:
    """Configure logging.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
    """
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=level,
    )


def print_result(result, use_meta_agent: bool = True) -> None:  # noqa: ANN001, PLR0915, PLR0912, C901
    """Print trading analysis results.

    Args:
        result: TradingWorkflowResult
        use_meta_agent: Whether meta-agent mode was used
    """
    console.print(f"\n[bold cyan]Trading Analysis for {result.symbol}[/bold cyan]\n")

    if use_meta_agent and result.regime:
        _print_regime_analysis(result)

    if result.technical.ensemble_result:
        _print_ensemble_technical(result)
    else:
        _print_momentum_technical(result)

    console.print(Panel(result.technical.interpretation, title="Technical Interpretation"))

    sentiment_table = Table(title="Sentiment Analysis", show_header=True)
    sentiment_table.add_column("Metric", style="cyan")
    sentiment_table.add_column("Value", style="yellow")

    sentiment_table.add_row("Overall", f"[bold]{result.sentiment.overall_sentiment}[/bold]")
    sentiment_table.add_row("Score", f"{result.sentiment.sentiment_score:.2f}")
    sentiment_table.add_row("Articles", str(result.sentiment.article_count))
    sentiment_table.add_row("Positive %", f"{result.sentiment.positive_ratio * 100:.1f}%")
    sentiment_table.add_row("Negative %", f"{result.sentiment.negative_ratio * 100:.1f}%")

    console.print(sentiment_table)

    news_table = Table(title="News Analysis", show_header=True)
    news_table.add_column("Aspect", style="cyan")
    news_table.add_column("Details", style="yellow")

    news_table.add_row("Key Themes", ", ".join(result.news.key_themes[:3]))
    news_table.add_row("Impact", result.news.impact_assessment[:100])
    news_table.add_row("Recommendation", result.news.recommendation[:100])

    console.print(news_table)

    fundamental_table = Table(title="Fundamental Analysis", show_header=True)
    fundamental_table.add_column("Metric", style="cyan")
    fundamental_table.add_column("Value", style="yellow")

    fundamental_table.add_row("Valuation", f"[bold]{result.fundamental.valuation}[/bold]")
    if result.fundamental.pe_ratio:
        fundamental_table.add_row("P/E Ratio", f"{result.fundamental.pe_ratio:.2f}")
    if result.fundamental.eps:
        fundamental_table.add_row("EPS", f"${result.fundamental.eps:.2f}")
    if result.fundamental.revenue_growth_yoy:
        fundamental_table.add_row("Revenue Growth YoY", f"{result.fundamental.revenue_growth_yoy * 100:.1f}%")
    if result.fundamental.earnings_growth_yoy:
        fundamental_table.add_row(
            "Earnings Growth YoY", f"{result.fundamental.earnings_growth_yoy * 100:.1f}%"
        )
    if result.fundamental.debt_to_equity:
        fundamental_table.add_row("Debt-to-Equity", f"{result.fundamental.debt_to_equity:.2f}")
    if result.fundamental.current_ratio:
        fundamental_table.add_row("Current Ratio", f"{result.fundamental.current_ratio:.2f}")
    fundamental_table.add_row("Confidence", f"{result.fundamental.confidence:.2f}")

    console.print(fundamental_table)
    console.print(Panel(result.fundamental.interpretation, title="Fundamental Interpretation"))

    risk_table = Table(title="Risk Management", show_header=True)
    risk_table.add_column("Metric", style="cyan")
    risk_table.add_column("Value", style="yellow")

    approval_status = "✅ APPROVED" if result.risk.validation.approved else "❌ REJECTED"
    approval_color = "green" if result.risk.validation.approved else "red"

    risk_table.add_row("Approval", f"[bold {approval_color}]{approval_status}[/bold {approval_color}]")
    risk_table.add_row("Risk Level", f"[bold]{result.risk.validation.risk_level}[/bold]")
    risk_table.add_row("Risk Score", f"{result.risk.validation.risk_score:.2f}")
    risk_table.add_row("Confidence", f"{result.risk.confidence:.2f}")

    if result.risk.action.value != "HOLD":
        risk_table.add_row("Shares", str(result.risk.position_sizing.recommended_shares))
        risk_table.add_row("Position Value", f"${result.risk.position_sizing.position_value:,.2f}")
        risk_table.add_row("Risk Amount", f"${result.risk.position_sizing.risk_amount:,.2f}")
        risk_table.add_row("Risk %", f"{result.risk.position_sizing.risk_percent:.2f}%")
        risk_table.add_row("Stop-Loss", f"${result.risk.stop_loss.stop_loss_price:.2f}")
        risk_table.add_row("Stop Method", result.risk.stop_loss.methodology)

    if result.risk.validation.warnings:
        risk_table.add_row("Warnings", str(len(result.risk.validation.warnings)))

    console.print(risk_table)

    if result.risk.validation.warnings:
        warnings_text = "\n".join(f"• {w}" for w in result.risk.validation.warnings)
        console.print(
            Panel(warnings_text, title="[bold yellow]Risk Warnings[/bold yellow]", border_style="yellow")
        )

    decision_color = {
        "BUY": "green",
        "SELL": "red",
        "HOLD": "yellow",
        "WAIT": "yellow",
    }

    display_action = result.decision.display_action
    action_color = decision_color[display_action]

    decision_panel = Panel(
        f"[bold {action_color}]{display_action}[/bold {action_color}]\n\n"
        f"Confidence: {result.decision.confidence:.2f}\n"
        f"Risk Level: {result.decision.risk_level}\n\n"
        f"{result.decision.reasoning}",
        title="[bold]Final Trading Decision[/bold]",
        border_style=action_color,
    )

    console.print(decision_panel)

    if result.order:
        order_table = Table(title="Order Execution", show_header=True)
        order_table.add_column("Field", style="cyan")
        order_table.add_column("Value", style="yellow")

        order_table.add_row("Order ID", result.order.order_id)
        order_table.add_row("Symbol", result.order.symbol)
        order_table.add_row("Side", result.order.side.upper())
        order_table.add_row("Quantity", str(int(result.order.qty)))
        order_table.add_row("Status", result.order.status.upper())
        order_table.add_row("Submitted", result.order.submitted_at.strftime("%Y-%m-%d %H:%M:%S"))

        if result.order.filled_avg_price:
            order_table.add_row("Filled Price", f"${result.order.filled_avg_price:.2f}")

        console.print(order_table)


def print_metrics_summary(tracker: MetricsTracker) -> None:
    """Print performance metrics summary.

    Args:
        tracker: MetricsTracker instance
    """
    metrics = tracker.calculate_metrics("all")

    metrics_table = Table(title="Performance Metrics", show_header=True)
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Value", style="yellow")

    metrics_table.add_row("Total Decisions", str(metrics.total_decisions))
    metrics_table.add_row("Approved Trades", str(metrics.approved_trades))
    metrics_table.add_row("Closed Trades", str(metrics.closed_trades))
    metrics_table.add_row("Total PnL", f"${metrics.total_pnl:,.2f}")
    metrics_table.add_row("Win Rate", f"{metrics.win_rate:.1f}%")
    metrics_table.add_row("Sharpe Ratio", f"{metrics.sharpe_ratio:.2f}")
    metrics_table.add_row("Max Drawdown", f"{metrics.max_drawdown_percent:.2f}%")

    console.print("\n")
    console.print(metrics_table)

    console.print("\n[dim]Trades saved to: logs/trades.jsonl[/dim]")
    console.print("[dim]Metrics saved to: logs/metrics_summary.json[/dim]\n")


async def analyze_stock(
    symbol: str,
    period_days: int = 90,
    enable_trading: bool = False,
    show_metrics: bool = False,
    use_meta_agent: bool = True,
) -> None:
    """Analyze a stock and print results.

    Args:
        symbol: Stock ticker symbol
        period_days: Days of historical data
        enable_trading: Enable live trading via Alpaca
        show_metrics: Show performance metrics
        use_meta_agent: Use meta-agent for dynamic strategy selection
    """
    try:
        mode_str = "meta-agent" if use_meta_agent else "momentum"
        console.print(f"\n[bold]Initializing trading system ({mode_str} mode)...[/bold]")

        llm_client = LLMClient()
        market_fetcher = MarketDataFetcher(use_alpha_vantage=False)
        news_fetcher = NewsFetcher()
        finbert = FinBERTSentiment()
        fundamental_fetcher = FundamentalDataFetcher()

        broker = None
        if enable_trading:
            if os.getenv("ALPACA_API_KEY") and os.getenv("ALPACA_SECRET_KEY"):
                broker = AlpacaBroker(paper=True)
                console.print("[bold green]Paper trading enabled[/bold green]")
            else:
                console.print(
                    "[bold yellow]Warning: Trading enabled but Alpaca credentials not found[/bold yellow]"
                )

        metrics_tracker = MetricsTracker() if show_metrics else None

        workflow = TradingWorkflow(
            llm_client,
            market_fetcher,
            news_fetcher,
            finbert,
            fundamental_fetcher,
            broker,
            metrics_tracker,
            use_meta_agent=use_meta_agent,
        )

        console.print(f"\n[bold]Analyzing {symbol}...[/bold]\n")

        result = await workflow.analyze(symbol, period_days)

        print_result(result, use_meta_agent=use_meta_agent)

        if metrics_tracker:
            print_metrics_summary(metrics_tracker)

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        logger.exception("Analysis failed")
        sys.exit(1)


def main() -> None:
    """Main CLI entry point."""
    if len(sys.argv) < 2:
        console.print("[bold red]Error:[/bold red] Missing symbol argument")
        console.print(
            "\nUsage: python -m src.main <SYMBOL> [--period DAYS] [--trade] [--show-metrics] [--no-meta-agent]"
        )
        console.print("\nExample: python -m src.main AAPL --period 90")
        console.print("\nFlags:")
        console.print("  --period DAYS    Days of historical data (default: 90)")
        console.print("  --trade          Enable paper trading via Alpaca")
        console.print("  --show-metrics   Show performance metrics")
        console.print("  --no-meta-agent  Disable meta-agent, use momentum strategy only")
        sys.exit(1)

    symbol = sys.argv[1].upper()

    period_days = 90
    if "--period" in sys.argv:
        try:
            period_idx = sys.argv.index("--period")
            period_days = int(sys.argv[period_idx + 1])
        except (IndexError, ValueError):
            console.print("[bold yellow]Warning:[/bold yellow] Invalid period, using default 90 days")

    enable_trading = "--trade" in sys.argv
    show_metrics = "--show-metrics" in sys.argv
    use_meta_agent = "--no-meta-agent" not in sys.argv

    setup_logging()

    asyncio.run(analyze_stock(symbol, period_days, enable_trading, show_metrics, use_meta_agent))


if __name__ == "__main__":
    main()
