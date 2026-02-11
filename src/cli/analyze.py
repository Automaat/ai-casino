"""Analyze subcommand for stock analysis."""

import asyncio
import os
import sys
from collections.abc import Callable
from typing import Annotated

import typer
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.di.container import AppContainer, create_container
from src.metrics.execution import WorkflowExecutionMetrics
from src.metrics.tracker import MetricsTracker
from src.utils.logging import sanitize_log_record
from src.workflows import TradingWorkflow
from src.workflows.types import TradingWorkflowResult

console = Console()


def _print_regime_analysis(result: TradingWorkflowResult) -> None:
    """Print regime analysis if available."""
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


def _print_momentum_technical(result: TradingWorkflowResult) -> None:
    """Print momentum strategy technical analysis."""
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


def _print_ensemble_technical(result: TradingWorkflowResult) -> None:
    """Print ensemble strategy technical analysis."""
    ensemble = result.technical.ensemble_result
    if not ensemble:
        return

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


def _print_sentiment(result: TradingWorkflowResult) -> None:
    """Print sentiment analysis."""
    sentiment_table = Table(title="Sentiment Analysis", show_header=True)
    sentiment_table.add_column("Metric", style="cyan")
    sentiment_table.add_column("Value", style="yellow")

    sentiment_table.add_row("Overall", f"[bold]{result.sentiment.overall_sentiment}[/bold]")
    sentiment_table.add_row("Score", f"{result.sentiment.sentiment_score:.2f}")
    sentiment_table.add_row("Articles", str(result.sentiment.article_count))
    sentiment_table.add_row("Positive %", f"{result.sentiment.positive_ratio * 100:.1f}%")
    sentiment_table.add_row("Negative %", f"{result.sentiment.negative_ratio * 100:.1f}%")

    console.print(sentiment_table)


def _print_news(result: TradingWorkflowResult) -> None:
    """Print news analysis."""
    news_table = Table(title="News Analysis", show_header=True)
    news_table.add_column("Aspect", style="cyan")
    news_table.add_column("Details", style="yellow")

    news_table.add_row("Key Themes", ", ".join(result.news.key_themes[:3]))
    news_table.add_row("Impact", result.news.impact_assessment[:100])
    news_table.add_row("Recommendation", result.news.recommendation[:100])

    console.print(news_table)


def _print_fundamental(result: TradingWorkflowResult) -> None:
    """Print fundamental analysis."""
    if not result.fundamental:
        console.print(
            Panel(
                "[yellow]Fundamental analysis unavailable (API rate limit)[/yellow]\n"
                "Decision based on remaining available signals (excluding fundamentals).",
                title="[bold yellow]⚠️ Fundamental Analysis[/bold yellow]",
                border_style="yellow",
            )
        )
        return

    fundamental_table = Table(title="Fundamental Analysis", show_header=True)
    fundamental_table.add_column("Metric", style="cyan")
    fundamental_table.add_column("Value", style="yellow")

    fundamental_table.add_row("Valuation", f"[bold]{result.fundamental.valuation}[/bold]")
    if result.fundamental.pe_ratio:
        fundamental_table.add_row("P/E Ratio", f"{result.fundamental.pe_ratio:.2f}")
    if result.fundamental.eps:
        fundamental_table.add_row("EPS", f"${result.fundamental.eps:.2f}")
    if result.fundamental.revenue_growth_yoy:
        growth = f"{result.fundamental.revenue_growth_yoy * 100:.1f}%"
        fundamental_table.add_row("Revenue Growth YoY", growth)
    if result.fundamental.earnings_growth_yoy:
        growth = f"{result.fundamental.earnings_growth_yoy * 100:.1f}%"
        fundamental_table.add_row("Earnings Growth YoY", growth)
    if result.fundamental.debt_to_equity:
        fundamental_table.add_row("Debt-to-Equity", f"{result.fundamental.debt_to_equity:.2f}")
    if result.fundamental.current_ratio:
        fundamental_table.add_row("Current Ratio", f"{result.fundamental.current_ratio:.2f}")
    fundamental_table.add_row("Confidence", f"{result.fundamental.confidence:.2f}")

    console.print(fundamental_table)
    console.print(Panel(result.fundamental.interpretation, title="Fundamental Interpretation"))


def _print_risk(result: TradingWorkflowResult) -> None:
    """Print risk management."""
    risk_table = Table(title="Risk Management", show_header=True)
    risk_table.add_column("Metric", style="cyan")
    risk_table.add_column("Value", style="yellow")

    approval_status = "APPROVED" if result.risk.validation.approved else "REJECTED"
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
        warnings_text = "\n".join(f"- {w}" for w in result.risk.validation.warnings)
        console.print(
            Panel(warnings_text, title="[bold yellow]Risk Warnings[/bold yellow]", border_style="yellow")
        )


def _print_data_warnings(result: TradingWorkflowResult) -> None:
    """Print warnings about incomplete data."""
    if not result.warnings:
        return

    warnings_text = "\n".join(f"• {w}" for w in result.warnings)
    console.print(
        Panel(
            f"[yellow]{warnings_text}[/yellow]\n\n"
            "[dim]Decision based on available data only. Results may be less reliable.[/dim]",
            title="[bold yellow]⚠️ Incomplete Data[/bold yellow]",
            border_style="yellow",
        )
    )


def _print_trump(result: TradingWorkflowResult) -> None:
    """Print Trump analysis if available."""
    if not result.trump:
        return

    trump_table = Table(title="Trump Social Media Analysis", show_header=True)
    trump_table.add_column("Metric", style="cyan")
    trump_table.add_column("Value", style="yellow")

    relevant_str = "[green]Yes[/green]" if result.trump.market_relevant else "[dim]No[/dim]"
    trump_table.add_row("Market Relevant", relevant_str)

    signal_color = {"BUY": "green", "SELL": "red", "HOLD": "yellow"}[result.trump.signal.value]
    trump_table.add_row("Signal", f"[{signal_color}]{result.trump.signal.value}[/{signal_color}]")
    trump_table.add_row("Sentiment", result.trump.sentiment)
    trump_table.add_row("Confidence", f"{result.trump.confidence:.2f}")
    trump_table.add_row("Posts Analyzed", str(result.trump.post_count))

    if result.trump.mentioned_tickers:
        trump_table.add_row("Mentioned Tickers", ", ".join(result.trump.mentioned_tickers))

    console.print(trump_table)

    if result.trump.key_phrases:
        phrases_text = "\n".join(f"• {p[:100]}" for p in result.trump.key_phrases[:5])
        console.print(Panel(phrases_text, title="Key Phrases"))

    console.print(Panel(result.trump.interpretation, title="Trump Analysis Interpretation"))


def _print_decision(result: TradingWorkflowResult) -> None:
    """Print final trading decision."""
    decision_color = {"BUY": "green", "SELL": "red", "HOLD": "yellow", "WAIT": "yellow"}
    display_action = result.decision.display_action
    action_color = decision_color[display_action]

    incomplete_notice = ""
    if result.has_incomplete_data:
        incomplete_notice = "[yellow](based on incomplete data)[/yellow]\n\n"

    decision_panel = Panel(
        f"[bold {action_color}]{display_action}[/bold {action_color}]\n\n"
        f"{incomplete_notice}"
        f"Confidence: {result.decision.confidence:.2f}\n"
        f"Risk Level: {result.decision.risk_level}\n\n"
        f"{result.decision.reasoning}",
        title="[bold]Final Trading Decision[/bold]",
        border_style=action_color,
    )

    console.print(decision_panel)


def _print_order(result: TradingWorkflowResult) -> None:
    """Print order execution details."""
    if not result.order:
        return

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


def _print_execution_metrics(metrics: WorkflowExecutionMetrics) -> None:
    """Print execution performance metrics summary."""
    metrics_table = Table(title="Execution Metrics", show_header=True)
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Value", style="yellow")

    metrics_table.add_row("Total time", f"{metrics.total_latency_ms / 1000:.1f}s")
    metrics_table.add_row("LLM calls", str(len(metrics.llm_calls)))
    metrics_table.add_row("Input tokens", f"{metrics.total_input_tokens:,}")
    metrics_table.add_row("Output tokens", f"{metrics.total_output_tokens:,}")
    metrics_table.add_row("Estimated cost", f"${metrics.total_estimated_cost_usd:.4f}")

    if metrics.agent_timings:
        slowest = max(metrics.agent_timings, key=lambda a: a.latency_ms)
        metrics_table.add_row("Slowest agent", f"{slowest.agent_name} ({slowest.latency_ms / 1000:.1f}s)")

    console.print(metrics_table)

    if metrics.sub_operations:
        sub_ops_text = ", ".join(f"{op.name}({op.latency_ms / 1000:.1f}s)" for op in metrics.sub_operations)
        console.print(f"[dim]Sub-operations: {sub_ops_text}[/dim]")

    if metrics.pipeline_stages:
        stages_text = ", ".join(f"{s.stage}({s.latency_ms / 1000:.1f}s)" for s in metrics.pipeline_stages)
        console.print(f"[dim]Pipeline: {stages_text}[/dim]")

    console.print("[dim]Metrics saved to: logs/execution_metrics.jsonl[/dim]\n")


def _print_result(result: TradingWorkflowResult, use_meta_agent: bool = True) -> None:
    """Print trading analysis results."""
    console.print(f"\n[bold cyan]Trading Analysis for {result.symbol}[/bold cyan]\n")

    if use_meta_agent and result.regime:
        _print_regime_analysis(result)

    if result.technical.ensemble_result:
        _print_ensemble_technical(result)
    else:
        _print_momentum_technical(result)

    console.print(Panel(result.technical.interpretation, title="Technical Interpretation"))
    _print_sentiment(result)
    _print_news(result)
    _print_trump(result)
    _print_fundamental(result)
    _print_data_warnings(result)
    _print_risk(result)
    _print_decision(result)
    _print_order(result)


def _print_metrics_summary(tracker: MetricsTracker) -> None:
    """Print performance metrics summary."""
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


def _select_workflow_factory(
    container: AppContainer,
    use_meta_agent: bool,
    trump_mode: bool,
) -> Callable[..., TradingWorkflow]:
    """Select workflow factory based on CLI flags."""
    if trump_mode:
        return container.workflow_trump
    if use_meta_agent:
        return container.workflow_meta
    return container.workflow_momentum


async def _analyze_stock(
    symbol: str,
    period_days: int,
    enable_trading: bool,
    show_metrics: bool,
    use_meta_agent: bool,
    trump_mode: bool = False,
    execution_metrics: bool = False,
) -> None:
    """Run stock analysis."""
    if execution_metrics:
        os.environ["EXECUTION_METRICS"] = "true"
    mode_str = "meta-agent" if use_meta_agent else "momentum"
    trump_str = "+trump" if trump_mode else ""
    console.print(f"\n[bold]Initializing trading system ({mode_str}{trump_str} mode)...[/bold]")

    container = create_container()

    broker = container.alpaca_broker() if enable_trading else None
    if enable_trading:
        if os.getenv("ALPACA_API_KEY") and os.getenv("ALPACA_SECRET_KEY"):
            console.print("[bold green]Paper trading enabled[/bold green]")
        else:
            console.print("[yellow]Warning: Trading enabled but Alpaca credentials not found[/yellow]")

    metrics_tracker = MetricsTracker() if show_metrics else None

    workflow_factory = _select_workflow_factory(container, use_meta_agent, trump_mode)
    workflow = workflow_factory(
        broker=broker,
        metrics_tracker=metrics_tracker,
    )

    console.print(f"\n[bold]Analyzing {symbol}...[/bold]\n")

    result = await workflow.analyze(symbol, period_days)

    _print_result(result, use_meta_agent=use_meta_agent)

    if result.execution_metrics:
        _print_execution_metrics(result.execution_metrics)

    if metrics_tracker:
        _print_metrics_summary(metrics_tracker)


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
    from dotenv import load_dotenv

    load_dotenv()

    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=os.getenv("LOG_LEVEL", "INFO"),
        filter=sanitize_log_record,
    )

    try:
        asyncio.run(
            _analyze_stock(
                symbol.upper(), period, trade, show_metrics, not no_meta_agent, trump_mode, metrics
            )
        )
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        logger.exception("Analysis failed")
        raise typer.Exit(1) from e
