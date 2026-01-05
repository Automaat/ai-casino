"""CLI for agentic trading system."""

import sys

from dotenv import load_dotenv
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.data.market import MarketDataFetcher
from src.data.news import NewsFetcher
from src.models.llm import LLMClient
from src.models.sentiment import FinBERTSentiment
from src.workflows.trading import TradingWorkflow

load_dotenv()

console = Console()


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


def print_result(result) -> None:  # noqa: ANN001
    """Print trading analysis results.

    Args:
        result: TradingWorkflowResult
    """
    console.print(f"\n[bold cyan]Trading Analysis for {result.symbol}[/bold cyan]\n")

    tech_table = Table(title="Technical Analysis", show_header=True)
    tech_table.add_column("Metric", style="cyan")
    tech_table.add_column("Value", style="yellow")

    tech_table.add_row("Signal", f"[bold]{result.technical.signal.value}[/bold]")
    tech_table.add_row("RSI", f"{result.technical.rsi:.2f}")
    tech_table.add_row("MACD Histogram", f"{result.technical.macd_hist:.4f}")
    tech_table.add_row("Confidence", f"{result.technical.confidence:.2f}")

    console.print(tech_table)
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

    decision_color = {
        "BUY": "green",
        "SELL": "red",
        "HOLD": "yellow",
    }

    decision_panel = Panel(
        f"[bold {decision_color[result.decision.action.value]}]{result.decision.action.value}[/bold {decision_color[result.decision.action.value]}]\n\n"
        f"Confidence: {result.decision.confidence:.2f}\n"
        f"Risk Level: {result.decision.risk_level}\n\n"
        f"{result.decision.reasoning}",
        title="[bold]Final Trading Decision[/bold]",
        border_style=decision_color[result.decision.action.value],
    )

    console.print(decision_panel)


def analyze_stock(symbol: str, period_days: int = 90) -> None:
    """Analyze a stock and print results.

    Args:
        symbol: Stock ticker symbol
        period_days: Days of historical data
    """
    try:
        console.print("\n[bold]Initializing trading system...[/bold]")

        llm_client = LLMClient()
        market_fetcher = MarketDataFetcher(use_alpha_vantage=False)
        news_fetcher = NewsFetcher()
        finbert = FinBERTSentiment()

        workflow = TradingWorkflow(llm_client, market_fetcher, news_fetcher, finbert)

        console.print(f"\n[bold]Analyzing {symbol}...[/bold]\n")

        result = workflow.analyze(symbol, period_days)

        print_result(result)

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        logger.exception("Analysis failed")
        sys.exit(1)


def main() -> None:
    """Main CLI entry point."""
    if len(sys.argv) < 2:
        console.print("[bold red]Error:[/bold red] Missing symbol argument")
        console.print("\nUsage: python -m src.main <SYMBOL> [--period DAYS]")
        console.print("\nExample: python -m src.main AAPL --period 90")
        sys.exit(1)

    symbol = sys.argv[1].upper()

    period_days = 90
    if "--period" in sys.argv:
        try:
            period_idx = sys.argv.index("--period")
            period_days = int(sys.argv[period_idx + 1])
        except (IndexError, ValueError):
            console.print(
                "[bold yellow]Warning:[/bold yellow] Invalid period, using default 90 days"
            )

    setup_logging()

    analyze_stock(symbol, period_days)


if __name__ == "__main__":
    main()
