"""Result box widget for analysis output."""

from typing import TYPE_CHECKING

from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from src.strategies.momentum import Signal

if TYPE_CHECKING:
    from src.workflows.types import TradingWorkflowResult

SIGNAL_COLORS = {
    Signal.BUY: "green",
    Signal.SELL: "red",
    Signal.HOLD: "yellow",
}


class ResultBox(Static):
    """Clean result display with bordered tables."""

    DEFAULT_CSS = """
    ResultBox {
        background: transparent;
        padding: 0;
        margin: 1 0;
        height: auto;
        width: 100%;
    }

    ResultBox .summary {
        color: #C5CDD9;
        margin-bottom: 1;
    }

    ResultBox .signal-buy {
        color: #27AE60;
    }

    ResultBox .signal-sell {
        color: #E74C3C;
    }

    ResultBox .signal-hold {
        color: #F39C12;
    }

    ResultBox .warning {
        color: #F39C12;
    }
    """

    def __init__(self, result: "TradingWorkflowResult") -> None:
        """Initialize result box."""
        super().__init__()
        self._result = result

    def _signal_class(self) -> str:
        """Get signal class name."""
        signal = self._result.decision.action
        if signal == Signal.BUY:
            return "signal-buy"
        if signal == Signal.SELL:
            return "signal-sell"
        return "signal-hold"

    def _build_snapshot_table(self) -> Table:
        """Build snapshot table with signal, confidence, risk."""
        result = self._result
        signal = result.decision.action
        signal_color = SIGNAL_COLORS.get(signal, "white")

        table = Table(title="Snapshot", expand=True, show_header=False, border_style="dim")
        table.add_column("Field", style="dim")
        table.add_column("Value", justify="right")

        table.add_row("Signal", Text(signal.value, style=f"bold {signal_color}"))
        table.add_row("Confidence", f"{result.decision.confidence:.0%}")
        table.add_row("Risk Level", result.decision.risk_level)

        return table

    def _build_technical_table(self) -> Table:
        """Build technical indicators table."""
        result = self._result
        rsi_str = f"{result.technical.rsi:.1f}" if result.technical.rsi is not None else "N/A"
        macd = result.technical.macd_hist
        macd_str = f"{macd:.4f}" if macd is not None else "N/A"

        table = Table(title="Technical Indicators", expand=True, show_header=False, border_style="dim")
        table.add_column("Field", style="dim")
        table.add_column("Value", justify="right")

        table.add_row("RSI", rsi_str)
        table.add_row("MACD Hist", macd_str)
        table.add_row("Tech Signal", result.technical.signal.value)

        return table

    def _build_sentiment_table(self) -> Table:
        """Build sentiment table."""
        result = self._result

        table = Table(title="Sentiment", expand=True, show_header=False, border_style="dim")
        table.add_column("Field", style="dim")
        table.add_column("Value", justify="right")

        table.add_row("Overall", result.sentiment.overall_sentiment)
        table.add_row("Score", f"{result.sentiment.sentiment_score:.2f}")
        table.add_row("Articles", str(result.sentiment.article_count))

        return table

    def _build_news_table(self) -> Table | None:
        """Build news themes table if themes exist."""
        result = self._result
        if not result.news.key_themes:
            return None

        table = Table(title="News Themes", expand=True, show_header=False, border_style="dim")
        table.add_column("Themes")

        themes = ", ".join(result.news.key_themes[:3])
        table.add_row(themes)

        return table

    def compose(self) -> ComposeResult:
        """Compose the result display."""
        result = self._result
        signal = result.decision.action.value
        signal_class = self._signal_class()

        with Vertical():
            yield Static(
                f"● {result.symbol} analysis complete. Recommendation: {signal}",
                classes=f"summary {signal_class}",
            )

            yield Static(self._build_snapshot_table())
            yield Static(self._build_technical_table())
            yield Static(self._build_sentiment_table())

            news_table = self._build_news_table()
            if news_table:
                yield Static(news_table)

            if result.warnings:
                yield Static("⚠️ Incomplete Data", classes="warning")
                for warning in result.warnings:
                    yield Static(f"  • {warning}", classes="warning")

            reasoning_table = Table(
                title="Reasoning", expand=True, show_header=False, border_style="dim", padding=(0, 1)
            )
            reasoning_table.add_column("Points")
            for point in result.decision.reasoning:
                reasoning_table.add_row(f"• {point}")
            yield Static(reasoning_table)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"ResultBox(symbol={self._result.symbol})"
