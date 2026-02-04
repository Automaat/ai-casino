"""Result box widget for analysis output."""

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from src.strategies.momentum import Signal
from src.workflows.trading import TradingWorkflowResult


class ResultBox(Static):
    """Clean result display with table style."""

    DEFAULT_CSS = """
    ResultBox {
        background: transparent;
        padding: 0;
        margin: 1 0;
        height: auto;
    }

    ResultBox .summary {
        color: #C5CDD9;
        margin-bottom: 1;
    }

    ResultBox .section-header {
        color: #8899A6;
        margin-top: 1;
    }

    ResultBox .table-row {
        height: 1;
    }

    ResultBox .table-label {
        color: #8899A6;
        width: 15;
    }

    ResultBox .table-value {
        color: #C5CDD9;
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

    ResultBox .reasoning {
        color: #8899A6;
        margin-top: 1;
    }

    ResultBox .warning {
        color: #F39C12;
        margin-top: 1;
    }

    ResultBox .warning-header {
        color: #F39C12;
        margin-top: 1;
    }
    """

    def __init__(self, result: TradingWorkflowResult) -> None:
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

    def compose(self) -> ComposeResult:
        """Compose the result display."""
        result = self._result
        signal = result.decision.action.value
        confidence = result.decision.confidence
        risk = result.decision.risk_level

        signal_class = self._signal_class()

        with Vertical():
            yield Static(
                f"● {result.symbol} analysis complete. Recommendation: {signal}",
                classes=f"summary {signal_class}",
            )

            yield Static("Snapshot", classes="section-header")

            yield Static(f"  {'Signal':<12} {signal}", classes=f"table-row {signal_class}")
            yield Static(f"  {'Confidence':<12} {confidence:.0%}", classes="table-row")
            yield Static(f"  {'Risk Level':<12} {risk}", classes="table-row")

            yield Static("Technical Indicators", classes="section-header")
            rsi_str = f"{result.technical.rsi:.1f}" if result.technical.rsi is not None else "N/A"
            macd = result.technical.macd_hist
            macd_str = f"{macd:.4f}" if macd is not None else "N/A"
            yield Static(f"  {'RSI':<12} {rsi_str}", classes="table-row")
            yield Static(f"  {'MACD Hist':<12} {macd_str}", classes="table-row")
            yield Static(
                f"  {'Tech Signal':<12} {result.technical.signal.value}",
                classes="table-row",
            )

            yield Static("Sentiment", classes="section-header")
            yield Static(
                f"  {'Overall':<12} {result.sentiment.overall_sentiment}",
                classes="table-row",
            )
            yield Static(
                f"  {'Score':<12} {result.sentiment.sentiment_score:.2f}",
                classes="table-row",
            )
            yield Static(
                f"  {'Articles':<12} {result.sentiment.article_count}",
                classes="table-row",
            )

            if result.news.key_themes:
                yield Static("News Themes", classes="section-header")
                themes = ", ".join(result.news.key_themes[:3])
                yield Static(f"  {themes}", classes="table-row")

            if result.warnings:
                yield Static("⚠️ Incomplete Data", classes="warning-header")
                for warning in result.warnings:
                    yield Static(f"  • {warning}", classes="warning")

            yield Static(f"Reasoning: {result.decision.reasoning}", classes="reasoning")

    def __repr__(self) -> str:
        """Return string representation."""
        return f"ResultBox(symbol={self._result.symbol})"
