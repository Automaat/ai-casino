"""Markdown formatters for TUI chat interface."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.screening.exporter import Watchlist
    from src.workflows.types import TradingWorkflowResult


def format_analysis_result(result: TradingWorkflowResult) -> str:
    """Format full analysis result as markdown.

    Args:
        result: Complete trading workflow result

    Returns:
        Formatted markdown string
    """
    signal = result.decision.action.value
    confidence = result.decision.confidence
    rsi_str = f"{result.technical.rsi:.2f}" if result.technical.rsi is not None else "N/A"

    return f"""## Analysis for {result.symbol}

**Decision: {signal}** (confidence: {confidence:.2f})
**Risk Level:** {result.decision.risk_level}

### Technical
- Signal: {result.technical.signal.value}
- RSI: {rsi_str}
- Confidence: {result.technical.confidence:.2f}

### Sentiment
{_format_sentiment_section(result)}

### News
{_format_news_section(result)}

### Reasoning
{result.decision.reasoning}"""


def format_technical(result: TradingWorkflowResult) -> str:
    """Format technical analysis as markdown.

    Args:
        result: Trading workflow result with technical analysis

    Returns:
        Formatted markdown string
    """
    rsi_str = f"{result.technical.rsi:.2f}" if result.technical.rsi is not None else "N/A"
    macd_str = f"{result.technical.macd_hist:.4f}" if result.technical.macd_hist is not None else "N/A"
    return f"""## Technical Analysis for {result.symbol}

- **Signal:** {result.technical.signal.value}
- **RSI:** {rsi_str}
- **MACD Histogram:** {macd_str}
- **Confidence:** {result.technical.confidence:.2f}

**Interpretation:**
{result.technical.interpretation}"""


def _format_sentiment_section(result: TradingWorkflowResult) -> str:
    """Format inline sentiment section."""
    if not result.sentiment:
        return "*Unavailable*"
    return (
        f"- Overall: {result.sentiment.overall_sentiment}\n"
        f"- Score: {result.sentiment.sentiment_score:.2f}\n"
        f"- Articles analyzed: {result.sentiment.article_count}"
    )


def _format_news_section(result: TradingWorkflowResult) -> str:
    """Format inline news section."""
    if not result.news:
        return "*Unavailable*"
    return (
        f"- Key themes: {', '.join(result.news.key_themes[:3])}\n"
        f"- Impact: {result.news.impact_assessment[:100]}"
    )


def format_sentiment(result: TradingWorkflowResult) -> str:
    """Format sentiment analysis as markdown.

    Args:
        result: Trading workflow result with sentiment analysis

    Returns:
        Formatted markdown string
    """
    if not result.sentiment:
        return f"## Sentiment Analysis for {result.symbol}\n\n*Unavailable*"
    return f"""## Sentiment Analysis for {result.symbol}

- **Overall:** {result.sentiment.overall_sentiment}
- **Score:** {result.sentiment.sentiment_score:.2f}
- **Articles:** {result.sentiment.article_count}
- **Positive:** {result.sentiment.positive_ratio * 100:.1f}%
- **Negative:** {result.sentiment.negative_ratio * 100:.1f}%"""


def format_news(result: TradingWorkflowResult) -> str:
    """Format news analysis as markdown.

    Args:
        result: Trading workflow result with news analysis

    Returns:
        Formatted markdown string
    """
    if not result.news:
        return f"## News Analysis for {result.symbol}\n\n*Unavailable*"
    themes = ", ".join(result.news.key_themes[:5]) if result.news.key_themes else "None"
    return f"""## News Analysis for {result.symbol}

**Key Themes:** {themes}

**Impact Assessment:**
{result.news.impact_assessment}

**Recommendation:**
{result.news.recommendation}"""


def format_candidates(_record: dict) -> str:
    """Format screening candidates for display (deprecated).

    Args:
        _record: dict with screening data (unused)

    Returns:
        Formatted markdown string
    """
    # Legacy function - screening removed from system
    return "## Screening Candidates\n\n*Screening functionality has been removed. Use discovery instead.*"


def format_watchlist(watchlist: Watchlist) -> str:
    """Format watchlist for display.

    Args:
        watchlist: Watchlist instance

    Returns:
        Formatted markdown string
    """
    lines = [
        f"## Watchlist: {watchlist.name}",
        f"*Updated: {watchlist.updated_at.strftime('%Y-%m-%d %H:%M')}*",
        "",
    ]

    if not watchlist.entries:
        lines.append("*No entries*")
    else:
        for entry in watchlist.entries:
            notes_str = f" - {entry.notes}" if entry.notes else ""
            lines.append(
                f"- **{entry.symbol}** ({entry.name}) | "
                f"Score: {entry.score:.2f} | {entry.criteria.value}{notes_str}"
            )

    return "\n".join(lines)
