"""Prompt builder for event-driven coordinator cycles."""

from collections.abc import Sequence
from datetime import UTC, datetime

from src.coordinator.models import CoordinatorConfig
from src.event_queue.models import QueuedMarketEvent
from src.prompts import PromptLoader
from src.strategies.session import TradingSession

_EVENT_TYPE_TEMPLATES = frozenset({"news", "social", "filing", "trump", "anomaly", "news_trending", "signal"})


class EventCyclePromptBuilder:
    """Builds prompts for event-driven coordinator cycles."""

    def __init__(self) -> None:
        """Initialize with prompt loader for coordinator/events/."""
        self._prompts = PromptLoader("coordinator/events")

    def build(
        self,
        events: Sequence[QueuedMarketEvent],
        positions_summary: str,
        session: TradingSession,
        config: CoordinatorConfig,
        market_open: bool,
    ) -> str:
        """Build the full event cycle prompt.

        Args:
            events: Dequeued market events
            positions_summary: Formatted positions string
            session: Current trading session
            config: Coordinator config for risk limits
            market_open: Whether market is currently open

        Returns:
            Rendered prompt string
        """
        symbols = extract_symbols(events)
        risk_limits = (
            f"**Risk Limits:** max position {config.max_position_pct}% | "
            f"max daily trades {config.max_daily_trades} | "
            f"min confidence {config.min_confidence_to_trade:.0%}"
        )

        header = self._prompts.load(
            "event_header",
            date=datetime.now(UTC).strftime("%Y-%m-%d"),
            session=session.value,
            market_open="Yes" if market_open else "No — analysis only, skip trade execution",
            symbols=", ".join(sorted(symbols)) if symbols else "None extracted",
            positions_summary=positions_summary,
            risk_limits=risk_limits,
            event_count=len(events),
        )

        event_sections = [self._format_single_event(ev) for ev in events]
        return header + "\n".join(event_sections)

    def _format_single_event(self, event: QueuedMarketEvent) -> str:
        """Render one event using its type-specific template.

        Args:
            event: Single queued event with payload containing event + triage data

        Returns:
            Rendered event section
        """
        triage = event.payload.get("triage", {})
        event_data = event.payload.get("event", {})

        event_details = _format_event_details(event_data)

        template_name = event.event_type if event.event_type in _EVENT_TYPE_TEMPLATES else "news"
        return self._prompts.load(
            template_name,
            event_details=event_details,
            urgency=triage.get("urgency", "UNKNOWN"),
            sentiment=triage.get("sentiment", "NEUTRAL"),
            confidence=triage.get("confidence", 0.0),
            reasoning=triage.get("reasoning", "No reasoning provided"),
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return "EventCyclePromptBuilder()"


def extract_symbols(events: Sequence[QueuedMarketEvent]) -> set[str]:
    """Extract unique symbols from event triage payloads.

    Args:
        events: List of queued events

    Returns:
        Set of symbol strings
    """
    symbols: set[str] = set()
    for event in events:
        triage = event.payload.get("triage", {})
        symbols.update(triage.get("symbols", []))
        event_data = event.payload.get("event", {})
        if symbol := event_data.get("symbol"):
            symbols.add(symbol)
    return symbols


def _format_event_details(event_data: dict) -> str:
    """Format raw event data into readable text.

    Args:
        event_data: Raw event dict from payload

    Returns:
        Human-readable event summary
    """
    lines = []
    event_type = event_data.get("event_type", "unknown")
    lines.append(f"Type: {event_type}")

    if source := event_data.get("source"):
        lines.append(f"Source: {source}")

    if symbol := event_data.get("symbol"):
        lines.append(f"Symbol: {symbol}")

    if article := event_data.get("article"):
        lines.append(f"Title: {article.get('title', 'N/A')}")
        lines.append(f"Description: {article.get('description', 'N/A')}")

    if post := event_data.get("post"):
        lines.append(f"Content: {str(post.get('content', 'N/A'))[:500]}")

    if mention_count := event_data.get("mention_count"):
        lines.append(f"Mentions: {mention_count}")

    if spike_ratio := event_data.get("spike_ratio"):
        lines.append(f"Spike ratio: {spike_ratio:.1f}x")

    if anomaly_types := event_data.get("anomaly_types"):
        lines.append(f"Anomaly types: {', '.join(anomaly_types)}")

    _append_signal_fields(event_data, lines)
    return "\n".join(lines)


def _append_signal_fields(event_data: dict, lines: list[str]) -> None:
    """Append signal-specific fields to lines list."""
    if signal_action := event_data.get("signal"):
        lines.append(f"Signal: {signal_action}")
    if confidence := event_data.get("confidence"):
        lines.append(f"Confidence: {confidence:.0%}")
    if session := event_data.get("session"):
        lines.append(f"Session: {session}")
    if reasoning := event_data.get("reasoning"):
        lines.append(f"Reasoning: {reasoning}")
