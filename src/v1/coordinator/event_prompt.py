"""Prompt builder for event-driven coordinator cycles."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime

from src.prompts import PromptLoader
from src.strategies.session import TradingSession
from src.v1.coordinator.models import CoordinatorConfig
from src.v1.event_queue.models import QueuedMarketEvent

_EVENT_TYPE_TEMPLATES = frozenset(
    {
        "news",
        "news_watchlist",
        "social",
        "filing",
        "trump",
        "anomaly",
        "news_trending",
        "signal",
        "position_review",
        "risk_report",
        "watchlist_stale",
    }
)


@dataclass
class EventCycleContext:
    """Runtime context for building an event cycle prompt."""

    positions_summary: str
    session: TradingSession
    market_open: bool
    game_plan: str = field(default="")


class EventCyclePromptBuilder:
    """Builds prompts for event-driven coordinator cycles."""

    def __init__(self) -> None:
        """Initialize with prompt loader for coordinator/events/."""
        self._prompts = PromptLoader("coordinator/events")

    def build(
        self,
        events: Sequence[QueuedMarketEvent],
        context: EventCycleContext,
        config: CoordinatorConfig,
    ) -> str:
        """Build the full event cycle prompt.

        Args:
            events: Dequeued market events
            context: Runtime context (positions, session, market_open, game_plan)
            config: Coordinator config for risk limits

        Returns:
            Rendered prompt string
        """
        symbols = extract_symbols(events)
        risk_limits = (
            f"**Risk Limits:** max position {config.max_position_pct}% | "
            f"max daily trades {config.max_daily_trades} | "
            f"min confidence {config.min_confidence_to_trade:.0%}"
        )

        game_plan_section = f"**Today's Game Plan:**\n{context.game_plan}" if context.game_plan else ""

        header = self._prompts.load(
            "event_header",
            date=datetime.now(UTC).strftime("%Y-%m-%d"),
            session=context.session.value,
            market_open="Yes" if context.market_open else "No — analysis only, skip trade execution",
            symbols=", ".join(sorted(symbols)) if symbols else "None extracted",
            positions_summary=context.positions_summary,
            risk_limits=risk_limits,
            game_plan_section=game_plan_section,
            event_count=len(events),
        )

        event_sections = [self._format_single_event(ev, game_plan=context.game_plan) for ev in events]
        return header + "\n".join(event_sections)

    def _format_single_event(self, event: QueuedMarketEvent, game_plan: str = "") -> str:
        """Render one event using its type-specific template.

        Args:
            event: Single queued event with payload containing event + triage data
            game_plan: Today's game plan (passed as extra kwarg; only signal.txt uses it)

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
            game_plan=game_plan,
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
    _append_position_review_fields(event_data, lines)
    _append_risk_report_fields(event_data, lines)
    _append_watchlist_stale_fields(event_data, lines)
    return "\n".join(lines)


def _append_position_review_fields(event_data: dict, lines: list[str]) -> None:
    """Append position_review-specific fields to lines list."""
    positions = event_data.get("positions")
    if not positions:
        return

    portfolio_value = event_data.get("portfolio_value", 0)
    total_exposure = event_data.get("total_exposure", 0)
    lines.append(f"Portfolio: ${portfolio_value:,.0f} | Exposure: ${total_exposure:,.0f}")
    lines.append(f"Positions ({len(positions)}):")

    for pos in positions:
        symbol = pos.get("symbol", "?")
        qty = pos.get("qty", 0)
        entry = pos.get("avg_entry_price", 0)
        current = pos.get("current_price", 0)
        pnl_pct = pos.get("unrealized_pnl_percent", 0)
        days = pos.get("days_held")
        confidence = pos.get("entry_confidence")
        stop = pos.get("stop_loss_price")
        flags = pos.get("flags", [])

        parts = [f"  {symbol}: {qty} @ ${entry:.2f} → ${current:.2f} ({pnl_pct:+.1f}%)"]
        if days is not None:
            parts.append(f"held={days}d")
        if confidence is not None:
            parts.append(f"conf={confidence:.0%}")
        if stop is not None:
            parts.append(f"stop=${stop:.2f}")
        if flags:
            parts.append(f"[{', '.join(flags)}]")

        lines.append(" ".join(parts))


def _append_risk_report_fields(event_data: dict, lines: list[str]) -> None:
    """Append risk_report-specific fields to lines list."""
    if event_data.get("event_type") != "risk_report":
        return
    if risk_status := event_data.get("risk_status"):
        lines.append(f"Risk Status: {risk_status}")
    var_95 = event_data.get("var_95")
    cvar_99 = event_data.get("cvar_99")
    cdar_95 = event_data.get("cdar_95")
    if var_95 is not None and cvar_99 is not None and cdar_95 is not None:
        lines.append(f"VaR95={var_95:.2%}, CVaR99={cvar_99:.2%}, CDaR95={cdar_95:.2%}")
    max_dd = event_data.get("max_drawdown")
    vol = event_data.get("portfolio_volatility")
    if max_dd is not None and vol is not None:
        lines.append(f"Max Drawdown={max_dd:.2%}, Volatility={vol:.2%}")
    exposure = event_data.get("current_exposure_percent")
    num_pos = event_data.get("num_positions")
    if exposure is not None and num_pos is not None:
        lines.append(f"Exposure={exposure:.1f}%, Positions={num_pos}")


def _append_watchlist_stale_fields(event_data: dict, lines: list[str]) -> None:
    """Append watchlist_stale-specific fields to lines list."""
    stale = event_data.get("stale_symbols", [])
    if not stale:
        return
    lines.append(f"Stale symbols ({len(stale)}):")
    for s in stale:
        lines.append(f"  {s['symbol']}: {s['last_analysis_age_hours']:.1f}h ago")


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
