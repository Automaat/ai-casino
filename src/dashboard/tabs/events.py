"""Events tab - event log with filters, timeline, warnings."""

from datetime import UTC, datetime, timedelta

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html
from loguru import logger

from src.dashboard.api_client import DaemonAPIClient

# Constants
_CONSECUTIVE_SIGNALS_THRESHOLD = 5
_HIGH_DRAWDOWN_THRESHOLD = 0.10
_STALENESS_THRESHOLD_MINUTES = 30
_DETAILS_MAX_LENGTH = 150


def render(client: DaemonAPIClient) -> list:
    """Render Events tab content.

    Args:
        client: API client

    Returns:
        Tab content components
    """
    try:
        # Fetch all data
        system_events = client.get_events(limit=100).events
        market_events_resp = client.get_market_events(limit=100)
        market_events = market_events_resp.events
        degradation_history = client.get_degradation_history(limit=50)

        # Merge events and sort by timestamp
        all_events = []

        for e in system_events:
            e["source"] = "system"
            all_events.append(e)

        for e in market_events:
            e["source"] = "market"
            all_events.append(e)

        all_events.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

        if not all_events:
            return [dbc.Alert("No events yet", color="info")]

        # Serialize events for dcc.Store
        events_data = [
            {
                "timestamp": e.get("timestamp"),
                "event_type": e.get("event_type") or (e.get("event", {}).get("event_type", "unknown")),
                "source": e.get("source"),
                "data": e.get("data", {}),
                "summary": e.get("summary", "-"),
            }
            for e in all_events
        ]

        # Generate warnings
        warnings = _generate_warnings(client)
        warnings_banner = _render_warnings_banner(warnings)

        # Build degradation timeline
        degradation_records = degradation_history.records
        timeline = _build_degradation_timeline(degradation_records)

        # Build filter controls
        unique_categories = sorted({_categorize_event(e) for e in all_events})

        filters = dbc.Card(
            dbc.CardBody(
                [
                    html.H5("Filters", className="mb-3"),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Label("Event Types"),
                                    dcc.Checklist(
                                        id="events-filter-type",
                                        options=[
                                            {"label": f" {cat}", "value": cat} for cat in unique_categories
                                        ],
                                        value=[
                                            c
                                            for c in unique_categories
                                            if c in ["ANALYSIS", "NEWS", "SOCIAL", "ANOMALY", "ERROR"]
                                        ],
                                        inline=True,
                                    ),
                                ],
                                width=8,
                            ),
                            dbc.Col(
                                [
                                    html.Label("Date Range"),
                                    dcc.DatePickerRange(
                                        id="events-filter-date",
                                        start_date=(datetime.now(UTC) - timedelta(days=7)).date(),
                                        end_date=datetime.now(UTC).date(),
                                        display_format="YYYY-MM-DD",
                                    ),
                                ],
                                width=4,
                            ),
                        ]
                    ),
                ]
            ),
            className="mb-4",
        )

        return [
            dcc.Store(id="events-data-store", data=events_data),
            html.H4("Events & Monitoring"),
            warnings_banner,
            timeline,
            filters,
            html.Div(id="events-filtered-content"),
        ]

    except Exception as e:
        logger.error(f"Events tab render failed: {e}")
        return [dbc.Alert(f"Failed to load events: {e!s}", color="danger")]


def register_callbacks(app: Dash) -> None:
    """Register Events tab filter callbacks.

    Args:
        app: Dash app instance
    """

    @app.callback(
        Output("events-filtered-content", "children"),
        [
            Input("events-filter-type", "value"),
            Input("events-filter-date", "start_date"),
            Input("events-filter-date", "end_date"),
        ],
        State("events-data-store", "data"),
    )
    def filter_events(event_types: list, start_date: str, end_date: str, events_data: list) -> list:
        """Filter events based on user selection.

        Args:
            event_types: Selected event type categories
            start_date: Start date string
            end_date: End date string
            events_data: Cached events from dcc.Store

        Returns:
            Filtered event table
        """
        if not events_data:
            return [dbc.Alert("No events available", color="info")]

        # Apply filters
        filtered = _apply_event_filters(events_data, event_types, start_date, end_date)

        if not filtered:
            return [dbc.Alert("No events match filters", color="info")]

        # Build event table
        table = _build_event_table(filtered)

        return [html.H5(f"Event Log ({len(filtered)} events)", className="mt-4 mb-3"), table]


def _apply_event_filters(events: list, event_types: list, start_date: str, end_date: str) -> list:
    """Filter events by criteria.

    Args:
        events: List of event dicts
        event_types: Selected event type categories
        start_date: Start date string
        end_date: End date string

    Returns:
        Filtered list of events
    """
    filtered = events

    # Event type filter
    if event_types:
        filtered = [e for e in filtered if _categorize_event(e) in event_types]

    # Date filter
    if start_date and end_date:
        start_dt = datetime.fromisoformat(start_date).replace(tzinfo=UTC)
        end_dt = datetime.fromisoformat(end_date).replace(hour=23, minute=59, second=59, tzinfo=UTC)
        filtered = [
            e
            for e in filtered
            if e.get("timestamp")
            and start_dt <= datetime.fromisoformat(e["timestamp"].replace("Z", "+00:00")) <= end_dt
        ]

    return filtered


def _categorize_event(event: dict) -> str:
    """Categorize event for filtering.

    Args:
        event: Event dict

    Returns:
        Category string
    """
    event_type = event.get("event_type", "").upper()

    # Market events
    if event_type in ["NEWS", "SOCIAL", "ANOMALY", "FILING"]:
        return event_type

    # System events
    if "ERROR" in event_type:
        return "ERROR"
    if event_type in ["CYCLE_START", "CYCLE_COMPLETE", "HEALTH_CHECK", "SCHEDULED_TASK", "STATE_UPDATE"]:
        return "SYSTEM"
    if event_type in ["ANALYSIS_START", "ANALYSIS_COMPLETE", "TRADE_EXECUTED"]:
        return "ANALYSIS"
    if event_type == "DEGRADATION":
        return "ERROR"

    return "SYSTEM"


def _event_severity(event: dict) -> tuple[str, str, str]:
    """Get (badge_color, icon, severity_label).

    Args:
        event: Event dict

    Returns:
        Tuple of (color, icon, severity_label)
    """
    category = _categorize_event(event)
    event_type = event.get("event_type", "").upper()

    if category == "ERROR" or "ERROR" in event_type:
        return "#ef4444", "🔴", "ERROR"
    if category in ["NEWS", "SOCIAL", "ANOMALY"]:
        return "#3b82f6", "🔵", "INFO"
    if category == "ANALYSIS" and "TRADE" in event_type:
        return "#22c55e", "🟢", "TRADE"
    return "#6b7280", "⚪", "SYSTEM"


def _build_degradation_timeline(records: list[dict]) -> dcc.Graph | dbc.Alert:
    """Build degradation timeline visualization.

    Args:
        records: Degradation history records

    Returns:
        Graph or alert if no data
    """
    if not records:
        return dbc.Alert("No degradation history", color="info", className="mb-3")

    tier_order = ["FULL", "DEGRADED", "MINIMAL", "HALTED"]
    tier_colors = {"FULL": "#22c55e", "DEGRADED": "#fbbf24", "MINIMAL": "#f97316", "HALTED": "#ef4444"}

    timestamps = [datetime.fromisoformat(r["timestamp"].replace("Z", "+00:00")) for r in records]
    tiers = [r["tier"] for r in records]
    services = [", ".join(r["unavailable_services"]) or "All healthy" for r in records]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=tiers,
            mode="lines+markers",
            marker={"color": [tier_colors.get(t, "#6b7280") for t in tiers], "size": 10},
            line={"color": "#6b7280", "width": 2},
            hovertemplate="<b>%{y}</b><br>%{x}<br>%{customdata}<extra></extra>",
            customdata=services,
        )
    )

    fig.update_layout(
        title="API Degradation Timeline",
        yaxis={"categoryorder": "array", "categoryarray": tier_order, "title": "Tier"},
        xaxis={"title": "Time"},
        height=300,
        showlegend=False,
    )

    return dcc.Graph(figure=fig)


def _check_degradation_warning(client: DaemonAPIClient) -> dict | None:
    """Check for API degradation warning."""
    try:
        degradation = client.get_degradation()
        if degradation.tier == "HALTED":
            return {
                "severity": "danger",
                "icon": "🔴",
                "message": f"HALTED: {degradation.halt_reason}",
                "details": f"Unavailable: {', '.join(degradation.unavailable_services)}",
            }
        if degradation.tier in ["DEGRADED", "MINIMAL"]:
            return {
                "severity": "warning",
                "icon": "🟡",
                "message": f"{degradation.tier} mode active",
                "details": f"Confidence adjustment: {degradation.confidence_adjustment:.0%}",
            }
    except Exception as e:
        logger.debug(f"Degradation check failed: {e}")
    return None


def _check_consecutive_losses(client: DaemonAPIClient) -> dict | None:
    """Check for consecutive non-BUY signals."""
    try:
        analyses_resp = client.get_analyses(limit=_CONSECUTIVE_SIGNALS_THRESHOLD)
        if not analyses_resp or not analyses_resp.analyses:
            return None

        recent_signals = [a.signal for a in analyses_resp.analyses[:_CONSECUTIVE_SIGNALS_THRESHOLD]]
        if all(s in ["SELL", "HOLD"] for s in recent_signals):
            return {
                "severity": "warning",
                "icon": "🟡",
                "message": f"Consecutive non-BUY signals: {len(recent_signals)}",
                "details": "Portfolio may be risk-averse or markets bearish",
            }
    except Exception as e:
        logger.debug(f"Consecutive losses check failed: {e}")
    return None


def _check_high_drawdown(client: DaemonAPIClient) -> dict | None:
    """Check for high portfolio drawdown."""
    try:
        risk = client.get_risk()
        if risk and hasattr(risk, "max_drawdown") and abs(risk.max_drawdown) > _HIGH_DRAWDOWN_THRESHOLD:
            return {
                "severity": "danger",
                "icon": "🔴",
                "message": f"High drawdown: {abs(risk.max_drawdown):.1%}",
                "details": f"Risk status: {risk.risk_status}",
            }
    except Exception as e:
        logger.debug(f"Drawdown check failed: {e}")
    return None


def _generate_warnings(client: DaemonAPIClient) -> list[dict]:
    """Generate warnings from multiple sources.

    Args:
        client: API client

    Returns:
        List of warning dicts with severity, icon, message, details
    """
    warnings = []

    for check_fn in [
        _check_degradation_warning,
        _check_consecutive_losses,
        _check_high_drawdown,
    ]:
        warning = check_fn(client)
        if warning:
            warnings.append(warning)

    return warnings


def _render_warnings_banner(warnings: list[dict]) -> html.Div:
    """Render warnings banner.

    Args:
        warnings: List of warning dicts

    Returns:
        Div containing warnings or empty div
    """
    if not warnings:
        return html.Div()

    alerts = [
        dbc.Alert(
            [
                html.Strong(f"{w['icon']} {w['message']}"),
                html.Br(),
                html.Small(w["details"]),
            ],
            color=w["severity"],
            className="mb-2",
        )
        for w in warnings
    ]

    return html.Div(
        [
            html.H5("⚠️ Active Warnings", className="mb-3"),
            html.Div(alerts),
            html.Hr(),
        ]
    )


def _build_event_table(all_events: list[dict]) -> dbc.Table:
    """Build event table from events list.

    Args:
        all_events: List of event dicts

    Returns:
        Bootstrap table component
    """
    table_rows = []
    for event in all_events[:100]:  # Limit display to 100
        timestamp = event.get("timestamp")
        if timestamp:
            try:
                ts_obj = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                timestamp_str = ts_obj.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                timestamp_str = str(timestamp)
        else:
            timestamp_str = "Unknown"

        event_type = event.get("event_type", "unknown")
        category = _categorize_event(event)
        color, icon, _severity = _event_severity(event)

        # Extract details
        if event.get("source") == "market":
            # Market event (EventSignal)
            summary = event.get("summary", "-")
            if len(summary) > _DETAILS_MAX_LENGTH:
                details_str = summary[:_DETAILS_MAX_LENGTH] + "..."
            else:
                details_str = summary
        else:
            # System event
            data = event.get("data", {})
            details_str = str(data) if data else "-"
            if len(details_str) > _DETAILS_MAX_LENGTH:
                details_str = details_str[: _DETAILS_MAX_LENGTH - 3] + "..."

        table_rows.append(
            html.Tr(
                [
                    html.Td(timestamp_str, className="font-monospace small"),
                    html.Td(
                        [
                            html.Span(icon, style={"margin-right": "5px"}),
                            html.Span(
                                event_type.replace("_", " "),
                                style={"color": color, "font-weight": "bold"},
                            ),
                        ]
                    ),
                    html.Td(category, className="small"),
                    html.Td(details_str, className="small"),
                ]
            )
        )

    return dbc.Table(
        [
            html.Thead(
                html.Tr(
                    [
                        html.Th("Timestamp"),
                        html.Th("Type"),
                        html.Th("Category"),
                        html.Th("Details"),
                    ]
                )
            ),
            html.Tbody(table_rows),
        ],
        bordered=True,
        hover=True,
        striped=True,
        size="sm",
    )
