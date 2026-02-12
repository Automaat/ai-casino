"""Overview tab - health + config + game plan + charts."""

from collections import Counter
from datetime import UTC, datetime, timedelta
from typing import Any

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate
from loguru import logger

from src.dashboard.api_client import DaemonAPIClient


def render(client: DaemonAPIClient) -> list:  # noqa: ARG001
    """Render Overview tab static structure with dcc.Store.

    Args:
        client: API client

    Returns:
        Tab content components
    """
    # Initial empty render - data loaded by interval callback
    return [
        dcc.Store(id="overview-data-store", data=None),
        html.Div(id="overview-timestamp", className="text-muted small mb-3", children="Last updated: -"),
        html.Div(id="overview-status-cards"),
        html.Div(id="overview-degradation-badge"),
        html.Hr(),
        html.Div(id="overview-service-health"),
        html.Hr(),
        html.Div(id="overview-watchlist"),
        html.Div(id="overview-game-plan"),
        html.Hr(),
        html.H4("Analyses (Last 24 Hours)"),
        html.Div(id="overview-analyses-chart"),
    ]


def register_callbacks(app: Dash) -> None:  # noqa: C901, PLR0915
    """Register Overview tab callbacks.

    Args:
        app: Dash app instance
    """
    client = app.api_client  # type: ignore[attr-defined]

    @app.callback(
        Output("overview-data-store", "data"),
        Input("interval-component", "n_intervals"),
        Input("tabs", "active_tab"),
        State("overview-data-store", "data"),
    )
    def update_overview_data(n_intervals: int, active_tab: str, current_data: dict | None) -> dict:  # noqa: ARG001
        """Update Overview store when tab active.

        Args:
            n_intervals: Interval counter
            active_tab: Active tab ID
            current_data: Current store data

        Returns:
            Serialized overview data
        """
        from dash import callback_context

        # Only check active tab if triggered by interval, not by tab switch
        if callback_context.triggered_id == "interval-component" and active_tab != "overview":
            raise PreventUpdate

        try:
            health = client.get_health()
            summary = client.get_state_summary()
            degradation = client.get_degradation()
            watchlist = client.get_watchlist()
            game_plan = client.get_game_plan()
            analyses = client.get_analyses(limit=500)

            return {
                "timestamp": datetime.now(UTC).isoformat(),
                "health": {
                    "status": health.status,
                    "running": health.running,
                    "uptime_seconds": health.uptime_seconds,
                    "last_run": health.last_run,
                },
                "summary": {
                    "total_analyses": summary.total_analyses,
                    "total_trades": summary.total_trades,
                    "error_count": summary.error_count,
                },
                "degradation": {
                    "tier": degradation.tier,
                    "unavailable_services": degradation.unavailable_services,
                    "confidence_adjustment": degradation.confidence_adjustment,
                    "halt_reason": degradation.halt_reason,
                },
                "watchlist": {
                    "symbols": watchlist.symbols,
                    "count": watchlist.count,
                    "sources": watchlist.sources,
                },
                "game_plan": {
                    "date": game_plan.date,
                    "risk_stance": game_plan.risk_stance,
                    "confidence": game_plan.confidence,
                    "priority_symbols": game_plan.priority_symbols,
                    "sector_focus": game_plan.sector_focus,
                    "reasoning": game_plan.reasoning,
                }
                if game_plan
                else None,
                "analyses": [
                    {
                        "timestamp": a.timestamp.isoformat(),
                    }
                    for a in analyses.analyses
                ],
            }
        except Exception as e:
            logger.opt(exception=True).error(f"Overview refresh failed: {e}")
            return current_data or {}

    @app.callback(
        Output("overview-timestamp", "children"),
        Input("overview-data-store", "data"),
    )
    def update_timestamp(data: dict | None) -> str:
        """Update timestamp display.

        Args:
            data: Store data

        Returns:
            Timestamp text
        """
        if not data or "timestamp" not in data:
            return "Last updated: -"
        ts = datetime.fromisoformat(data["timestamp"])
        return f"Last updated: {ts.strftime('%Y-%m-%d %H:%M:%S')}"

    @app.callback(
        Output("overview-status-cards", "children"),
        Input("overview-data-store", "data"),
    )
    def update_status_cards(data: dict | None) -> dbc.Row | str:
        """Update status cards.

        Args:
            data: Store data

        Returns:
            Status cards row
        """
        if not data:
            return ""

        health = data.get("health", {})
        summary = data.get("summary", {})
        degradation = data.get("degradation", {})

        status_color = "success" if health.get("status") == "healthy" else "warning"

        return dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardBody(
                                [
                                    html.H5("Status", className="card-title"),
                                    html.H3(
                                        health.get("status", "unknown").upper(),
                                        className=f"text-{status_color}",
                                    ),
                                    html.P(f"Running: {health.get('running', False)}", className="card-text"),
                                ]
                            )
                        ]
                    ),
                    width=3,
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardBody(
                                [
                                    html.H5("Uptime", className="card-title"),
                                    html.H3(f"{health.get('uptime_seconds', 0):.0f}s"),
                                    html.P(
                                        f"Last run: {health.get('last_run') or 'Never'}",
                                        className="card-text text-muted small",
                                    ),
                                ]
                            )
                        ]
                    ),
                    width=3,
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardBody(
                                [
                                    html.H5("Analyses", className="card-title"),
                                    html.H3(summary.get("total_analyses", 0)),
                                    html.P(
                                        f"Trades: {summary.get('total_trades', 0)}", className="card-text"
                                    ),
                                ]
                            )
                        ]
                    ),
                    width=3,
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardBody(
                                [
                                    html.H5("Errors", className="card-title"),
                                    html.H3(summary.get("error_count", 0), className="text-danger"),
                                    html.P(
                                        f"Tier: {degradation.get('tier', 'UNKNOWN')}", className="card-text"
                                    ),
                                ]
                            )
                        ]
                    ),
                    width=3,
                ),
            ],
            className="mb-4",
        )

    @app.callback(
        Output("overview-degradation-badge", "children"),
        Input("overview-data-store", "data"),
    )
    def update_degradation_badge(data: dict | None) -> html.Div | str:
        """Update degradation badge.

        Args:
            data: Store data

        Returns:
            Degradation badge div
        """
        if not data:
            return ""

        degradation = data.get("degradation", {})
        tier = degradation.get("tier", "UNKNOWN")
        tier_colors = {"FULL": "success", "DEGRADED": "warning", "MINIMAL": "danger", "HALTED": "dark"}
        color = tier_colors.get(tier, "secondary")

        if tier == "FULL":
            description = "All systems operational"
        elif tier == "HALTED":
            description = f"Trading halted: {degradation.get('halt_reason', 'Unknown')}"
        else:
            unavailable = ", ".join(degradation.get("unavailable_services", []))
            adjustment = degradation.get("confidence_adjustment", 0.0)
            description = f"Unavailable: {unavailable} | Confidence: {adjustment:.0%}"

        return html.Div(
            [
                html.H5("System Health", className="mb-2"),
                html.Span(
                    f"{tier} CAPACITY",
                    className=f"badge bg-{color} me-2",
                    style={"fontSize": "1.2rem"},
                ),
                html.Span(description, className="text-muted"),
            ],
            className="mb-3",
        )

    @app.callback(
        Output("overview-service-health", "children"),
        Input("overview-data-store", "data"),
    )
    def update_service_health(data: dict | None) -> html.Div | str:
        """Update service health indicators.

        Args:
            data: Store data

        Returns:
            Service health div
        """
        if not data:
            return ""

        degradation = data.get("degradation", {})
        services = [
            ("alpha_vantage", "Alpha Vantage"),
            ("marketaux", "Marketaux"),
            ("alpaca", "Alpaca"),
            ("llm", "LLM"),
            ("finnhub", "Finnhub"),
        ]

        unavailable = set(degradation.get("unavailable_services", []))
        badges = []

        for service_id, service_name in services:
            is_available = service_id not in unavailable
            color = "success" if is_available else "danger"
            icon = "✓" if is_available else "✗"
            badges.append(
                html.Span(
                    f"{icon} {service_name}", className=f"badge bg-{color} me-2", style={"fontSize": "0.9rem"}
                )
            )

        return html.Div(
            [
                html.H5("Service Health", className="mb-2"),
                html.Div(badges),
            ],
            className="mb-3",
        )

    @app.callback(
        Output("overview-watchlist", "children"),
        Input("overview-data-store", "data"),
    )
    def update_watchlist(data: dict | None) -> html.Div | str:
        """Update watchlist section.

        Args:
            data: Store data

        Returns:
            Watchlist div
        """
        if not data:
            return ""

        watchlist = data.get("watchlist", {})
        sources = watchlist.get("sources", {})
        breakdown = (
            f"Config: {sources.get('config', 0)}, "
            f"Broker: {sources.get('broker', 0)}, "
            f"Screening: {sources.get('screening', 0)}"
        )

        symbol_badges = [
            html.Span(symbol, className="badge bg-primary me-2") for symbol in watchlist.get("symbols", [])
        ]

        return html.Div(
            [
                html.H5(f"Watchlist ({watchlist.get('count', 0)} symbols)", className="mb-2"),
                html.P(breakdown, className="text-muted small mb-2"),
                html.Div(symbol_badges),
            ],
            className="mb-3",
        )

    @app.callback(
        Output("overview-game-plan", "children"),
        Input("overview-data-store", "data"),
    )
    def update_game_plan(data: dict | None) -> list[Any]:
        """Update game plan section.

        Args:
            data: Store data

        Returns:
            Game plan components or empty list
        """
        if not data or not data.get("game_plan"):
            return []

        game_plan = data["game_plan"]
        stance_colors = {"AGGRESSIVE": "danger", "NEUTRAL": "secondary", "DEFENSIVE": "success"}
        stance_color = stance_colors.get(game_plan.get("risk_stance", ""), "secondary")

        priority_badges = [
            html.Span(symbol, className="badge bg-info me-2")
            for symbol in game_plan.get("priority_symbols", [])
        ]
        sector_badges = [
            html.Span(sector, className="badge bg-warning text-dark me-2")
            for sector in game_plan.get("sector_focus", [])
        ]

        return [
            html.Hr(),
            html.Div(
                [
                    html.H5(
                        [
                            "Game Plan",
                            html.Span(f" ({game_plan.get('date', '')})", className="text-muted small"),
                        ],
                        className="mb-2",
                    ),
                    html.Div(
                        [
                            html.Span("Risk Stance: ", className="fw-bold"),
                            html.Span(
                                game_plan.get("risk_stance", "UNKNOWN"),
                                className=f"badge bg-{stance_color} me-3",
                            ),
                            html.Span(
                                f"Confidence: {game_plan.get('confidence', 0):.0%}",
                                className="text-muted small",
                            ),
                        ],
                        className="mb-2",
                    ),
                    html.Div(
                        [
                            html.Span("Priority: ", className="fw-bold"),
                            html.Span(priority_badges),
                        ],
                        className="mb-2",
                    ),
                    html.Div(
                        [
                            html.Span("Sectors: ", className="fw-bold"),
                            html.Span(sector_badges),
                        ],
                        className="mb-2",
                    ),
                    html.P(
                        [html.Span("Reasoning: ", className="fw-bold"), game_plan.get("reasoning", "")],
                        className="text-muted",
                    ),
                ],
                className="mb-3",
            ),
        ]

    @app.callback(
        Output("overview-analyses-chart", "children"),
        Input("overview-data-store", "data"),
    )
    def update_analyses_chart(data: dict | None) -> dbc.Alert | dcc.Graph:
        """Update analyses chart.

        Args:
            data: Store data

        Returns:
            Analyses chart or alert
        """
        if not data or not data.get("analyses"):
            return dbc.Alert("No analyses in last 24 hours", color="info")

        analyses = data["analyses"]
        now = datetime.now(UTC)
        cutoff = now - timedelta(hours=24)

        recent_analyses = []
        for analysis in analyses:
            ts = datetime.fromisoformat(analysis["timestamp"])
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
            if ts >= cutoff:
                recent_analyses.append(ts)

        if not recent_analyses:
            return dbc.Alert("No analyses in last 24 hours", color="info")

        hour_buckets = Counter()
        for ts in recent_analyses:
            hour_key = ts.replace(minute=0, second=0, microsecond=0)
            hour_buckets[hour_key] += 1

        all_hours = []
        end_hour = now.replace(minute=0, second=0, microsecond=0)
        start_hour = end_hour - timedelta(hours=23)
        current = start_hour
        while current <= end_hour:
            all_hours.append(current)
            current += timedelta(hours=1)

        x_values = [h.strftime("%H:%M") for h in all_hours]
        y_values = [hour_buckets.get(h, 0) for h in all_hours]

        fig = go.Figure(data=[go.Bar(x=x_values, y=y_values, marker_color="rgb(55, 83, 109)")])
        fig.update_layout(
            title="Analyses per Hour (Last 24h)",
            xaxis_title="Hour",
            yaxis_title="Count",
            height=400,
            margin={"l": 40, "r": 40, "t": 40, "b": 40},
        )

        return dcc.Graph(figure=fig)
