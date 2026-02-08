"""Overview tab - health + config + game plan + charts."""

from collections import Counter
from datetime import UTC, datetime, timedelta

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import Dash, dcc, html

from src.daemon.api import (
    ConfigResponse,
    DegradationResponse,
    GamePlanResponse,
    HealthResponse,
    StateSummaryResponse,
    WatchlistResponse,
)
from src.dashboard.api_client import DaemonAPIClient


def render(client: DaemonAPIClient) -> list:
    """Render Overview tab content.

    Args:
        client: API client

    Returns:
        Tab content components
    """
    health = client.get_health()
    summary = client.get_state_summary()
    config = client.get_config()
    degradation = client.get_degradation()
    watchlist = client.get_watchlist()
    game_plan = client.get_game_plan()

    status_cards = _build_status_cards(health, summary, degradation)
    degradation_badge = _build_degradation_badge(degradation)
    service_health = _build_service_health_indicators(degradation)
    watchlist_section = _build_watchlist_section(watchlist)
    game_plan_section = _build_game_plan_section(game_plan) if game_plan else None
    config_table = _build_config_table(config)
    analyses_chart = _build_analyses_chart(client)

    components = [
        status_cards,
        degradation_badge,
        html.Hr(),
        service_health,
        html.Hr(),
        watchlist_section,
    ]

    if game_plan_section:
        components.extend([html.Hr(), game_plan_section])

    components.extend(
        [
            html.Hr(),
            html.H4("Configuration"),
            config_table,
            html.Hr(),
            html.H4("Analyses (Last 24 Hours)"),
            analyses_chart,
        ]
    )

    return components


def register_callbacks(app: Dash) -> None:
    """Register Overview tab callbacks (none needed).

    Args:
        app: Dash app instance
    """


def _build_status_cards(
    health: HealthResponse, summary: StateSummaryResponse, degradation: DegradationResponse
) -> dbc.Row:
    """Build 4 status cards row."""
    status_color = "success" if health.status == "healthy" else "warning"

    return dbc.Row(
        [
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardBody(
                            [
                                html.H5("Status", className="card-title"),
                                html.H3(health.status.upper(), className=f"text-{status_color}"),
                                html.P(f"Running: {health.running}", className="card-text"),
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
                                html.H3(f"{health.uptime_seconds:.0f}s"),
                                html.P(
                                    f"Last run: {health.last_run or 'Never'}",
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
                                html.H3(summary.total_analyses),
                                html.P(f"Trades: {summary.total_trades}", className="card-text"),
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
                                html.H3(summary.error_count, className="text-danger"),
                                html.P(f"Tier: {degradation.tier}", className="card-text"),
                            ]
                        )
                    ]
                ),
                width=3,
            ),
        ],
        className="mb-4",
    )


def _build_degradation_badge(degradation: DegradationResponse) -> html.Div:
    """Build degradation tier badge."""
    tier_colors = {"FULL": "success", "DEGRADED": "warning", "MINIMAL": "danger", "HALTED": "dark"}
    color = tier_colors.get(degradation.tier, "secondary")

    if degradation.tier == "FULL":
        description = "All systems operational"
    elif degradation.tier == "HALTED":
        description = f"Trading halted: {degradation.halt_reason}"
    else:
        unavailable = ", ".join(degradation.unavailable_services)
        adjustment = degradation.confidence_adjustment
        description = f"Unavailable: {unavailable} | Confidence: {adjustment:.0%}"

    return html.Div(
        [
            html.H5("System Health", className="mb-2"),
            html.Span(
                f"{degradation.tier} CAPACITY",
                className=f"badge bg-{color} me-2",
                style={"fontSize": "1.2rem"},
            ),
            html.Span(description, className="text-muted"),
        ],
        className="mb-3",
    )


def _build_service_health_indicators(degradation: DegradationResponse) -> html.Div:
    """Build service health indicator badges."""
    services = [
        ("alpha_vantage", "Alpha Vantage"),
        ("marketaux", "Marketaux"),
        ("alpaca", "Alpaca"),
        ("llm", "LLM"),
        ("finnhub", "Finnhub"),
    ]

    unavailable = set(degradation.unavailable_services)
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


def _build_watchlist_section(watchlist: WatchlistResponse) -> html.Div:
    """Build watchlist display section."""
    sources = watchlist.sources
    breakdown = f"Config: {sources['config']}, Broker: {sources['broker']}, Screening: {sources['screening']}"

    symbol_badges = [html.Span(symbol, className="badge bg-primary me-2") for symbol in watchlist.symbols]

    return html.Div(
        [
            html.H5(f"Watchlist ({watchlist.count} symbols)", className="mb-2"),
            html.P(breakdown, className="text-muted small mb-2"),
            html.Div(symbol_badges),
        ],
        className="mb-3",
    )


def _build_game_plan_section(game_plan: GamePlanResponse) -> html.Div:
    """Build game plan summary section."""
    stance_colors = {"AGGRESSIVE": "danger", "NEUTRAL": "secondary", "DEFENSIVE": "success"}
    stance_color = stance_colors.get(game_plan.risk_stance, "secondary")

    priority_badges = [
        html.Span(symbol, className="badge bg-info me-2") for symbol in game_plan.priority_symbols
    ]
    sector_badges = [
        html.Span(sector, className="badge bg-warning text-dark me-2") for sector in game_plan.sector_focus
    ]

    return html.Div(
        [
            html.H5(
                [
                    "Game Plan",
                    html.Span(f" ({game_plan.date})", className="text-muted small"),
                ],
                className="mb-2",
            ),
            html.Div(
                [
                    html.Span("Risk Stance: ", className="fw-bold"),
                    html.Span(game_plan.risk_stance, className=f"badge bg-{stance_color} me-3"),
                    html.Span(f"Confidence: {game_plan.confidence:.0%}", className="text-muted small"),
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
                [html.Span("Reasoning: ", className="fw-bold"), game_plan.reasoning], className="text-muted"
            ),
        ],
        className="mb-3",
    )


def _build_config_table(config: ConfigResponse) -> dbc.Table:
    """Build configuration table."""
    return dbc.Table(
        [
            html.Thead(html.Tr([html.Th("Setting"), html.Th("Value")])),
            html.Tbody(
                [
                    html.Tr([html.Td("Watchlist"), html.Td(", ".join(config.watchlist))]),
                    html.Tr([html.Td("Interval"), html.Td(f"{config.interval_minutes} minutes")]),
                    html.Tr([html.Td("Market Hours Only"), html.Td(str(config.market_hours_only))]),
                    html.Tr([html.Td("Auto Trade"), html.Td(str(config.auto_trade))]),
                    html.Tr([html.Td("Trading Mode"), html.Td(config.trading_mode.upper())]),
                    html.Tr([html.Td("Pre-Market"), html.Td(str(config.pre_market_enabled))]),
                ]
            ),
        ],
        bordered=True,
        hover=True,
        striped=True,
    )


def _build_analyses_chart(client: DaemonAPIClient) -> dbc.Alert | dcc.Graph:
    """Build analyses per hour bar chart (last 24h)."""
    analyses_resp = client.get_analyses(limit=500)

    if analyses_resp.returned_count == 0:
        return dbc.Alert("No analyses in last 24 hours", color="info")

    now = datetime.now(UTC)
    cutoff = now - timedelta(hours=24)

    recent_analyses = []
    for analysis in analyses_resp.analyses:
        ts = analysis.timestamp if analysis.timestamp.tzinfo else analysis.timestamp.replace(tzinfo=UTC)
        if ts >= cutoff:
            recent_analyses.append(analysis)

    if not recent_analyses:
        return dbc.Alert("No analyses in last 24 hours", color="info")

    hour_buckets = Counter()
    for analysis in recent_analyses:
        ts = analysis.timestamp if analysis.timestamp.tzinfo else analysis.timestamp.replace(tzinfo=UTC)
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
