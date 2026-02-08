"""Dash app factory with layout and callbacks."""

from collections import Counter
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html
from loguru import logger

from src.daemon.api import (
    ConfigResponse,
    DegradationResponse,
    GamePlanResponse,
    HealthResponse,
    StateSummaryResponse,
    WatchlistResponse,
)
from src.dashboard.api_client import DaemonAPIClient
from src.dashboard.config import DashboardConfig


def create_dash_app(config: DashboardConfig) -> Dash:
    """Create Dash app with layout and callbacks.

    Args:
        config: Dashboard configuration

    Returns:
        Dash app instance
    """
    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    app.title = "AI Casino Dashboard"

    # Store API client in app state
    app.api_client = DaemonAPIClient(config.api_url)

    # Layout
    app.layout = dbc.Container(
        [
            dcc.Interval(id="interval-component", interval=config.refresh_interval, n_intervals=0),
            html.H1("AI Casino Daemon Monitor", className="mt-4 mb-4"),
            dbc.Tabs(
                id="tabs",
                active_tab="overview",
                children=[
                    dbc.Tab(label="Overview", tab_id="overview"),
                    dbc.Tab(label="Portfolio", tab_id="portfolio"),
                    dbc.Tab(label="Signals", tab_id="signals"),
                    dbc.Tab(label="Risk", tab_id="risk"),
                    dbc.Tab(label="Events", tab_id="events"),
                ],
            ),
            html.Div(id="tab-content", className="mt-4"),
        ],
        fluid=True,
    )

    @app.callback(
        Output("tab-content", "children"),
        [Input("tabs", "active_tab"), Input("interval-component", "n_intervals")],
        [State("tab-content", "children")],
    )
    def render_tab_content(active_tab: str, n_intervals: int, current_content: list) -> list:  # noqa: ARG001, PLR0911
        """Render tab content (triggered by tab switch OR interval refresh).

        Args:
            active_tab: Active tab ID
            n_intervals: Interval counter
            current_content: Current tab content (unused)

        Returns:
            Tab content
        """
        try:
            if active_tab == "overview":
                return _render_overview_tab(app.api_client)
            if active_tab == "portfolio":
                return _render_portfolio_tab(app.api_client)
            if active_tab == "signals":
                return _render_signals_tab(app.api_client)
            if active_tab == "risk":
                return _render_risk_tab(app.api_client)
            if active_tab == "events":
                return _render_events_tab(app.api_client)
            return [html.Div("Invalid tab")]
        except Exception as e:
            logger.exception("Tab render failed")
            return [
                dbc.Alert(
                    [
                        html.H4("Error Loading Data", className="alert-heading"),
                        html.P(f"Failed to fetch data from daemon API: {e}"),
                        html.Hr(),
                        html.P("Make sure the daemon is running:", className="mb-0"),
                        html.Code("mise daemon --config daemon.toml"),
                    ],
                    color="danger",
                )
            ]

    logger.info("Dash app created")
    return app


def _render_overview_tab(client: DaemonAPIClient) -> list:
    """Render Overview tab (health + config + game plan + charts).

    Args:
        client: API client

    Returns:
        Tab content
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


def _build_equity_curve(client: DaemonAPIClient) -> dbc.Alert | dcc.Graph:
    """Build equity curve from snapshots (last 30 days).

    Args:
        client: API client

    Returns:
        Graph or alert if no data
    """
    try:
        snapshots_resp = client.get_snapshots(days=30)
        if snapshots_resp.count == 0:
            return dbc.Alert("No portfolio history available", color="info")

        timestamps = [s.timestamp.astimezone(ZoneInfo("America/New_York")) for s in snapshots_resp.snapshots]
        values = [s.portfolio_value for s in snapshots_resp.snapshots]

        fig = go.Figure(
            data=[
                go.Scatter(
                    x=timestamps,
                    y=values,
                    mode="lines+markers",
                    line={"color": "rgb(55, 83, 109)", "width": 2},
                )
            ]
        )
        fig.update_layout(
            title="Portfolio Value (30 Days)",
            xaxis_title="Date",
            yaxis_title="Value ($)",
            height=300,
            margin={"l": 40, "r": 40, "t": 40, "b": 40},
        )
        return dcc.Graph(figure=fig)
    except Exception as e:
        logger.error(f"Failed to build equity curve: {e}")
        return dbc.Alert("Error loading portfolio history", color="danger")


def _build_allocation_pie(positions: list) -> dcc.Graph:
    """Build allocation pie by market value.

    Args:
        positions: List of PositionResponse

    Returns:
        Pie chart graph
    """
    labels = [p.symbol for p in positions]
    values = [p.current_qty * p.current_price for p in positions]

    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.3)])
    fig.update_layout(title="Portfolio Allocation", height=300, margin={"l": 20, "r": 20, "t": 40, "b": 20})
    return dcc.Graph(figure=fig)


def _build_summary_cards(positions: list) -> list:
    """Build summary metric cards.

    Args:
        positions: List of PositionResponse

    Returns:
        List of card rows
    """
    total_value = sum(p.current_qty * p.current_price for p in positions)
    total_pnl = sum((p.current_price - p.entry_price) * p.current_qty for p in positions)
    pnl_pct = (total_pnl / (total_value - total_pnl)) * 100 if total_value > total_pnl else 0
    avg_conf = sum(p.entry_confidence for p in positions) / len(positions) if positions else 0

    cards = [
        dbc.Card(dbc.CardBody([html.H5("Total Value"), html.H3(f"${total_value:,.2f}")])),
        dbc.Card(
            dbc.CardBody(
                [
                    html.H5("Total P&L"),
                    html.H3(
                        f"${total_pnl:+,.2f}",
                        style={"color": "green" if total_pnl > 0 else ("red" if total_pnl < 0 else "gray")},
                    ),
                ]
            )
        ),
        dbc.Card(
            dbc.CardBody(
                [
                    html.H5("P&L %"),
                    html.H3(
                        f"{pnl_pct:+.2f}%",
                        style={"color": "green" if pnl_pct > 0 else ("red" if pnl_pct < 0 else "gray")},
                    ),
                ]
            )
        ),
        dbc.Card(dbc.CardBody([html.H5("Avg Confidence"), html.H3(f"{avg_conf:.1%}")])),
    ]
    return [
        dbc.Row([dbc.Col(cards[0], width=6), dbc.Col(cards[1], width=6)]),
        dbc.Row([dbc.Col(cards[2], width=6), dbc.Col(cards[3], width=6)]),
    ]


def _build_rebalance_chart(client: DaemonAPIClient) -> dbc.Alert | dcc.Graph:
    """Build target vs actual weights grouped bar.

    Args:
        client: API client

    Returns:
        Graph or alert if no data
    """
    try:
        rebalance = client.get_rebalance()
        if not rebalance:
            return dbc.Alert("No rebalancing data available", color="info")

        symbols = [a.symbol for a in rebalance.allocations]
        targets = [a.target_weight * 100 for a in rebalance.allocations]
        actuals = [a.current_weight * 100 for a in rebalance.allocations]

        fig = go.Figure(
            data=[go.Bar(name="Target", x=symbols, y=targets), go.Bar(name="Current", x=symbols, y=actuals)]
        )
        fig.update_layout(
            title="Target vs Current Allocation",
            barmode="group",
            xaxis_title="Symbol",
            yaxis_title="Weight (%)",
            height=300,
            margin={"l": 40, "r": 40, "t": 40, "b": 40},
        )
        return dcc.Graph(figure=fig)
    except Exception as e:
        logger.error(f"Failed to build rebalance chart: {e}")
        return dbc.Alert("Error loading rebalancing data", color="danger")


def _render_portfolio_tab(client: DaemonAPIClient) -> list:
    """Render Portfolio tab (active positions).

    Args:
        client: API client

    Returns:
        Tab content
    """
    positions_resp = client.get_positions()

    if positions_resp.count == 0:
        return [dbc.Alert("No active positions", color="info")]

    table_rows = []
    for pos in positions_resp.positions:
        pnl_dollars = (pos.current_price - pos.entry_price) * pos.current_qty
        pnl_percent = ((pos.current_price / pos.entry_price) - 1) * 100 if pos.entry_price > 0 else 0
        pnl_color = "success" if pnl_dollars > 0 else "danger" if pnl_dollars < 0 else "secondary"

        table_rows.append(
            html.Tr(
                [
                    html.Td(pos.symbol),
                    html.Td(f"{pos.current_qty:.2f}"),
                    html.Td(f"${pos.entry_price:.2f}"),
                    html.Td(f"${pos.current_price:.2f}"),
                    html.Td(html.Span(f"${pnl_dollars:+,.2f}", className=f"badge bg-{pnl_color}")),
                    html.Td(html.Span(f"{pnl_percent:+.2f}%", className=f"badge bg-{pnl_color}")),
                    html.Td(f"${pos.current_stop_loss:.2f}"),
                ]
            )
        )

    table = dbc.Table(
        [
            html.Thead(
                html.Tr(
                    [
                        html.Th("Symbol"),
                        html.Th("Qty"),
                        html.Th("Entry"),
                        html.Th("Current"),
                        html.Th("P&L $"),
                        html.Th("P&L %"),
                        html.Th("Stop Loss"),
                    ]
                )
            ),
            html.Tbody(table_rows),
        ],
        bordered=True,
        hover=True,
        striped=True,
    )

    return [
        html.H4("Portfolio Overview"),
        html.Hr(),
        _build_equity_curve(client),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(_build_allocation_pie(positions_resp.positions), width=6),
                dbc.Col(_build_summary_cards(positions_resp.positions), width=6),
            ]
        ),
        html.Hr(),
        _build_rebalance_chart(client),
        html.Hr(),
        html.H5(f"Active Positions ({positions_resp.count})"),
        table,
    ]


def _render_signals_tab(client: DaemonAPIClient) -> list:
    """Render Signals tab (recent analyses).

    Args:
        client: API client

    Returns:
        Tab content
    """
    analyses_resp = client.get_analyses(limit=50)

    if analyses_resp.returned_count == 0:
        return [dbc.Alert("No analyses yet", color="info")]

    table_rows = []
    for analysis in analyses_resp.analyses:
        signal_color = (
            "success" if analysis.signal == "BUY" else "danger" if analysis.signal == "SELL" else "secondary"
        )
        session_badge = ""
        if analysis.trading_session == "PRE_MARKET":
            session_badge = html.Span(" (PRE-MARKET)", className="badge bg-info ms-2")

        table_rows.append(
            html.Tr(
                [
                    html.Td(
                        [
                            analysis.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                            session_badge,
                        ]
                    ),
                    html.Td(analysis.symbol),
                    html.Td(html.Span(analysis.signal, className=f"badge bg-{signal_color}")),
                    html.Td(f"{analysis.confidence:.2f}"),
                    html.Td("✓" if analysis.executed_trade else "✗"),
                    html.Td("📄" if analysis.is_paper_trade else "💵"),
                ]
            )
        )

    table = dbc.Table(
        [
            html.Thead(
                html.Tr(
                    [
                        html.Th("Timestamp"),
                        html.Th("Symbol"),
                        html.Th("Signal"),
                        html.Th("Confidence"),
                        html.Th("Executed"),
                        html.Th("Mode"),
                    ]
                )
            ),
            html.Tbody(table_rows),
        ],
        bordered=True,
        hover=True,
        striped=True,
    )

    return [html.H4(f"Recent Analyses ({analyses_resp.returned_count}/{analyses_resp.total_count})"), table]


def _render_risk_tab(client: DaemonAPIClient) -> list:
    """Render Risk tab (VaR/CVaR/drawdown).

    Args:
        client: API client

    Returns:
        Tab content
    """
    risk = client.get_risk()

    if not risk:
        return [dbc.Alert("No risk report available", color="info")]

    risk_status_color = (
        "success" if risk.risk_status == "LOW" else "warning" if risk.risk_status == "MEDIUM" else "danger"
    )

    cards = dbc.Row(
        [
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardBody(
                            [
                                html.H5("VaR 95%", className="card-title"),
                                html.H3(f"{risk.var_95:.2%}"),
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
                                html.H5("CVaR 95%", className="card-title"),
                                html.H3(f"{risk.cvar_95:.2%}"),
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
                                html.H5("Max Drawdown", className="card-title"),
                                html.H3(f"{risk.max_drawdown:.2%}", className="text-danger"),
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
                                html.H5("Risk Status", className="card-title"),
                                html.H3(risk.risk_status, className=f"text-{risk_status_color}"),
                            ]
                        )
                    ]
                ),
                width=3,
            ),
        ]
    )

    details_table = dbc.Table(
        [
            html.Thead(html.Tr([html.Th("Metric"), html.Th("Value")])),
            html.Tbody(
                [
                    html.Tr([html.Td("VaR 99%"), html.Td(f"{risk.var_99:.2%}")]),
                    html.Tr([html.Td("CVaR 99%"), html.Td(f"{risk.cvar_99:.2%}")]),
                    html.Tr([html.Td("CDaR 95%"), html.Td(f"{risk.cdar_95:.2%}")]),
                    html.Tr([html.Td("Timestamp"), html.Td(risk.timestamp.strftime("%Y-%m-%d %H:%M:%S"))]),
                ]
            ),
        ],
        bordered=True,
        hover=True,
        striped=True,
        className="mt-4",
    )

    return [cards, details_table]


def _render_events_tab(client: DaemonAPIClient) -> list:
    """Render Events tab (event log).

    Args:
        client: API client

    Returns:
        Tab content
    """
    events_resp = client.get_events(limit=100)

    if events_resp.returned_count == 0:
        return [dbc.Alert("No events yet", color="info")]

    table_rows = []
    for event in events_resp.events:
        event_type = event.get("event_type", "unknown")
        timestamp = event.get("timestamp")
        details = event.get("details", {})

        if timestamp:
            try:
                ts_obj = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                timestamp_str = ts_obj.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                timestamp_str = str(timestamp)
        else:
            timestamp_str = "Unknown"

        details_str = str(details) if details else "-"
        max_len = 100
        if len(details_str) > max_len:
            details_str = details_str[: max_len - 3] + "..."

        table_rows.append(
            html.Tr(
                [
                    html.Td(timestamp_str),
                    html.Td(event_type),
                    html.Td(details_str, className="font-monospace small"),
                ]
            )
        )

    table = dbc.Table(
        [
            html.Thead(html.Tr([html.Th("Timestamp"), html.Th("Type"), html.Th("Details")])),
            html.Tbody(table_rows),
        ],
        bordered=True,
        hover=True,
        striped=True,
    )

    return [html.H4(f"Recent Events ({events_resp.returned_count})"), table]
