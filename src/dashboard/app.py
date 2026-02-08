"""Dash app factory with layout and callbacks."""

from datetime import datetime

import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, dcc, html
from loguru import logger

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
    """Render Overview tab (health + config).

    Args:
        client: API client

    Returns:
        Tab content
    """
    health = client.get_health()
    summary = client.get_state_summary()
    config = client.get_config()
    degradation = client.get_degradation()

    status_color = "success" if health.status == "healthy" else "warning"

    cards = dbc.Row(
        [
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardBody(
                            [
                                html.H5("Status", className="card-title"),
                                html.H3(health.status.upper(), className=f"text-{status_color}"),
                                html.P(
                                    f"Running: {health.running}",
                                    className="card-text",
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

    config_table = dbc.Table(
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

    return [cards, html.H4("Configuration"), config_table]


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
        if pos.days_held < 30:  # noqa: PLR2004
            days_held_color = "success"
        elif pos.days_held < 90:  # noqa: PLR2004
            days_held_color = "warning"
        else:
            days_held_color = "danger"
        table_rows.append(
            html.Tr(
                [
                    html.Td(pos.symbol),
                    html.Td(f"{pos.current_qty:.2f}"),
                    html.Td(f"${pos.entry_price:.2f}"),
                    html.Td(f"${pos.current_stop_loss:.2f}"),
                    html.Td(pos.entry_timestamp.strftime("%Y-%m-%d %H:%M")),
                    html.Td(html.Span(f"{pos.days_held} days", className=f"badge bg-{days_held_color}")),
                    html.Td(pos.entry_signal),
                    html.Td(f"{pos.entry_confidence:.2f}"),
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
                        html.Th("Stop Loss"),
                        html.Th("Entry Time"),
                        html.Th("Days Held"),
                        html.Th("Signal"),
                        html.Th("Confidence"),
                    ]
                )
            ),
            html.Tbody(table_rows),
        ],
        bordered=True,
        hover=True,
        striped=True,
    )

    return [html.H4(f"Active Positions ({positions_resp.count})"), table]


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
