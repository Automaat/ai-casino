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
    CorrelationMatrixResponse,
    DegradationResponse,
    GamePlanResponse,
    HealthResponse,
    SectorRotationResponse,
    StateSummaryResponse,
    WatchlistResponse,
)
from src.daemon.scheduler import MarketScheduler
from src.dashboard.api_client import DaemonAPIClient
from src.dashboard.config import DashboardConfig

# Constants for events tab
_CONSECUTIVE_SIGNALS_THRESHOLD = 5
_HIGH_DRAWDOWN_THRESHOLD = 0.10
_STALENESS_THRESHOLD_MINUTES = 30
_DETAILS_MAX_LENGTH = 150


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


def _build_confidence_histogram(analyses: list) -> dbc.Alert | dcc.Graph:
    """Build confidence distribution histogram.

    Args:
        analyses: List of AnalysisRecordResponse

    Returns:
        Graph or alert if no data
    """
    if not analyses:
        return dbc.Alert("No data available", color="info")

    confidences = [a.confidence for a in analyses]

    fig = go.Figure(data=[go.Histogram(x=confidences, nbinsx=20, marker_color="#22c55e")])
    fig.update_layout(
        title="Confidence Distribution",
        xaxis_title="Confidence",
        yaxis_title="Count",
        height=300,
        margin={"l": 40, "r": 40, "t": 40, "b": 40},
    )

    return dcc.Graph(figure=fig)


def _build_signal_breakdown_chart(analyses: list) -> dbc.Alert | dcc.Graph:
    """Build signal breakdown stacked bar per symbol.

    Args:
        analyses: List of AnalysisRecordResponse

    Returns:
        Graph or alert if no data
    """
    if not analyses:
        return dbc.Alert("No data available", color="info")

    symbol_signals: dict[str, dict[str, int]] = {}
    for a in analyses:
        if a.symbol not in symbol_signals:
            symbol_signals[a.symbol] = {"BUY": 0, "SELL": 0, "HOLD": 0}
        symbol_signals[a.symbol][a.signal] += 1

    symbols = sorted(symbol_signals.keys())
    buy_counts = [symbol_signals[s]["BUY"] for s in symbols]
    sell_counts = [symbol_signals[s]["SELL"] for s in symbols]
    hold_counts = [symbol_signals[s]["HOLD"] for s in symbols]

    fig = go.Figure(
        data=[
            go.Bar(name="BUY", x=symbols, y=buy_counts, marker_color="#22c55e"),
            go.Bar(name="SELL", x=symbols, y=sell_counts, marker_color="#ef4444"),
            go.Bar(name="HOLD", x=symbols, y=hold_counts, marker_color="#6b7280"),
        ]
    )
    fig.update_layout(
        title="Signal Breakdown by Symbol",
        barmode="stack",
        xaxis_title="Symbol",
        yaxis_title="Count",
        height=300,
        margin={"l": 40, "r": 40, "t": 40, "b": 40},
    )

    return dcc.Graph(figure=fig)


def _build_rsi_gauge(analyses: list) -> dbc.Alert | dcc.Graph:
    """Build RSI gauge with colored ranges.

    Args:
        analyses: List of AnalysisRecordResponse

    Returns:
        Graph or alert if no data
    """
    rsi_values = [a.rsi for a in analyses if a.rsi is not None]

    if not rsi_values:
        return dbc.Alert("No RSI data available", color="info")

    latest_rsi = rsi_values[0]

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=latest_rsi,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Latest RSI"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, 30], "color": "#22c55e"},
                    {"range": [30, 70], "color": "#fbbf24"},
                    {"range": [70, 100], "color": "#ef4444"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 4},
                    "thickness": 0.75,
                    "value": latest_rsi,
                },
            },
        )
    )
    fig.update_layout(height=250, margin={"l": 20, "r": 20, "t": 40, "b": 20})

    return dcc.Graph(figure=fig)


def _build_macd_histogram(analyses: list) -> dbc.Alert | dcc.Graph:
    """Build MACD histogram with green/red bars.

    Args:
        analyses: List of AnalysisRecordResponse

    Returns:
        Graph or alert if no data
    """
    macd_data = [(a.timestamp, a.macd_hist) for a in analyses if a.macd_hist is not None]

    if not macd_data:
        return dbc.Alert("No MACD data available", color="info")

    macd_data = macd_data[:20]
    macd_data.reverse()

    timestamps = [d[0].strftime("%m/%d %H:%M") for d in macd_data]
    values = [d[1] for d in macd_data]
    colors = ["#22c55e" if v > 0 else "#ef4444" for v in values]

    fig = go.Figure(data=[go.Bar(x=timestamps, y=values, marker_color=colors)])
    fig.update_layout(
        title="MACD Histogram (Last 20)",
        xaxis_title="Time",
        yaxis_title="MACD Histogram",
        height=300,
        margin={"l": 40, "r": 40, "t": 40, "b": 80},
        xaxis_tickangle=-45,
    )

    return dcc.Graph(figure=fig)


def _render_signals_tab(client: DaemonAPIClient) -> list:
    """Render Signals tab (recent analyses with filters and indicators).

    Args:
        client: API client

    Returns:
        Tab content
    """
    analyses_resp = client.get_analyses(limit=200)

    if analyses_resp.returned_count == 0:
        return [dbc.Alert("No analyses yet", color="info")]

    unique_symbols = sorted({a.symbol for a in analyses_resp.analyses})

    filter_controls = dbc.Card(
        dbc.CardBody(
            [
                html.H5("Filters"),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label("Symbols"),
                                dcc.Dropdown(
                                    id="signals-filter-symbol",
                                    options=[{"label": s, "value": s} for s in unique_symbols],
                                    value=[],
                                    multi=True,
                                    placeholder="All symbols",
                                ),
                            ],
                            width=4,
                        ),
                        dbc.Col(
                            [
                                html.Label("Signal Type"),
                                dcc.Checklist(
                                    id="signals-filter-signal-type",
                                    options=[
                                        {"label": " BUY", "value": "BUY"},
                                        {"label": " SELL", "value": "SELL"},
                                        {"label": " HOLD", "value": "HOLD"},
                                    ],
                                    value=["BUY", "SELL", "HOLD"],
                                    inline=True,
                                ),
                            ],
                            width=4,
                        ),
                        dbc.Col(
                            [
                                html.Label("Date Range"),
                                dcc.DatePickerRange(
                                    id="signals-filter-date-range",
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

    analyses = analyses_resp.analyses

    visualizations = [
        html.H4("Technical Indicators & Signal Distribution"),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(_build_confidence_histogram(analyses), width=6),
                dbc.Col(_build_signal_breakdown_chart(analyses), width=6),
            ]
        ),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(_build_rsi_gauge(analyses), width=6),
                dbc.Col(_build_macd_histogram(analyses), width=6),
            ]
        ),
        html.Hr(),
    ]

    table_rows = []
    for analysis in analyses[:50]:
        signal_color = (
            "success" if analysis.signal == "BUY" else "danger" if analysis.signal == "SELL" else "secondary"
        )
        session_badge = ""
        if analysis.trading_session == "PRE_MARKET":
            session_badge = html.Span(" (PRE-MARKET)", className="badge bg-info ms-2")

        rsi_str = f"{analysis.rsi:.1f}" if analysis.rsi is not None else "-"
        macd_str = f"{analysis.macd_hist:.3f}" if analysis.macd_hist is not None else "-"

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
                    html.Td(rsi_str),
                    html.Td(macd_str),
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
                        html.Th("RSI"),
                        html.Th("MACD Hist"),
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

    return [
        html.H4(f"Signal History ({analyses_resp.returned_count}/{analyses_resp.total_count})"),
        filter_controls,
        *visualizations,
        html.H5("Recent Signals (Last 50)"),
        table,
    ]


def _build_var_gauge(value: float, title: str) -> dcc.Graph:
    """Build VaR/CVaR gauge indicator.

    Args:
        value: Risk metric value (0.0-1.0)
        title: Gauge title

    Returns:
        dcc.Graph with gauge indicator
    """
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value * 100,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": title},
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 50]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, 5], "color": "#22c55e"},
                    {"range": [5, 15], "color": "#fbbf24"},
                    {"range": [15, 50], "color": "#ef4444"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 4},
                    "thickness": 0.75,
                    "value": value * 100,
                },
            },
        )
    )
    fig.update_layout(height=300, margin={"l": 20, "r": 20, "t": 50, "b": 20})
    return dcc.Graph(figure=fig)


def _build_risk_trend_chart(history: list) -> dbc.Alert | dcc.Graph:
    """Build risk metrics trend chart.

    Args:
        history: List of RiskReportResponse

    Returns:
        dcc.Graph with trend lines or Alert if insufficient data
    """
    min_trend_points = 2

    if not history:
        return dbc.Alert("No historical risk data available", color="info")

    if len(history) < min_trend_points:
        return dbc.Alert("Insufficient data for trend (need 2+ points)", color="info")

    timestamps = [r.timestamp for r in history]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=[r.var_95 * 100 for r in history],
            mode="lines+markers",
            name="VaR 95%",
            line={"color": "#3b82f6", "width": 2},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=[r.cvar_95 * 100 for r in history],
            mode="lines+markers",
            name="CVaR 95%",
            line={"color": "#f59e0b", "width": 2},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=[abs(r.max_drawdown) * 100 for r in history],
            mode="lines+markers",
            name="Max Drawdown",
            line={"color": "#ef4444", "width": 2},
        )
    )

    fig.update_layout(
        title="Risk Metrics Over Time",
        xaxis_title="Date",
        yaxis_title="Percentage (%)",
        height=400,
        margin={"l": 40, "r": 40, "t": 40, "b": 40},
        hovermode="x unified",
    )
    return dcc.Graph(figure=fig)


def _build_sector_heatmap(rotation: SectorRotationResponse) -> dcc.Graph:
    """Build sector rotation heatmap.

    Args:
        rotation: SectorRotationResponse

    Returns:
        dcc.Graph with sector heatmap
    """
    sectors = list(rotation.sector_strengths.keys())
    strengths = [rotation.sector_strengths[s] for s in sectors]
    momenta = [rotation.sector_momenta[s] for s in sectors]

    fig = go.Figure(
        data=go.Heatmap(
            z=[[s] for s in strengths],
            x=["Relative Strength"],
            y=sectors,
            colorscale="RdYlGn",
            text=[[f"{s:.2f}%<br>{m}"] for s, m in zip(strengths, momenta, strict=True)],
            texttemplate="%{text}",
            hovertemplate="Sector: %{y}<br>Strength: %{z:.2f}%<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"Sector Rotation (Leading: {', '.join(rotation.leading_sectors)})",
        height=500,
        margin={"l": 150, "r": 40, "t": 80, "b": 40},
    )
    return dcc.Graph(figure=fig)


def _build_correlation_heatmap(matrix_data: CorrelationMatrixResponse) -> dcc.Graph:
    """Build correlation matrix heatmap.

    Args:
        matrix_data: CorrelationMatrixResponse

    Returns:
        dcc.Graph with correlation heatmap
    """
    symbols = matrix_data.symbols
    matrix = [[matrix_data.correlation_matrix[s1].get(s2, 0.0) for s2 in symbols] for s1 in symbols]

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=symbols,
            y=symbols,
            colorscale="RdBu_r",
            zmid=0,
            zmin=-1,
            zmax=1,
            text=matrix,
            texttemplate="%{text:.2f}",
            hovertemplate="Correlation<br>%{y} vs %{x}: %{z:.3f}<extra></extra>",
        )
    )

    max_corr = matrix_data.max_correlation
    avg_corr = matrix_data.avg_correlation
    fig.update_layout(
        title=f"Portfolio Correlation Matrix (Max: {max_corr:.2f}, Avg: {avg_corr:.2f})",
        height=600,
        margin={"l": 100, "r": 40, "t": 80, "b": 100},
        xaxis={"side": "bottom", "tickangle": -45},
        yaxis={"autorange": "reversed"},
    )
    return dcc.Graph(figure=fig)


def _render_risk_tab(client: DaemonAPIClient) -> list:
    """Render Risk tab (VaR/CVaR/drawdown + trends + heatmaps).

    Args:
        client: API client

    Returns:
        Tab content
    """
    risk = client.get_risk()
    if not risk:
        return [dbc.Alert("No risk report available", color="info")]

    # Gauges row
    gauges_row = dbc.Row(
        [
            dbc.Col(_build_var_gauge(risk.var_95, "VaR 95%"), width=4),
            dbc.Col(_build_var_gauge(risk.cvar_99, "CVaR 99%"), width=4),
            dbc.Col(_build_var_gauge(abs(risk.max_drawdown), "Max Drawdown"), width=4),
        ],
        className="mb-4",
    )

    # Risk status card
    status_to_color = {
        "HEALTHY": "success",
        "WARNING": "warning",
        "BREACH": "danger",
    }
    risk_color = status_to_color.get(risk.risk_status, "secondary")
    status_card = dbc.Card(
        [
            dbc.CardBody(
                [
                    html.H5("Risk Status", className="card-title"),
                    html.H3(risk.risk_status, className=f"text-{risk_color}"),
                    html.P(
                        f"Last updated: {risk.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
                        className="text-muted",
                    ),
                ]
            )
        ],
        className="mb-4",
    )

    # Risk trend
    try:
        risk_history = client.get_risk_history()
        trend_chart = _build_risk_trend_chart(risk_history.reports)
    except Exception as e:
        logger.error(f"Failed to load risk history: {e}")
        trend_chart = dbc.Alert(f"Error loading risk trend: {e}", color="danger")

    # Sector heatmap
    try:
        sector_data = client.get_sector_rotation()
        sector_heatmap = (
            _build_sector_heatmap(sector_data)
            if sector_data
            else dbc.Alert("Sector rotation not enabled or no data available", color="info")
        )
    except Exception as e:
        logger.error(f"Failed to load sector rotation: {e}")
        sector_heatmap = dbc.Alert(f"Error loading sector data: {e}", color="danger")

    # Correlation matrix
    try:
        corr_data = client.get_correlation_matrix()
        corr_heatmap = (
            _build_correlation_heatmap(corr_data)
            if corr_data
            else dbc.Alert(
                "No correlation audit available. Enable correlation audit in daemon config.", color="info"
            )
        )
    except Exception as e:
        logger.error(f"Failed to load correlation matrix: {e}")
        corr_heatmap = dbc.Alert(f"Error loading correlation data: {e}", color="danger")

    # Details table
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
    )

    return [
        gauges_row,
        status_card,
        html.Hr(),
        html.H4("Risk Metrics Trend", className="mb-3"),
        trend_chart,
        html.Hr(),
        html.H4("Sector Rotation Strength", className="mb-3"),
        sector_heatmap,
        html.Hr(),
        html.H4("Portfolio Correlation Matrix", className="mb-3"),
        corr_heatmap,
        html.Hr(),
        html.H4("Detailed Metrics", className="mb-3"),
        details_table,
    ]


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
        state = client.get_state_summary()
        if len(state.analyses) >= _CONSECUTIVE_SIGNALS_THRESHOLD:
            recent_signals = [a["signal"] for a in state.analyses[:_CONSECUTIVE_SIGNALS_THRESHOLD]]
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
        risk = client.get_risk_report()
        if risk and abs(risk.max_drawdown) > _HIGH_DRAWDOWN_THRESHOLD:
            return {
                "severity": "danger",
                "icon": "🔴",
                "message": f"High drawdown: {abs(risk.max_drawdown):.1%}",
                "details": f"Risk status: {risk.risk_status}",
            }
    except Exception as e:
        logger.debug(f"Drawdown check failed: {e}")
    return None


def _check_data_staleness(client: DaemonAPIClient) -> dict | None:
    """Check for stale data during market hours."""
    try:
        state = client.get_state_summary()
        config = client.get_config()
        scheduler = MarketScheduler(config.daemon.schedule)

        if state.last_run and scheduler.is_market_open():
            staleness = (datetime.now(UTC) - state.last_run).total_seconds() / 60
            if staleness > _STALENESS_THRESHOLD_MINUTES:
                return {
                    "severity": "warning",
                    "icon": "🟡",
                    "message": f"Stale data: {staleness:.0f}min since last analysis",
                    "details": f"Last run: {state.last_run.strftime('%H:%M:%S %Z')}",
                }
    except Exception as e:
        logger.debug(f"Staleness check failed: {e}")
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
        _check_data_staleness,
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


def _render_error_log(errors: list[str]) -> html.Div:
    """Render error log section.

    Args:
        errors: List of error strings

    Returns:
        Div containing error log or empty div
    """
    if not errors:
        return html.Div()

    error_rows = []
    for err in errors[-20:]:
        # Parse "2024-01-15 10:23:45: Error message" or use raw
        parts = err.split(": ", 1)
        expected_parts = 2
        timestamp = parts[0] if len(parts) == expected_parts else "Unknown"
        message = parts[1] if len(parts) == expected_parts else err

        error_rows.append(
            html.Tr(
                [
                    html.Td(timestamp, className="font-monospace small"),
                    html.Td(message, style={"color": "#ef4444"}),
                ]
            )
        )

    return html.Div(
        [
            html.H5("Error Log", className="mt-4 mb-3"),
            dbc.Table(
                [
                    html.Thead(html.Tr([html.Th("Timestamp"), html.Th("Error Message")])),
                    html.Tbody(error_rows),
                ],
                bordered=True,
                hover=True,
                size="sm",
                striped=True,
            ),
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


def _render_events_tab(client: DaemonAPIClient) -> list:
    """Render Events tab with filters, timeline, warnings, event log, error log.

    Args:
        client: API client

    Returns:
        Tab content
    """
    try:
        # Fetch all data
        system_events = client.get_events(limit=100).events
        market_events_resp = client.get_market_events(limit=100)
        market_events = market_events_resp.get("events", [])
        degradation_history = client.get_degradation_history(limit=50)
        state = client.get_state_summary()

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

        # Generate warnings
        warnings = _generate_warnings(client)
        warnings_banner = _render_warnings_banner(warnings)

        # Build degradation timeline
        degradation_records = degradation_history.get("records", [])
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

        # Build event table
        table = _build_event_table(all_events)

        # Build error log
        errors = state.errors if hasattr(state, "errors") else []
        error_log = _render_error_log(errors)

        return [
            html.H4("Events & Monitoring"),
            warnings_banner,
            timeline,
            filters,
            html.H5(f"Event Log ({len(all_events)} events)", className="mt-4 mb-3"),
            table,
            error_log,
        ]

    except Exception as e:
        logger.error(f"Events tab render failed: {e}")
        return [dbc.Alert(f"Failed to load events: {e!s}", color="danger")]
