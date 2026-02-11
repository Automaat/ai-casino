"""Signals tab - recent analyses with filters and indicators."""

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import MATCH, Dash, Input, Output, State, dcc, html

if TYPE_CHECKING:
    from src.dashboard.api_client import DaemonAPIClient


def render(client: DaemonAPIClient) -> list:  # noqa: ARG001
    """Render Signals tab content.

    Args:
        client: API client

    Returns:
        Tab content components
    """
    # Initial data loaded by interval callback
    unique_symbols = []  # Will be populated dynamically

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
                                    start_date=datetime.now(UTC) - timedelta(days=7),
                                    end_date=datetime.now(UTC),
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
        dcc.Store(id="signals-data-store", data=None),
        html.Div(id="signals-timestamp", className="text-muted small mb-3", children="Last updated: -"),
        html.Div(id="signals-header"),
        filter_controls,
        html.Div(id="signals-filtered-content"),
    ]


def register_callbacks(app: Dash) -> None:  # noqa: C901
    """Register Signals tab filter callbacks.

    Args:
        app: Dash app instance
    """
    client = app.api_client  # type: ignore[attr-defined]

    @app.callback(
        Output("signals-data-store", "data"),
        Input("interval-component", "n_intervals"),
        State("tabs", "active_tab"),
        State("signals-data-store", "data"),
    )
    def update_signals_data(n_intervals: int, active_tab: str, current_data: dict | None) -> dict:  # noqa: ARG001
        """Update Signals store when tab active.

        Args:
            n_intervals: Interval counter
            active_tab: Active tab ID
            current_data: Current store data

        Returns:
            Serialized analyses data with timestamp
        """
        from datetime import UTC, datetime

        from dash.exceptions import PreventUpdate
        from loguru import logger

        if active_tab != "signals":
            raise PreventUpdate

        try:
            analyses_resp = client.get_analyses(limit=200)

            return {
                "timestamp": datetime.now(UTC).isoformat(),
                "returned_count": analyses_resp.returned_count,
                "total_count": analyses_resp.total_count,
                "analyses": [
                    {
                        "timestamp": a.timestamp.isoformat(),
                        "symbol": a.symbol,
                        "signal": a.signal,
                        "confidence": a.confidence,
                        "rsi": a.rsi,
                        "macd_hist": a.macd_hist,
                        "executed_trade": a.executed_trade,
                        "is_paper_trade": a.is_paper_trade,
                        "trading_session": a.trading_session,
                        "reasoning": a.reasoning,
                    }
                    for a in analyses_resp.analyses
                ],
            }
        except Exception as e:
            logger.error(f"Signals refresh failed: {e}")
            return current_data or {}

    @app.callback(
        Output("signals-filter-symbol", "options"),
        Input("signals-data-store", "data"),
    )
    def update_symbol_options(data: dict | None) -> list[dict]:
        """Update symbol filter options from store.

        Args:
            data: Store data

        Returns:
            Symbol options list
        """
        if not data or not data.get("analyses"):
            return []
        symbols = sorted({a["symbol"] for a in data["analyses"]})
        return [{"label": s, "value": s} for s in symbols]

    @app.callback(
        Output("signals-timestamp", "children"),
        Input("signals-data-store", "data"),
    )
    def update_signals_timestamp(data: dict | None) -> str:
        """Update timestamp display.

        Args:
            data: Store data

        Returns:
            Timestamp text
        """
        from datetime import datetime

        if not data or "timestamp" not in data:
            return "Last updated: -"
        ts = datetime.fromisoformat(data["timestamp"])
        return f"Last updated: {ts.strftime('%Y-%m-%d %H:%M:%S')}"

    @app.callback(
        Output("signals-header", "children"),
        Input("signals-data-store", "data"),
    )
    def update_signals_header(data: dict | None) -> str | html.H4:
        """Update signals header.

        Args:
            data: Store data

        Returns:
            Header text
        """
        if not data:
            return ""
        return html.H4(f"Signal History ({data.get('returned_count', 0)}/{data.get('total_count', 0)})")

    @app.callback(
        Output("signals-filtered-content", "children"),
        [
            Input("signals-data-store", "data"),
            Input("signals-filter-symbol", "value"),
            Input("signals-filter-signal-type", "value"),
            Input("signals-filter-date-range", "start_date"),
            Input("signals-filter-date-range", "end_date"),
        ],
    )
    def filter_signals(
        store_data: dict | None, symbols: list, signal_types: list, start_date: str, end_date: str
    ) -> list:
        """Filter signals based on user selection.

        Args:
            symbols: Selected symbols list
            signal_types: Selected signal types list
            start_date: Start date string
            end_date: End date string
            store_data: Cached data from dcc.Store

        Returns:
            Filtered content (charts + table)
        """
        if not store_data or not store_data.get("analyses"):
            return [dbc.Alert("No data available", color="info")]

        analyses_data = store_data["analyses"]

        # Apply filters
        filtered = _apply_filters(analyses_data, symbols, signal_types, start_date, end_date)

        if not filtered:
            return [dbc.Alert("No data matches filters", color="info")]

        # Build visualizations
        visualizations = [
            html.H4("Technical Indicators & Signal Distribution"),
            html.Hr(),
            dbc.Row(
                [
                    dbc.Col(_build_confidence_histogram(filtered), width=6),
                    dbc.Col(_build_signal_breakdown_chart(filtered), width=6),
                ]
            ),
            html.Hr(),
            dbc.Row(
                [
                    dbc.Col(_build_rsi_gauge(filtered), width=6),
                    dbc.Col(_build_macd_histogram(filtered), width=6),
                ]
            ),
            html.Hr(),
        ]

        # Build table with collapsible rows
        table_rows = []

        for idx, analysis in enumerate(filtered[:50]):
            signal_color = (
                "success"
                if analysis["signal"] == "BUY"
                else "danger"
                if analysis["signal"] == "SELL"
                else "secondary"
            )
            session_badge = ""
            if analysis.get("trading_session") == "PRE_MARKET":
                session_badge = html.Span(" (PRE-MARKET)", className="badge bg-info ms-2")

            rsi_str = f"{analysis['rsi']:.1f}" if analysis["rsi"] is not None else "-"
            macd_str = f"{analysis['macd_hist']:.3f}" if analysis["macd_hist"] is not None else "-"
            timestamp_obj = datetime.fromisoformat(analysis["timestamp"])

            # Main row (clickable to expand)
            table_rows.append(
                html.Tr(
                    [
                        html.Td(
                            [
                                html.Span(
                                    "▶ ",
                                    id={"type": "expand-icon", "index": idx},
                                    style={"cursor": "pointer", "user-select": "none"},
                                ),
                                timestamp_obj.strftime("%Y-%m-%d %H:%M:%S"),
                                session_badge,
                            ],
                            id={"type": "expand-trigger", "index": idx},
                            style={"cursor": "pointer"},
                        ),
                        html.Td(analysis["symbol"]),
                        html.Td(html.Span(analysis["signal"], className=f"badge bg-{signal_color}")),
                        html.Td(f"{analysis['confidence']:.2f}"),
                        html.Td(rsi_str),
                        html.Td(macd_str),
                        html.Td("✓" if analysis["executed_trade"] else "✗"),
                        html.Td("📄" if analysis["is_paper_trade"] else "💵"),
                    ],
                    id={"type": "table-row", "index": idx},
                )
            )

            # Reasoning collapse row (hidden by default)
            reasoning_content = (
                html.Ul([html.Li(r) for r in analysis.get("reasoning", [])])
                if analysis.get("reasoning")
                else html.P("No reasoning available", className="text-muted fst-italic")
            )

            table_rows.append(
                html.Tr(
                    [
                        html.Td(
                            dbc.Collapse(
                                dbc.Card(
                                    dbc.CardBody(
                                        [
                                            html.Strong("Decision Reasoning:"),
                                            reasoning_content,
                                        ],
                                        className="bg-light",
                                    ),
                                    className="border-0",
                                ),
                                id={"type": "collapse", "index": idx},
                                is_open=False,
                            ),
                            colSpan=8,
                            className="p-0",
                        )
                    ],
                    id={"type": "collapse-row", "index": idx},
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
            *visualizations,
            html.H5(f"Recent Signals ({len(filtered)} filtered, showing {min(50, len(filtered))})"),
            table,
        ]

    @app.callback(
        Output({"type": "collapse", "index": MATCH}, "is_open"),
        Output({"type": "expand-icon", "index": MATCH}, "children"),
        Input({"type": "expand-trigger", "index": MATCH}, "n_clicks"),
        State({"type": "collapse", "index": MATCH}, "is_open"),
        prevent_initial_call=True,
    )
    def toggle_reasoning(n_clicks: int, is_open: bool) -> tuple[bool, str]:
        """Toggle reasoning collapse on row click.

        Args:
            n_clicks: Number of clicks on trigger
            is_open: Current collapse state

        Returns:
            Tuple of (new_is_open, icon_text)
        """
        if n_clicks:
            new_state = not is_open
            icon = "▼ " if new_state else "▶ "
            return new_state, icon
        return is_open, "▶ "


def _apply_filters(analyses: list, symbols: list, signal_types: list, start_date: str, end_date: str) -> list:
    """Filter analyses by criteria.

    Args:
        analyses: List of analysis dicts
        symbols: Selected symbols (empty = all)
        signal_types: Selected signal types
        start_date: Start date string
        end_date: End date string

    Returns:
        Filtered list of analyses
    """
    filtered = analyses

    # Symbol filter
    if symbols:
        filtered = [a for a in filtered if a["symbol"] in symbols]

    # Signal type filter
    if signal_types:
        filtered = [a for a in filtered if a["signal"] in signal_types]

    # Date filter
    if start_date and end_date:
        start_dt = datetime.fromisoformat(start_date).replace(tzinfo=UTC)
        end_dt = datetime.fromisoformat(end_date).replace(hour=23, minute=59, second=59, tzinfo=UTC)
        filtered = [
            a
            for a in filtered
            if start_dt <= datetime.fromisoformat(a["timestamp"]).replace(tzinfo=UTC) <= end_dt
        ]

    return filtered


def _build_confidence_histogram(analyses: list) -> dbc.Alert | dcc.Graph:
    """Build confidence distribution histogram.

    Args:
        analyses: List of analysis dicts

    Returns:
        Graph or alert if no data
    """
    if not analyses:
        return dbc.Alert("No data available", color="info")

    confidences = [a["confidence"] for a in analyses]

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
        analyses: List of analysis dicts

    Returns:
        Graph or alert if no data
    """
    if not analyses:
        return dbc.Alert("No data available", color="info")

    symbol_signals: dict[str, dict[str, int]] = {}
    for a in analyses:
        if a["symbol"] not in symbol_signals:
            symbol_signals[a["symbol"]] = {"BUY": 0, "SELL": 0, "HOLD": 0}
        symbol_signals[a["symbol"]][a["signal"]] += 1

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
        analyses: List of analysis dicts

    Returns:
        Graph or alert if no data
    """
    rsi_values = [a["rsi"] for a in analyses if a["rsi"] is not None]

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
        analyses: List of analysis dicts

    Returns:
        Graph or alert if no data
    """
    macd_data = [
        (datetime.fromisoformat(a["timestamp"]), a["macd_hist"])
        for a in analyses
        if a["macd_hist"] is not None
    ]

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
