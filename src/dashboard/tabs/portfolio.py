"""Portfolio tab - active positions, equity curve, allocation."""

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate
from loguru import logger

from src.dashboard.api_client import DaemonAPIClient


def render(client: DaemonAPIClient) -> list:  # noqa: ARG001
    """Render Portfolio tab static structure with dcc.Store.

    Args:
        client: API client

    Returns:
        Tab content components
    """
    return [
        dcc.Store(id="portfolio-data-store", data=None),
        html.Div(id="portfolio-timestamp", className="text-muted small mb-3", children="Last updated: -"),
        html.H4("Portfolio Overview"),
        html.Hr(),
        html.Div(id="portfolio-equity-curve"),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(html.Div(id="portfolio-allocation-pie"), width=6),
                dbc.Col(html.Div(id="portfolio-summary-cards"), width=6),
            ]
        ),
        html.Hr(),
        html.Div(id="portfolio-rebalance-chart"),
        html.Hr(),
        html.Div(id="portfolio-positions-header"),
        html.Div(id="portfolio-positions-table"),
    ]


def register_callbacks(app: Dash) -> None:  # noqa: C901, PLR0915
    """Register Portfolio tab callbacks.

    Args:
        app: Dash app instance
    """
    client = app.api_client  # type: ignore[attr-defined]

    @app.callback(
        Output("portfolio-data-store", "data"),
        Input("interval-component", "n_intervals"),
        State("tabs", "active_tab"),
        State("portfolio-data-store", "data"),
    )
    def update_portfolio_data(n_intervals: int, active_tab: str, current_data: dict | None) -> dict:  # noqa: ARG001
        """Update Portfolio store when tab active.

        Args:
            n_intervals: Interval counter
            active_tab: Active tab ID
            current_data: Current store data

        Returns:
            Serialized portfolio data
        """
        if active_tab != "portfolio":
            raise PreventUpdate

        try:
            positions_resp = client.get_positions()
            snapshots_resp = client.get_snapshots(days=30)
            rebalance = client.get_rebalance()

            return {
                "timestamp": datetime.now(UTC).isoformat(),
                "positions": [
                    {
                        "symbol": p.symbol,
                        "current_qty": p.current_qty,
                        "entry_price": p.entry_price,
                        "current_price": p.current_price,
                        "current_stop_loss": p.current_stop_loss,
                        "entry_confidence": p.entry_confidence,
                    }
                    for p in positions_resp.positions
                ],
                "snapshots": [
                    {
                        "timestamp": s.timestamp.isoformat(),
                        "portfolio_value": s.portfolio_value,
                    }
                    for s in snapshots_resp.snapshots
                ],
                "rebalance": {
                    "allocations": [
                        {
                            "symbol": a.symbol,
                            "target_weight": a.target_weight,
                            "current_weight": a.current_weight,
                        }
                        for a in rebalance.allocations
                    ]
                }
                if rebalance
                else None,
            }
        except Exception as e:
            logger.error(f"Portfolio refresh failed: {e}")
            return current_data if current_data else {}

    @app.callback(
        Output("portfolio-timestamp", "children"),
        Input("portfolio-data-store", "data"),
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
        age = (datetime.now(UTC) - ts).total_seconds()
        return f"Last updated: {age:.0f}s ago"

    @app.callback(
        Output("portfolio-positions-header", "children"),
        Output("portfolio-positions-table", "children"),
        Input("portfolio-data-store", "data"),
    )
    def update_positions_table(data: dict | None) -> tuple[str | html.H5, dbc.Alert | dbc.Table]:
        """Update positions table.

        Args:
            data: Store data

        Returns:
            Tuple of (header, table)
        """
        if not data or not data.get("positions"):
            return "", dbc.Alert("No active positions", color="info")

        positions = data["positions"]

        table_rows = []
        for pos in positions:
            pnl_dollars = (pos["current_price"] - pos["entry_price"]) * pos["current_qty"]
            pnl_percent = (
                ((pos["current_price"] / pos["entry_price"]) - 1) * 100 if pos["entry_price"] > 0 else 0
            )
            pnl_color = "success" if pnl_dollars > 0 else "danger" if pnl_dollars < 0 else "secondary"

            table_rows.append(
                html.Tr(
                    [
                        html.Td(pos["symbol"]),
                        html.Td(f"{pos['current_qty']:.2f}"),
                        html.Td(f"${pos['entry_price']:.2f}"),
                        html.Td(f"${pos['current_price']:.2f}"),
                        html.Td(html.Span(f"${pnl_dollars:+,.2f}", className=f"badge bg-{pnl_color}")),
                        html.Td(html.Span(f"{pnl_percent:+.2f}%", className=f"badge bg-{pnl_color}")),
                        html.Td(f"${pos['current_stop_loss']:.2f}"),
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

        return html.H5(f"Active Positions ({len(positions)})"), table

    @app.callback(
        Output("portfolio-equity-curve", "children"),
        Input("portfolio-data-store", "data"),
    )
    def update_equity_curve(data: dict | None) -> dbc.Alert | dcc.Graph:
        """Update equity curve chart.

        Args:
            data: Store data

        Returns:
            Equity curve graph or alert
        """
        if not data or not data.get("snapshots"):
            return dbc.Alert("No portfolio history available", color="info")

        snapshots = data["snapshots"]
        timestamps = [
            datetime.fromisoformat(s["timestamp"]).astimezone(ZoneInfo("America/New_York")) for s in snapshots
        ]
        values = [s["portfolio_value"] for s in snapshots]

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

    @app.callback(
        Output("portfolio-allocation-pie", "children"),
        Input("portfolio-data-store", "data"),
    )
    def update_allocation_pie(data: dict | None) -> str | dcc.Graph:
        """Update allocation pie chart.

        Args:
            data: Store data

        Returns:
            Allocation pie chart
        """
        if not data or not data.get("positions"):
            return ""

        positions = data["positions"]
        labels = [p["symbol"] for p in positions]
        values = [p["current_qty"] * p["current_price"] for p in positions]

        fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.3)])
        fig.update_layout(
            title="Portfolio Allocation", height=300, margin={"l": 20, "r": 20, "t": 40, "b": 20}
        )
        return dcc.Graph(figure=fig)

    @app.callback(
        Output("portfolio-summary-cards", "children"),
        Input("portfolio-data-store", "data"),
    )
    def update_summary_cards(data: dict | None) -> list | str:
        """Update summary cards.

        Args:
            data: Store data

        Returns:
            Summary cards rows
        """
        if not data or not data.get("positions"):
            return ""

        positions = data["positions"]
        total_value = sum(p["current_qty"] * p["current_price"] for p in positions)
        total_pnl = sum((p["current_price"] - p["entry_price"]) * p["current_qty"] for p in positions)
        pnl_pct = (total_pnl / (total_value - total_pnl)) * 100 if total_value > total_pnl else 0
        avg_conf = sum(p["entry_confidence"] for p in positions) / len(positions) if positions else 0

        cards = [
            dbc.Card(dbc.CardBody([html.H5("Total Value"), html.H3(f"${total_value:,.2f}")])),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H5("Total P&L"),
                        html.H3(
                            f"${total_pnl:+,.2f}",
                            style={
                                "color": "green" if total_pnl > 0 else ("red" if total_pnl < 0 else "gray")
                            },
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

    @app.callback(
        Output("portfolio-rebalance-chart", "children"),
        Input("portfolio-data-store", "data"),
    )
    def update_rebalance_chart(data: dict | None) -> dbc.Alert | dcc.Graph:
        """Update rebalance chart.

        Args:
            data: Store data

        Returns:
            Rebalance chart or alert
        """
        if not data or not data.get("rebalance"):
            return dbc.Alert("No rebalancing data available", color="info")

        rebalance = data["rebalance"]
        allocations = rebalance["allocations"]

        symbols = [a["symbol"] for a in allocations]
        targets = [a["target_weight"] * 100 for a in allocations]
        actuals = [a["current_weight"] * 100 for a in allocations]

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
