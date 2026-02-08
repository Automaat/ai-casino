"""Portfolio tab - active positions, equity curve, allocation."""

from zoneinfo import ZoneInfo

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import Dash, dcc, html
from loguru import logger

from src.dashboard.api_client import DaemonAPIClient


def render(client: DaemonAPIClient) -> list:
    """Render Portfolio tab content.

    Args:
        client: API client

    Returns:
        Tab content components
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


def register_callbacks(app: Dash) -> None:
    """Register Portfolio tab callbacks (none needed).

    Args:
        app: Dash app instance
    """


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
