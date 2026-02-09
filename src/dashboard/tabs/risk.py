"""Risk tab - VaR/CVaR/drawdown + trends + heatmaps."""

from datetime import UTC, datetime

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate
from loguru import logger

from src.dashboard.api_client import DaemonAPIClient


def render(client: DaemonAPIClient) -> list:  # noqa: ARG001
    """Render Risk tab static structure with dcc.Store.

    Args:
        client: API client

    Returns:
        Tab content components
    """
    return [
        dcc.Store(id="risk-data-store", data=None),
        html.Div(id="risk-timestamp", className="text-muted small mb-3", children="Last updated: -"),
        html.Div(id="risk-gauges-row"),
        html.Div(id="risk-status-card"),
        html.Hr(),
        html.H4("Risk Metrics Trend", className="mb-3"),
        html.Div(id="risk-trend-chart"),
        html.Hr(),
        html.H4("Sector Rotation Strength", className="mb-3"),
        html.Div(id="risk-sector-heatmap"),
        html.Hr(),
        html.H4("Portfolio Correlation Matrix", className="mb-3"),
        html.Div(id="risk-correlation-heatmap"),
        html.Hr(),
        html.H4("Detailed Metrics", className="mb-3"),
        html.Div(id="risk-details-table"),
    ]


def register_callbacks(app: Dash) -> None:  # noqa: C901, PLR0915
    """Register Risk tab callbacks.

    Args:
        app: Dash app instance
    """
    client = app.api_client  # type: ignore[attr-defined]

    @app.callback(
        Output("risk-data-store", "data"),
        Input("interval-component", "n_intervals"),
        State("tabs", "active_tab"),
        State("risk-data-store", "data"),
    )
    def update_risk_data(n_intervals: int, active_tab: str, current_data: dict | None) -> dict:  # noqa: ARG001
        """Update Risk store when tab active.

        Args:
            n_intervals: Interval counter
            active_tab: Active tab ID
            current_data: Current store data

        Returns:
            Serialized risk data
        """
        if active_tab != "risk":
            raise PreventUpdate

        try:
            risk = client.get_risk()
            risk_history = client.get_risk_history()
            sector_data = client.get_sector_rotation()
            corr_data = client.get_correlation_matrix()

            return {
                "timestamp": datetime.now(UTC).isoformat(),
                "risk": {
                    "var_95": risk.var_95,
                    "var_99": risk.var_99,
                    "cvar_95": risk.cvar_95,
                    "cvar_99": risk.cvar_99,
                    "cdar_95": risk.cdar_95,
                    "max_drawdown": risk.max_drawdown,
                    "risk_status": risk.risk_status,
                    "risk_timestamp": risk.timestamp.isoformat(),
                }
                if risk
                else None,
                "history": [
                    {
                        "timestamp": r.timestamp.isoformat(),
                        "var_95": r.var_95,
                        "cvar_95": r.cvar_95,
                        "max_drawdown": r.max_drawdown,
                    }
                    for r in risk_history.reports
                ],
                "sector": {
                    "sector_strengths": sector_data.sector_strengths,
                    "sector_momenta": sector_data.sector_momenta,
                    "leading_sectors": sector_data.leading_sectors,
                }
                if sector_data
                else None,
                "correlation": {
                    "symbols": corr_data.symbols,
                    "correlation_matrix": corr_data.correlation_matrix,
                    "max_correlation": corr_data.max_correlation,
                    "avg_correlation": corr_data.avg_correlation,
                }
                if corr_data
                else None,
            }
        except Exception as e:
            logger.error(f"Risk refresh failed: {e}")
            return current_data if current_data else {}

    @app.callback(
        Output("risk-timestamp", "children"),
        Input("risk-data-store", "data"),
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
        Output("risk-gauges-row", "children"),
        Input("risk-data-store", "data"),
    )
    def update_gauges(data: dict | None) -> str | dbc.Row:
        """Update risk gauges.

        Args:
            data: Store data

        Returns:
            Gauges row
        """
        if not data or not data.get("risk"):
            return ""

        risk = data["risk"]

        return dbc.Row(
            [
                dbc.Col(_build_var_gauge(risk["var_95"], "VaR 95%"), width=4),
                dbc.Col(_build_var_gauge(risk["cvar_99"], "CVaR 99%"), width=4),
                dbc.Col(_build_var_gauge(abs(risk["max_drawdown"]), "Max Drawdown"), width=4),
            ],
            className="mb-4",
        )

    @app.callback(
        Output("risk-status-card", "children"),
        Input("risk-data-store", "data"),
    )
    def update_status_card(data: dict | None) -> str | dbc.Card:
        """Update risk status card.

        Args:
            data: Store data

        Returns:
            Status card
        """
        if not data or not data.get("risk"):
            return ""

        risk = data["risk"]
        status_to_color = {
            "HEALTHY": "success",
            "WARNING": "warning",
            "BREACH": "danger",
        }
        risk_color = status_to_color.get(risk["risk_status"], "secondary")
        risk_ts = datetime.fromisoformat(risk["risk_timestamp"])

        return dbc.Card(
            [
                dbc.CardBody(
                    [
                        html.H5("Risk Status", className="card-title"),
                        html.H3(risk["risk_status"], className=f"text-{risk_color}"),
                        html.P(
                            f"Last updated: {risk_ts.strftime('%Y-%m-%d %H:%M:%S')}",
                            className="text-muted",
                        ),
                    ]
                )
            ],
            className="mb-4",
        )

    @app.callback(
        Output("risk-trend-chart", "children"),
        Input("risk-data-store", "data"),
    )
    def update_trend_chart(data: dict | None) -> dbc.Alert | dcc.Graph:
        """Update risk trend chart.

        Args:
            data: Store data

        Returns:
            Trend chart or alert
        """
        if not data or not data.get("history"):
            return dbc.Alert("No historical risk data available", color="info")

        history = data["history"]
        min_trend_points = 2
        if len(history) < min_trend_points:
            return dbc.Alert("Insufficient data for trend (need 2+ points)", color="info")

        timestamps = [datetime.fromisoformat(r["timestamp"]) for r in history]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=[r["var_95"] * 100 for r in history],
                mode="lines+markers",
                name="VaR 95%",
                line={"color": "#3b82f6", "width": 2},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=[r["cvar_95"] * 100 for r in history],
                mode="lines+markers",
                name="CVaR 95%",
                line={"color": "#f59e0b", "width": 2},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=[abs(r["max_drawdown"]) * 100 for r in history],
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

    @app.callback(
        Output("risk-sector-heatmap", "children"),
        Input("risk-data-store", "data"),
    )
    def update_sector_heatmap(data: dict | None) -> dbc.Alert | dcc.Graph:
        """Update sector rotation heatmap.

        Args:
            data: Store data

        Returns:
            Sector heatmap or alert
        """
        if not data or not data.get("sector"):
            return dbc.Alert("Sector rotation not enabled or no data available", color="info")

        sector = data["sector"]
        sectors = list(sector["sector_strengths"].keys())
        strengths = [sector["sector_strengths"][s] for s in sectors]
        momenta = [sector["sector_momenta"][s] for s in sectors]

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

        leading = ", ".join(sector["leading_sectors"])
        fig.update_layout(
            title=f"Sector Rotation (Leading: {leading})",
            height=500,
            margin={"l": 150, "r": 40, "t": 80, "b": 40},
        )
        return dcc.Graph(figure=fig)

    @app.callback(
        Output("risk-correlation-heatmap", "children"),
        Input("risk-data-store", "data"),
    )
    def update_correlation_heatmap(data: dict | None) -> dbc.Alert | dcc.Graph:
        """Update correlation matrix heatmap.

        Args:
            data: Store data

        Returns:
            Correlation heatmap or alert
        """
        if not data or not data.get("correlation"):
            return dbc.Alert(
                "No correlation audit available. Enable correlation audit in daemon config.", color="info"
            )

        corr = data["correlation"]
        symbols = corr["symbols"]
        matrix = [[corr["correlation_matrix"][s1].get(s2, 0.0) for s2 in symbols] for s1 in symbols]

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

        max_corr = corr["max_correlation"]
        avg_corr = corr["avg_correlation"]
        fig.update_layout(
            title=f"Portfolio Correlation Matrix (Max: {max_corr:.2f}, Avg: {avg_corr:.2f})",
            height=600,
            margin={"l": 100, "r": 40, "t": 80, "b": 100},
            xaxis={"side": "bottom", "tickangle": -45},
            yaxis={"autorange": "reversed"},
        )
        return dcc.Graph(figure=fig)

    @app.callback(
        Output("risk-details-table", "children"),
        Input("risk-data-store", "data"),
    )
    def update_details_table(data: dict | None) -> str | dbc.Table:
        """Update risk details table.

        Args:
            data: Store data

        Returns:
            Details table
        """
        if not data or not data.get("risk"):
            return ""

        risk = data["risk"]
        risk_ts = datetime.fromisoformat(risk["risk_timestamp"])

        return dbc.Table(
            [
                html.Thead(html.Tr([html.Th("Metric"), html.Th("Value")])),
                html.Tbody(
                    [
                        html.Tr([html.Td("VaR 99%"), html.Td(f"{risk['var_99']:.2%}")]),
                        html.Tr([html.Td("CVaR 99%"), html.Td(f"{risk['cvar_99']:.2%}")]),
                        html.Tr([html.Td("CDaR 95%"), html.Td(f"{risk['cdar_95']:.2%}")]),
                        html.Tr([html.Td("Timestamp"), html.Td(risk_ts.strftime("%Y-%m-%d %H:%M:%S"))]),
                    ]
                ),
            ],
            bordered=True,
            hover=True,
            striped=True,
        )


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
