"""Risk tab - VaR/CVaR/drawdown + trends + heatmaps."""

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import Dash, dcc, html
from loguru import logger

from src.daemon.api import CorrelationMatrixResponse, SectorRotationResponse
from src.dashboard.api_client import DaemonAPIClient


def render(client: DaemonAPIClient) -> list:
    """Render Risk tab content.

    Args:
        client: API client

    Returns:
        Tab content components
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


def register_callbacks(app: Dash) -> None:
    """Register Risk tab callbacks (none needed).

    Args:
        app: Dash app instance
    """


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
