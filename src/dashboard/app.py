"""Dash app factory with layout and callbacks."""

import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, dcc, html
from loguru import logger

from src.dashboard.api_client import DaemonAPIClient
from src.dashboard.config import DashboardConfig
from src.dashboard.tabs import config as config_tab
from src.dashboard.tabs import events, overview, portfolio, risk, signals


def create_dash_app(config: DashboardConfig) -> Dash:
    """Create Dash app with layout and callbacks.

    Args:
        config: Dashboard configuration

    Returns:
        Dash app instance
    """
    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
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
                    dbc.Tab(label="Config", tab_id="config"),
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
                return overview.render(app.api_client)
            if active_tab == "portfolio":
                return portfolio.render(app.api_client)
            if active_tab == "signals":
                return signals.render(app.api_client)
            if active_tab == "risk":
                return risk.render(app.api_client)
            if active_tab == "events":
                return events.render(app.api_client)
            if active_tab == "config":
                return config_tab.render(app.api_client)
            return [html.Div("Invalid tab")]
        except Exception as e:
            logger.exception("Tab render failed")
            return [
                dbc.Alert(
                    [
                        html.H4("Error", className="alert-heading"),
                        html.P("Failed to load tab from the AI Casino daemon."),
                        html.P(
                            "This usually means the daemon process is not running or is not reachable. "
                            f"Please verify the daemon is running and accessible at: {config.api_url}"
                        ),
                        html.Small(f"Details: {e!s}"),
                    ],
                    color="danger",
                )
            ]

    # Register tab callbacks
    overview.register_callbacks(app)
    portfolio.register_callbacks(app)
    signals.register_callbacks(app)
    risk.register_callbacks(app)
    events.register_callbacks(app)
    config_tab.register_callbacks(app)

    return app
