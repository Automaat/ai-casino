"""Dash app factory with layout and callbacks."""

import logging

import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, dcc, html
from loguru import logger

from src.dashboard.api_client import DaemonAPIClient
from src.dashboard.config import DashboardConfig
from src.dashboard.tabs import config as config_tab
from src.dashboard.tabs import events, overview, portfolio, risk, signals, workflow


def _setup_logging(debug: bool) -> None:
    """Configure logging based on debug mode.

    Args:
        debug: Enable debug mode (shows all HTTP requests)
    """
    if not debug:
        log = logging.getLogger("werkzeug")
        log.setLevel(logging.ERROR)


def _render_tab_by_id(active_tab: str, api_client: DaemonAPIClient) -> list:
    """Route to appropriate tab renderer.

    Args:
        active_tab: Active tab ID
        api_client: API client for daemon communication

    Returns:
        Tab content
    """
    tab_renderers = {
        "overview": overview.render,
        "portfolio": portfolio.render,
        "signals": signals.render,
        "risk": risk.render,
        "events": events.render,
        "workflow": workflow.render,
        "config": config_tab.render,
    }

    renderer = tab_renderers.get(active_tab)
    if renderer:
        return renderer(api_client)
    return [html.Div("Invalid tab")]


def _create_error_alert(config: DashboardConfig, error: Exception) -> list:
    """Create error alert with daemon connection guidance.

    Args:
        config: Dashboard configuration
        error: Exception that occurred

    Returns:
        Alert component
    """
    return [
        dbc.Alert(
            [
                html.H4("Error", className="alert-heading"),
                html.P("Failed to load tab from the AI Casino daemon."),
                html.P(
                    "This usually means the daemon process is not running or is not reachable. "
                    f"Please verify the daemon is running and accessible at: {config.api_url}"
                ),
                html.Small(f"Details: {error!s}"),
            ],
            color="danger",
        )
    ]


def create_dash_app(config: DashboardConfig, debug: bool = False) -> Dash:
    """Create Dash app with layout and callbacks.

    Args:
        config: Dashboard configuration
        debug: Enable debug mode (shows all HTTP requests)

    Returns:
        Dash app instance
    """
    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
    app.title = "AI Casino Dashboard"

    _setup_logging(debug)

    # Store API client in app state (dynamic attribute - not in Dash type stubs)
    app.api_client = DaemonAPIClient(config.api_url)  # type: ignore[attr-defined]

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
                    dbc.Tab(label="Workflow", tab_id="workflow"),
                    dbc.Tab(label="Config", tab_id="config"),
                ],
            ),
            html.Div(id="tab-content", className="mt-4"),
        ],
        fluid=True,
    )

    @app.callback(
        Output("tab-content", "children"),
        Input("tabs", "active_tab"),
    )
    def render_tab_content(active_tab: str) -> list:
        """Render tab content (triggered by tab switch only).

        Args:
            active_tab: Active tab ID

        Returns:
            Tab content
        """
        try:
            return _render_tab_by_id(active_tab, app.api_client)
        except Exception as e:
            logger.exception("Tab render failed")
            return _create_error_alert(config, e)

    # Register tab callbacks
    overview.register_callbacks(app)
    portfolio.register_callbacks(app)
    signals.register_callbacks(app)
    risk.register_callbacks(app)
    events.register_callbacks(app)
    workflow.register_callbacks(app)
    config_tab.register_callbacks(app)

    return app
