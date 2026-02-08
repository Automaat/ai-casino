"""Config tab - full daemon configuration with masking."""

import dash_bootstrap_components as dbc
from dash import Dash, html

from src.daemon.api import FullConfigResponse
from src.dashboard.api_client import DaemonAPIClient


def render(client: DaemonAPIClient) -> list:
    """Render Config tab content.

    Args:
        client: API client

    Returns:
        Tab content components
    """
    try:
        config = client.get_full_config()
    except Exception as e:
        return [dbc.Alert(f"Failed to load config: {e}", color="danger")]

    return [
        _build_top_level_summary(config),
        html.Hr(),
        _build_config_accordion(config),
    ]


def register_callbacks(app: Dash) -> None:
    """Register Config tab callbacks (none needed).

    Args:
        app: Dash app instance
    """


def _build_top_level_summary(config: FullConfigResponse) -> dbc.Row:
    """Build 6 core config fields as cards."""
    return dbc.Row(
        [
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H5("Watchlist", className="card-title"),
                            html.Div(
                                [
                                    html.Span(symbol, className="badge bg-primary me-1")
                                    for symbol in config.watchlist
                                ]
                            ),
                        ]
                    )
                ),
                width=4,
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H5("Interval", className="card-title"),
                            html.H3(f"{config.interval_minutes} min"),
                        ]
                    )
                ),
                width=4,
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H5("Market Hours Only", className="card-title"),
                            html.H3(
                                html.Span(
                                    "✓" if config.market_hours_only else "✗",
                                    style={"color": "green" if config.market_hours_only else "red"},
                                )
                            ),
                        ]
                    )
                ),
                width=4,
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H5("Auto Trade", className="card-title"),
                            html.H3(
                                html.Span(
                                    "✓" if config.auto_trade else "✗",
                                    style={"color": "green" if config.auto_trade else "red"},
                                )
                            ),
                        ]
                    )
                ),
                width=4,
                className="mt-3",
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H5("Trading Mode", className="card-title"),
                            html.H3(config.trading_mode.upper()),
                        ]
                    )
                ),
                width=4,
                className="mt-3",
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H5("Pre-Market", className="card-title"),
                            html.H3(
                                html.Span(
                                    "✓" if config.schedule.get("enable_pre_market", False) else "✗",
                                    style={
                                        "color": "green"
                                        if config.schedule.get("enable_pre_market", False)
                                        else "red"
                                    },
                                )
                            ),
                        ]
                    )
                ),
                width=4,
                className="mt-3",
            ),
        ],
        className="mb-4",
    )


def _build_config_accordion(config: FullConfigResponse) -> dbc.Accordion:
    """Build 6 category accordions."""
    categories = [
        {
            "title": "Trading & Execution",
            "sections": [
                ("Schedule", config.schedule, config.schedule.get("enable_pre_market", False)),
                ("Analysis Orchestration", config.analysis_orchestration, True),
            ],
        },
        {
            "title": "Risk Management",
            "sections": [
                ("Paper Trading", config.paper_trading, True),
                ("Risk Limits", config.risk_limits, config.risk_limits.get("enabled", False)),
                (
                    "Pre-Trade Backtesting",
                    config.pre_trade_backtesting,
                    config.pre_trade_backtesting.get("enabled", False),
                ),
                (
                    "Position Management",
                    config.position_management,
                    config.position_management.get("enabled", False),
                ),
                ("Monte Carlo", config.monte_carlo, config.monte_carlo.get("enabled", False)),
            ],
        },
        {
            "title": "Market Surveillance",
            "sections": [
                ("News Watcher", config.news_watcher, config.news_watcher.get("enabled", False)),
                ("Social Watcher", config.social_watcher, config.social_watcher.get("enabled", False)),
                ("Filings Watcher", config.filings_watcher, config.filings_watcher.get("enabled", False)),
                ("Anomaly Watcher", config.anomaly_watcher, config.anomaly_watcher.get("enabled", False)),
                (
                    "Earnings Calendar",
                    config.earnings_calendar,
                    config.earnings_calendar.get("enabled", False),
                ),
            ],
        },
        {
            "title": "After-Hours Operations",
            "sections": [
                ("Screening", config.screening, config.screening.get("enabled", False)),
                ("Prefetch", config.prefetch, config.prefetch.get("enabled", False)),
                ("Journal", config.journal, config.journal.get("enabled", False)),
                ("Health", config.health, config.health.get("enabled", False)),
                ("Optimization", config.optimization, config.optimization.get("enabled", False)),
                ("Sector Rotation", config.sector_rotation, config.sector_rotation.get("enabled", False)),
                ("Peer Analysis", config.peer_analysis, config.peer_analysis.get("enabled", False)),
                (
                    "Correlation Audit",
                    config.correlation_audit,
                    config.correlation_audit.get("enabled", False),
                ),
                ("Reporting", config.reporting, config.reporting.get("enabled", False)),
                ("Rebalancing", config.rebalancing, config.rebalancing.get("enabled", False)),
                ("Signal Tracking", config.signal_tracking, config.signal_tracking.get("enabled", False)),
                ("Game Plan", config.game_plan, config.game_plan.get("enabled", False)),
            ],
        },
        {
            "title": "Scheduling & Infrastructure",
            "sections": [
                ("State", config.state, True),
                ("API", config.api, config.api.get("enabled", False)),
                ("Notifications", config.notifications, config.notifications.get("enabled", False)),
            ],
        },
        {
            "title": "LLM & API Keys",
            "sections": [
                ("LLM", config.llm, True),
                ("API Keys", config.api_keys, True),
            ],
        },
    ]

    accordion_items = []
    for idx, category in enumerate(categories):
        category_cards = [
            _build_section_card(name, data, enabled) for name, data, enabled in category["sections"]
        ]

        accordion_items.append(
            dbc.AccordionItem(
                category_cards,
                title=category["title"],
                item_id=f"category-{idx}",
            )
        )

    # First category (Trading) open by default
    return dbc.Accordion(accordion_items, start_collapsed=False, always_open=True, active_item="category-0")


def _build_section_card(section_name: str, section_data: dict, enabled: bool) -> dbc.Card:
    """Build card for config section with enabled badge and field table."""
    # Special handling for API Keys section - mask values
    if section_name == "API Keys":
        field_rows = []
        for key, value in sorted(section_data.items()):
            # Already masked by API
            field_rows.append(html.Tr([html.Td(key), html.Td(value)]))
    else:
        field_rows = []
        for key, value in sorted(section_data.items()):
            formatted_value = _format_value(value)
            field_rows.append(html.Tr([html.Td(key), html.Td(formatted_value)]))

    enabled_badge = html.Span(
        "✓ Enabled" if enabled else "✗ Disabled",
        className=f"badge bg-{'success' if enabled else 'secondary'} float-end",
    )

    border_color = "success" if enabled else "secondary"

    return dbc.Card(
        [
            dbc.CardHeader(
                [
                    html.Span(section_name, className="fw-bold"),
                    enabled_badge,
                ]
            ),
            dbc.CardBody(
                dbc.Table(
                    [
                        html.Thead(html.Tr([html.Th("Field"), html.Th("Value")])),
                        html.Tbody(field_rows),
                    ],
                    bordered=True,
                    hover=True,
                    striped=True,
                    size="sm",
                )
            ),
        ],
        className="mb-3",
        style={"border-left": f"4px solid {border_color}"},
    )


def _format_value(value: bool | list | dict | None | str | int | float) -> html.Span | list[html.Span]:
    """Format field value for display (booleans as icons, lists as badges)."""
    if isinstance(value, bool):
        return html.Span(
            "✓" if value else "✗",
            style={"color": "green" if value else "red"},
        )
    if isinstance(value, list):
        if not value:
            return html.Span("[]", className="text-muted")
        return [html.Span(str(item), className="badge bg-primary me-1") for item in value]
    if isinstance(value, dict):
        return html.Span(f"{len(value)} items", className="text-muted")
    if value is None:
        return html.Span("None", className="text-muted")
    return str(value)
