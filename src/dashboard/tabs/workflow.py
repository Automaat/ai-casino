"""Workflow tab - real-time execution status and historical analysis."""

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html
from loguru import logger

if TYPE_CHECKING:
    from src.dashboard.api_client import DaemonAPIClient


def render(client: "DaemonAPIClient") -> list:  # noqa: ARG001
    """Render Workflow tab content.

    Args:
        client: API client

    Returns:
        Tab content components
    """
    return [
        dcc.Store(id="workflow-data-store"),
        dcc.Store(id="workflow-live-state"),
        dcc.Interval(id="workflow-interval", interval=5000),
        dbc.Tabs(
            [
                dbc.Tab(
                    label="Live Execution",
                    children=[
                        html.Div(id="workflow-live-status", className="mb-3 mt-3"),
                        html.Div(id="workflow-live-progress", className="mb-3"),
                        html.Div(id="workflow-live-agents"),
                    ],
                ),
                dbc.Tab(
                    label="Historical Analysis",
                    children=[
                        html.Div(
                            [
                                html.Label("Select Analysis Run:", className="mt-3"),
                                dcc.Dropdown(
                                    id="workflow-history-selector",
                                    options=[],
                                    placeholder="Select a workflow run...",
                                ),
                            ],
                            className="mb-4",
                        ),
                        html.Div(id="workflow-metrics-summary"),
                        dcc.Graph(id="workflow-gantt-chart"),
                        dcc.Graph(id="workflow-pipeline-waterfall"),
                        dcc.Graph(id="workflow-agent-breakdown"),
                        html.Div(id="workflow-llm-calls-table"),
                    ],
                ),
            ]
        ),
    ]


def register_callbacks(app: Dash) -> None:  # noqa: C901, PLR0915
    """Register Workflow tab callbacks.

    Args:
        app: Dash app instance
    """
    client = app.api_client  # type: ignore[attr-defined]

    @app.callback(
        [Output("workflow-live-status", "children"), Output("workflow-live-agents", "children")],
        Input("workflow-interval", "n_intervals"),
        State("tabs", "active_tab"),
    )
    def update_live_view(n_intervals: int, active_tab: str) -> tuple[list, list]:  # noqa: ARG001
        """Update live execution view.

        Args:
            n_intervals: Interval counter
            active_tab: Active tab ID

        Returns:
            Tuple of (status HTML, agents HTML)
        """
        from dash.exceptions import PreventUpdate

        if active_tab != "workflow":
            raise PreventUpdate

        try:
            events = client.get_events(limit=10).events

            # Find latest ANALYSIS_START and ANALYSIS_COMPLETE
            analysis_start = None
            analysis_complete = None

            for event in events:
                if event["event_type"] == "ANALYSIS_START" and not analysis_start:
                    analysis_start = event
                if event["event_type"] == "ANALYSIS_COMPLETE" and not analysis_complete:
                    analysis_complete = event

            # Determine state
            if analysis_start and (
                not analysis_complete or analysis_start["timestamp"] > analysis_complete["timestamp"]
            ):
                # Active analysis
                symbol = analysis_start["data"].get("symbol", "Unknown")
                status_html = [
                    dbc.Alert(
                        [
                            dbc.Spinner(size="sm", spinner_class_name="me-2"),
                            html.Span(f"Analyzing {symbol}..."),
                        ],
                        color="info",
                    )
                ]

                # Build agent grid (placeholder for now)
                agents_html = [
                    html.H5("Active Agents", className="mb-3"),
                    dbc.Alert("Agent status tracking coming soon", color="secondary"),
                ]

            else:
                # Idle
                status_html = [dbc.Alert("No analysis running", color="secondary", className="text-center")]
                agents_html = []

            return status_html, agents_html

        except Exception as e:
            logger.error(f"Live view update failed: {e}")
            return [dbc.Alert(f"Error: {e}", color="danger")], []

    @app.callback(
        Output("workflow-history-selector", "options"),
        Input("workflow-interval", "n_intervals"),
        State("tabs", "active_tab"),
    )
    def populate_dropdown(n_intervals: int, active_tab: str) -> list[dict]:  # noqa: ARG001
        """Populate dropdown with workflow history.

        Args:
            n_intervals: Interval counter
            active_tab: Active tab ID

        Returns:
            Dropdown options
        """
        from dash.exceptions import PreventUpdate

        if active_tab != "workflow":
            raise PreventUpdate

        try:
            metrics_resp = client.get_execution_metrics(limit=50)

            if not metrics_resp.metrics:
                return []

            options = []
            for metric in metrics_resp.metrics:
                ts = datetime.fromisoformat(metric["timestamp"])
                time_str = ts.strftime("%Y-%m-%d %H:%M:%S")
                latency = metric["total_latency_ms"] / 1000
                label = f"{metric['symbol']} @ {time_str} ({latency:.1f}s)"
                options.append({"label": label, "value": metric["workflow_id"]})

            return options

        except Exception as e:
            logger.error(f"Dropdown population failed: {e}")
            return []

    @app.callback(
        [
            Output("workflow-metrics-summary", "children"),
            Output("workflow-gantt-chart", "figure"),
            Output("workflow-pipeline-waterfall", "figure"),
            Output("workflow-agent-breakdown", "figure"),
            Output("workflow-llm-calls-table", "children"),
        ],
        Input("workflow-history-selector", "value"),
    )
    def render_historical_detail(workflow_id: str | None) -> tuple:
        """Render historical workflow detail.

        Args:
            workflow_id: Selected workflow ID

        Returns:
            Tuple of (summary cards, gantt, waterfall, breakdown, table)
        """
        from dash.exceptions import PreventUpdate

        if not workflow_id:
            raise PreventUpdate

        try:
            metrics = client.get_execution_metric_detail(workflow_id)

            # Summary cards
            total_time = metrics["total_latency_ms"] / 1000
            llm_calls = len(metrics["llm_calls"])
            total_input = metrics["total_input_tokens"]
            total_output = metrics["total_output_tokens"]
            total_cost = metrics["total_estimated_cost_usd"]

            summary = dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H6("Total Time", className="text-muted"),
                                    html.H3(f"{total_time:.1f}s"),
                                ]
                            ),
                            className="text-center",
                        ),
                        width=3,
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H6("LLM Calls", className="text-muted"),
                                    html.H3(str(llm_calls)),
                                ]
                            ),
                            className="text-center",
                        ),
                        width=3,
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H6("Tokens", className="text-muted"),
                                    html.H3(f"{(total_input + total_output) / 1000:.1f}K"),
                                    html.Small(f"↓{total_input / 1000:.1f}K ↑{total_output / 1000:.1f}K"),
                                ]
                            ),
                            className="text-center",
                        ),
                        width=3,
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H6("Cost", className="text-muted"),
                                    html.H3(f"${total_cost:.2f}"),
                                ]
                            ),
                            className="text-center",
                        ),
                        width=3,
                    ),
                ],
                className="mb-4 mt-3",
            )

            # Gantt chart
            gantt_fig = _build_gantt_chart(metrics)

            # Waterfall chart
            waterfall_fig = _build_waterfall_chart(metrics)

            # Agent breakdown
            breakdown_fig = _build_agent_breakdown(metrics)

            # LLM calls table
            llm_table = _build_llm_calls_table(metrics)

            return summary, gantt_fig, waterfall_fig, breakdown_fig, llm_table

        except Exception as e:
            logger.error(f"Historical detail render failed: {e}")
            empty_fig = go.Figure()
            error_alert = dbc.Alert(f"Error: {e}", color="danger")
            return error_alert, empty_fig, empty_fig, empty_fig, error_alert


def _build_gantt_chart(metrics: dict) -> go.Figure:
    """Build Gantt chart from metrics.

    Args:
        metrics: Workflow execution metrics

    Returns:
        Plotly figure
    """
    # Extract stages and agents
    stages = metrics.get("pipeline_stages", [])
    agents = metrics.get("agent_timings", [])

    if not stages and not agents:
        fig = go.Figure()
        fig.add_annotation(text="No timing data available", showarrow=False, font={"size": 16})
        return fig

    # Build tasks for Gantt
    tasks = []
    start_time = datetime.fromisoformat(metrics["timestamp"])

    # Add pipeline stages
    cumulative = 0.0
    for stage in stages:
        tasks.append(
            {
                "Task": f"Stage: {stage['stage']}",
                "Start": start_time.timestamp() + cumulative / 1000,
                "Finish": start_time.timestamp() + (cumulative + stage["latency_ms"]) / 1000,
                "Type": "Stage",
            }
        )
        cumulative += stage["latency_ms"]

    # Add agent timings (assumes sequential for now)
    cumulative = 0.0
    for agent in agents:
        tasks.append(
            {
                "Task": f"Agent: {agent['agent_name']}",
                "Start": start_time.timestamp() + cumulative / 1000,
                "Finish": start_time.timestamp() + (cumulative + agent["latency_ms"]) / 1000,
                "Type": "Agent",
            }
        )
        cumulative += agent["latency_ms"]

    if not tasks:
        fig = go.Figure()
        fig.add_annotation(text="No tasks to display", showarrow=False, font={"size": 16})
        return fig

    # Convert to DataFrame-like structure
    df_tasks = []
    for task in tasks:
        df_tasks.append(
            {
                "Task": task["Task"],
                "Start": datetime.fromtimestamp(task["Start"], tz=UTC),
                "Finish": datetime.fromtimestamp(task["Finish"], tz=UTC),
                "Type": task["Type"],
            }
        )

    # Create timeline
    fig = px.timeline(
        df_tasks,
        x_start="Start",
        x_end="Finish",
        y="Task",
        color="Type",
        title="Workflow Execution Timeline",
        color_discrete_map={"Stage": "#3b82f6", "Agent": "#22c55e"},
    )

    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=400, margin={"l": 200, "r": 40, "t": 60, "b": 40})

    return fig


def _build_waterfall_chart(metrics: dict) -> go.Figure:
    """Build waterfall chart from pipeline stages.

    Args:
        metrics: Workflow execution metrics

    Returns:
        Plotly figure
    """
    stages = metrics.get("pipeline_stages", [])

    if not stages:
        fig = go.Figure()
        fig.add_annotation(text="No pipeline stage data", showarrow=False, font={"size": 16})
        return fig

    stage_names = [s["stage"] for s in stages]
    stage_times = [s["latency_ms"] / 1000 for s in stages]

    fig = go.Figure(
        go.Waterfall(
            name="Pipeline",
            orientation="v",
            x=stage_names,
            y=stage_times,
            textposition="outside",
            text=[f"{t:.2f}s" for t in stage_times],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        )
    )

    fig.update_layout(
        title="Pipeline Stage Waterfall",
        xaxis_title="Stage",
        yaxis_title="Time (seconds)",
        height=400,
        margin={"l": 40, "r": 40, "t": 60, "b": 80},
    )

    return fig


def _build_agent_breakdown(metrics: dict) -> go.Figure:
    """Build agent breakdown bar chart.

    Args:
        metrics: Workflow execution metrics

    Returns:
        Plotly figure
    """
    agents = metrics.get("agent_timings", [])

    if not agents:
        fig = go.Figure()
        fig.add_annotation(text="No agent timing data", showarrow=False, font={"size": 16})
        return fig

    # Sort by latency descending
    agents_sorted = sorted(agents, key=lambda x: x["latency_ms"], reverse=True)

    agent_names = [a["agent_name"] for a in agents_sorted]
    latencies = [a["latency_ms"] / 1000 for a in agents_sorted]
    llm_calls = [a["llm_calls"] for a in agents_sorted]

    # Color intensity by LLM call count
    max_calls = max(llm_calls) if llm_calls else 1
    colors = [f"rgba(34, 197, 94, {0.3 + 0.7 * (c / max_calls)})" for c in llm_calls]

    fig = go.Figure(
        go.Bar(
            y=agent_names,
            x=latencies,
            orientation="h",
            marker={"color": colors},
            text=[f"{lat:.2f}s ({c} calls)" for lat, c in zip(latencies, llm_calls, strict=True)],
            textposition="outside",
        )
    )

    fig.update_layout(
        title="Agent Execution Time Breakdown",
        xaxis_title="Time (seconds)",
        yaxis_title="Agent",
        height=max(400, len(agents) * 40),
        margin={"l": 200, "r": 40, "t": 60, "b": 40},
    )

    return fig


def _build_llm_calls_table(metrics: dict) -> list:
    """Build LLM calls table.

    Args:
        metrics: Workflow execution metrics

    Returns:
        Table HTML components
    """
    llm_calls = metrics.get("llm_calls", [])

    if not llm_calls:
        return [dbc.Alert("No LLM call data available", color="info", className="mt-3")]

    rows = []
    for call in llm_calls:
        ts = datetime.fromisoformat(call["timestamp"])
        success_badge = "✓" if call["success"] else "✗"
        badge_color = "success" if call["success"] else "danger"

        rows.append(
            html.Tr(
                [
                    html.Td(ts.strftime("%H:%M:%S")),
                    html.Td(call["agent_name"]),
                    html.Td(call["method"]),
                    html.Td(call["model"]),
                    html.Td(f"{call['latency_ms'] / 1000:.2f}s"),
                    html.Td(f"{call['input_tokens'] or 0}"),
                    html.Td(f"{call['output_tokens'] or 0}"),
                    html.Td(f"${call['estimated_cost_usd'] or 0:.4f}"),
                    html.Td(html.Span(success_badge, className=f"badge bg-{badge_color}")),
                ]
            )
        )

    table = dbc.Table(
        [
            html.Thead(
                html.Tr(
                    [
                        html.Th("Time"),
                        html.Th("Agent"),
                        html.Th("Method"),
                        html.Th("Model"),
                        html.Th("Latency"),
                        html.Th("In Tokens"),
                        html.Th("Out Tokens"),
                        html.Th("Cost"),
                        html.Th("Status"),
                    ]
                )
            ),
            html.Tbody(rows),
        ],
        bordered=True,
        hover=True,
        striped=True,
        size="sm",
        className="mt-3",
    )

    return [
        html.H5("LLM Call Details", className="mt-4"),
        table,
    ]
