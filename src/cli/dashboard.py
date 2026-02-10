"""Dashboard subcommand for daemon monitoring."""

import os
import sys

import typer
from loguru import logger
from rich.console import Console

from src.dashboard.api_client import DaemonAPIClient
from src.dashboard.app import create_dash_app
from src.dashboard.config import DashboardConfig

console = Console()


def _check_daemon_health(api_url: str) -> DaemonAPIClient:
    """Check daemon health and warn if unreachable."""
    client = DaemonAPIClient(api_url)
    if not client.is_healthy():
        console.print("[bold yellow]Warning:[/bold yellow] Daemon API is not reachable")
        console.print()

        # Skip interactive prompt if not in TTY (Docker, CI, etc.)
        if not sys.stdin.isatty():
            console.print("Running in non-interactive mode, proceeding anyway...")
            console.print("Dashboard will show errors until daemon is healthy")
            console.print()
        else:
            console.print("Make sure the daemon is running:")
            console.print("  [cyan]mise daemon --config daemon.yaml[/cyan]")
            console.print("  (copy docs/daemon.yaml.example to daemon.yaml)")
            console.print()
            console.print("Options:")
            console.print("  1. Start the daemon")
            console.print("  2. Check daemon configuration")
            console.print("  3. Proceed anyway (dashboard will show errors)")
            console.print()

            choice = console.input("[bold]Continue? (y/N): [/bold]")
            if choice.lower() not in ["y", "yes"]:
                console.print("[yellow]Aborted[/yellow]")
                raise typer.Exit(0)

    return client


def dashboard(
    api_url: str | None = None,
    port: int = 8050,
    debug: bool = False,
) -> None:
    """Launch Dash dashboard for daemon monitoring.

    Args:
        api_url: Daemon API URL. If not provided, DashboardConfig uses $DAEMON_API_URL or defaults to http://localhost:8484
        port: Dashboard server port (default: 8050)
        debug: Enable debug mode (default: False)
    """
    from dotenv import load_dotenv

    load_dotenv()

    # Configure logging
    logger.remove()
    log_level = "DEBUG" if debug else os.getenv("LOG_LEVEL", "INFO")
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=log_level,
    )

    try:
        # Create config - let DashboardConfig handle api_url default
        # Normalize empty/whitespace-only strings to None so DashboardConfig applies env/default
        normalized_api_url = None
        if api_url is not None:
            stripped_api_url = api_url.strip()
            if stripped_api_url:
                normalized_api_url = stripped_api_url

        if normalized_api_url is not None:
            config = DashboardConfig(api_url=normalized_api_url, port=port)
        else:
            config = DashboardConfig(port=port)

        console.print("[bold cyan]AI Casino Dashboard[/bold cyan]")
        console.print(f"API URL: {config.api_url}")
        console.print(f"Refresh interval: {config.refresh_interval}ms")
        console.print()

        # Check daemon health
        client = _check_daemon_health(config.api_url)
        client.close()

        # Create and run app
        app = create_dash_app(config, debug=debug)

        console.print()
        console.print(f"[bold green]Starting dashboard at http://{config.host}:{config.port}[/bold green]")
        console.print("[dim]Press Ctrl+C to stop[/dim]")
        console.print()

        try:
            app.run(host=config.host, port=config.port, debug=debug)
        finally:
            app.api_client.close()

    except KeyboardInterrupt:
        console.print("\n[bold yellow]Dashboard stopped[/bold yellow]")
    except Exception as e:
        console.print(f"\n[bold red]Dashboard error:[/bold red] {e}")
        logger.exception("Dashboard failed")
        raise typer.Exit(1) from e
