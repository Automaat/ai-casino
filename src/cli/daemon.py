"""Daemon subcommand for autonomous trading mode."""

import asyncio
import os
import sys
from pathlib import Path
from typing import Annotated

import typer
from loguru import logger
from rich.console import Console

from src.cache.historical import HistoricalCache
from src.daemon.config import DaemonConfig
from src.daemon.runner import DaemonRunner
from src.daemon.trump_watcher import TrumpWatcher
from src.daemon.watchers.news_watcher import NewsWatcher

console = Console()


def daemon(
    config: Annotated[
        Path | None, typer.Option("--config", "-c", help="Path to daemon config file (TOML)")
    ] = None,
) -> None:
    """Run autonomous trading daemon (24/7 scheduled analysis)."""
    from dotenv import load_dotenv

    load_dotenv()

    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=os.getenv("LOG_LEVEL", "INFO"),
    )

    try:
        if config is not None:
            if not config.exists():
                console.print(f"[bold red]Error:[/bold red] Config file not found: {config}")
                raise typer.Exit(1)  # noqa: TRY301
            daemon_config = DaemonConfig.from_toml(config)
        else:
            daemon_config = DaemonConfig()

        runner = DaemonRunner(daemon_config)
        asyncio.run(runner.run())
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Daemon interrupted[/bold yellow]")
    except Exception as e:
        console.print(f"\n[bold red]Daemon error:[/bold red] {e}")
        logger.exception("Daemon failed")
        raise typer.Exit(1) from e


def trump_daemon(
    poll_interval: Annotated[int, typer.Option("--interval", "-i", help="Poll interval in seconds")] = 60,
    max_analyses: Annotated[
        int, typer.Option("--max-analyses", "-m", help="Max stocks to analyze per signal")
    ] = 5,
) -> None:
    """Run Trump social media watcher daemon.

    Monitors Trump's Truth Social posts and triggers stock analysis
    when market-relevant posts are detected.
    """
    from dotenv import load_dotenv

    load_dotenv()

    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=os.getenv("LOG_LEVEL", "INFO"),
    )

    try:
        watcher = TrumpWatcher(poll_interval=poll_interval, max_analyses=max_analyses)
        asyncio.run(watcher.run())
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Trump watcher interrupted[/bold yellow]")
    except Exception as e:
        console.print(f"\n[bold red]Trump watcher error:[/bold red] {e}")
        logger.exception("Trump watcher failed")
        raise typer.Exit(1) from e


def events_daemon(
    config: Annotated[
        Path | None, typer.Option("--config", "-c", help="Path to daemon config file (TOML)")
    ] = None,
) -> None:
    """Run event-driven analysis daemon.

    Monitors real-time events (news, social, filings, anomalies) and triggers
    immediate trading analysis for high-relevance signals.
    """
    from dotenv import load_dotenv

    load_dotenv()

    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=os.getenv("LOG_LEVEL", "INFO"),
    )

    try:
        # Load config
        if config is not None:
            if not config.exists():
                console.print(f"[bold red]Error:[/bold red] Config file not found: {config}")
                raise typer.Exit(1)  # noqa: TRY301
            daemon_config = DaemonConfig.from_toml(config)
        else:
            daemon_config = DaemonConfig()

        # Check if any watcher enabled
        any_enabled = (
            daemon_config.news_watcher.enabled
            or daemon_config.social_watcher.enabled
            or daemon_config.filings_watcher.enabled
            or daemon_config.anomaly_watcher.enabled
        )

        if not any_enabled:
            console.print("[bold red]Error:[/bold red] No event watchers enabled in config")
            console.print("Enable at least one watcher in daemon.toml:")
            console.print("  [daemon.news_watcher]")
            console.print("  enabled = true")
            raise typer.Exit(1)  # noqa: TRY301

        # Initialize enabled watchers
        watchers = []
        historical_cache = HistoricalCache()

        if daemon_config.news_watcher.enabled:
            watchers.append(
                NewsWatcher(
                    historical_cache=historical_cache,
                    poll_interval=daemon_config.news_watcher.poll_interval_minutes * 60,
                    relevance_threshold=daemon_config.news_watcher.relevance_threshold,
                    cooldown_minutes=daemon_config.news_watcher.cooldown_minutes,
                    breaking_threshold_minutes=daemon_config.news_watcher.breaking_threshold_minutes,
                )
            )
            console.print("[green]✓[/green] NewsWatcher enabled")

        # TODO: Add other watchers (social, filings, anomaly) when implemented

        async def run_all() -> None:
            """Run all enabled watchers concurrently."""
            tasks = [w.run() for w in watchers]
            await asyncio.gather(*tasks)

        console.print()
        console.print(f"[bold green]Starting {len(watchers)} event watcher(s)...[/bold green]")
        asyncio.run(run_all())

    except KeyboardInterrupt:
        console.print("\n[bold yellow]Event daemon interrupted[/bold yellow]")
    except Exception as e:
        console.print(f"\n[bold red]Event daemon error:[/bold red] {e}")
        logger.exception("Event daemon failed")
        raise typer.Exit(1) from e
