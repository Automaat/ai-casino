"""Daemon subcommand for autonomous trading mode."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from loguru import logger
from rich.console import Console

from src.cache.historical import HistoricalCache
from src.daemon.config import DaemonConfig
from src.daemon.event_watcher import EventWatcher
from src.daemon.runner import DaemonRunner
from src.daemon.trump_watcher import TrumpWatcher
from src.daemon.watchers import AnomalyWatcher, NewsWatcher, SocialWatcher
from src.di.container import create_container
from src.utils.logging import sanitize_log_record

if TYPE_CHECKING:
    from src.di.container import AppContainer

console = Console()


def daemon(
    config: Annotated[
        Path | None, typer.Option("--config", "-c", help="Path to daemon config file (TOML)")
    ] = None,
) -> None:
    """Run autonomous trading daemon (24/7 scheduled analysis)."""
    # Load config first to get log level
    try:
        if config is not None:
            if not config.exists():
                console.print(f"[bold red]Error:[/bold red] Config file not found: {config}")
                raise typer.Exit(1)
            daemon_config = DaemonConfig.from_yaml(config)
        else:
            daemon_config = DaemonConfig()
    except Exception as e:
        console.print(f"[bold red]Config error:[/bold red] {e}")
        raise typer.Exit(1) from e

    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=daemon_config.logging.log_level,
        filter=sanitize_log_record,
    )

    try:
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
    # Load default config for log level
    daemon_config = DaemonConfig()

    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=daemon_config.logging.log_level,
        filter=sanitize_log_record,
    )

    try:
        container = create_container()
        watcher = TrumpWatcher(poll_interval=poll_interval, max_analyses=max_analyses, container=container)
        asyncio.run(watcher.run())
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Trump watcher interrupted[/bold yellow]")
    except Exception as e:
        console.print(f"\n[bold red]Trump watcher error:[/bold red] {e}")
        logger.exception("Trump watcher failed")
        raise typer.Exit(1) from e


def _load_daemon_config(config: Path | None) -> DaemonConfig:
    """Load daemon config from file or defaults."""
    if config is not None:
        if not config.exists():
            console.print(f"[bold red]Error:[/bold red] Config file not found: {config}")
            raise typer.Exit(1)
        return DaemonConfig.from_yaml(config)
    return DaemonConfig()


def _init_event_watchers(
    daemon_config: DaemonConfig, historical_cache: HistoricalCache, container: AppContainer
) -> list[EventWatcher]:
    """Initialize enabled event watchers.

    Args:
        daemon_config: Daemon configuration
        historical_cache: Shared historical data cache
        container: DI container for resolving dependencies
    """
    watchers: list[EventWatcher] = []

    if daemon_config.news_watcher.enabled:
        watchers.append(
            NewsWatcher(
                historical_cache=historical_cache,
                poll_interval=daemon_config.news_watcher.poll_interval_minutes * 60,
                relevance_threshold=daemon_config.news_watcher.relevance_threshold,
                cooldown_minutes=daemon_config.news_watcher.cooldown_minutes,
                breaking_threshold_minutes=daemon_config.news_watcher.breaking_threshold_minutes,
                max_concurrent_analyses=daemon_config.news_watcher.max_concurrent_analyses,
            )
        )
        console.print("[green]✓[/green] NewsWatcher enabled")

    if daemon_config.social_watcher.enabled:
        watchers.append(
            SocialWatcher(
                historical_cache=historical_cache,
                poll_interval=daemon_config.social_watcher.poll_interval_minutes * 60,
                relevance_threshold=daemon_config.social_watcher.relevance_threshold,
                cooldown_minutes=daemon_config.social_watcher.cooldown_minutes,
                volume_spike_threshold=daemon_config.social_watcher.volume_spike_threshold,
                viral_score_threshold=daemon_config.social_watcher.viral_score_threshold,
                viral_upvote_ratio=daemon_config.social_watcher.viral_upvote_ratio,
                subreddits=daemon_config.social_watcher.subreddits,
                max_concurrent_analyses=daemon_config.social_watcher.max_concurrent_analyses,
            )
        )
        console.print("[green]✓[/green] SocialWatcher enabled")

    if daemon_config.anomaly_watcher.enabled:
        watchers.append(
            AnomalyWatcher(
                historical_cache=historical_cache,
                market_fetcher=container.market_fetcher(),
                poll_interval=daemon_config.anomaly_watcher.poll_interval_minutes * 60,
                relevance_threshold=daemon_config.anomaly_watcher.relevance_threshold,
                cooldown_minutes=daemon_config.anomaly_watcher.cooldown_minutes,
                volume_spike_multiplier=daemon_config.anomaly_watcher.volume_spike_multiplier,
                price_move_threshold_pct=daemon_config.anomaly_watcher.price_move_threshold_pct,
                gap_threshold_pct=daemon_config.anomaly_watcher.gap_threshold_pct,
                watchlist=daemon_config.watchlist,
                max_symbols_per_cycle=daemon_config.anomaly_watcher.max_symbols_per_cycle,
                max_concurrent_analyses=daemon_config.anomaly_watcher.max_concurrent_analyses,
            )
        )
        console.print("[green]✓[/green] AnomalyWatcher enabled")

    return watchers


def _validate_watchers_config(daemon_config: DaemonConfig) -> None:
    """Validate watcher configuration and exit if invalid.

    Args:
        daemon_config: Daemon configuration to validate

    Raises:
        typer.Exit: If configuration is invalid
    """
    # Check only implemented watchers
    implemented_enabled = (
        daemon_config.news_watcher.enabled
        or daemon_config.social_watcher.enabled
        or daemon_config.anomaly_watcher.enabled
    )

    # Check unimplemented watchers
    unimplemented_enabled = daemon_config.filings_watcher.enabled

    if unimplemented_enabled:
        console.print("[bold red]Error:[/bold red] Unsupported event watchers enabled")
        console.print("The following watchers are not yet implemented:")
        if daemon_config.filings_watcher.enabled:
            console.print("  - filings_watcher")
        console.print("\nAvailable watchers: news_watcher, social_watcher, anomaly_watcher")
        raise typer.Exit(1)

    if not implemented_enabled:
        console.print("[bold red]Error:[/bold red] No event watchers enabled in config")
        console.print("Enable at least one watcher in daemon.yaml:")
        console.print("  [daemon.news_watcher]")
        console.print("  enabled = true")
        raise typer.Exit(1)


async def _run_watchers(watchers: list[EventWatcher]) -> None:
    """Run all watchers with graceful shutdown.

    Args:
        watchers: List of watcher instances to run
    """
    import signal

    def shutdown_handler(sig: int, _frame: object) -> None:
        logger.info(f"Received signal {sig}, shutting down watchers...")
        for w in watchers:
            w.running = False

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    # Wrap watcher tasks to handle exceptions
    async def safe_run_watcher(watcher: object) -> BaseException | None:
        try:
            await watcher.run()  # type: ignore[attr-defined]
            return None
        except BaseException as e:
            # Re-raise control-flow exceptions so TaskGroup can cancel siblings promptly
            if isinstance(e, (asyncio.CancelledError, KeyboardInterrupt)):
                raise
            return e

    # Run watchers in parallel using TaskGroup
    async with asyncio.TaskGroup() as tg:
        task_results = [tg.create_task(safe_run_watcher(w)) for w in watchers]

    # Extract results and handle exceptions
    results = [task.result() for task in task_results]

    # Log failures and re-raise cancellation/shutdown exceptions
    for i, result in enumerate(results):
        if isinstance(result, BaseException):
            if isinstance(result, (asyncio.CancelledError, KeyboardInterrupt)):
                raise result
            logger.error(f"Watcher {i} failed: {result}")


def events_daemon(
    config: Annotated[
        Path | None, typer.Option("--config", "-c", help="Path to daemon config file (TOML)")
    ] = None,
) -> None:
    """Run event-driven analysis daemon.

    Monitors real-time events (news, social, filings, anomalies) and triggers
    immediate trading analysis for high-relevance signals.
    """
    # Load config first to get log level
    daemon_config = _load_daemon_config(config)

    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=daemon_config.logging.log_level,
        filter=sanitize_log_record,
    )

    try:
        _validate_watchers_config(daemon_config)

        container = create_container()
        container.daemon_config.override(daemon_config)
        historical_cache = container.historical_cache()
        watchers = _init_event_watchers(daemon_config, historical_cache, container)

        console.print()
        console.print(f"[bold green]Starting {len(watchers)} event watcher(s)...[/bold green]")
        asyncio.run(_run_watchers(watchers))

    except KeyboardInterrupt:
        console.print("\n[bold yellow]Event daemon interrupted[/bold yellow]")
    except Exception as e:
        console.print(f"\n[bold red]Event daemon error:[/bold red] {e}")
        logger.exception("Event daemon failed")
        raise typer.Exit(1) from e
