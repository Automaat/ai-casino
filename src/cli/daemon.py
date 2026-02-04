"""Daemon subcommand for autonomous trading mode."""

import asyncio
import os
import sys
from pathlib import Path
from typing import Annotated

import typer
from loguru import logger
from rich.console import Console

from src.daemon.config import DaemonConfig
from src.daemon.runner import DaemonRunner

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
