"""Chat subcommand for TUI mode."""

import os
import sys

import typer
from loguru import logger


def chat() -> None:
    """Launch interactive TUI chat interface."""
    from dotenv import load_dotenv

    load_dotenv()

    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=os.getenv("LOG_LEVEL", "WARNING"),
    )

    try:
        from src.tui.app import TradingChatApp

        app = TradingChatApp()
        app.run()
    except Exception as e:
        typer.echo(f"TUI error: {e}")
        logger.exception("TUI failed")
        raise typer.Exit(1) from e
