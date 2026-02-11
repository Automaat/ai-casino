"""Chat subcommand for TUI mode."""

import contextlib
import logging
import sys
from pathlib import Path

import typer
from loguru import logger

from src.utils.logging import sanitize_log_record


def chat() -> None:
    """Launch interactive TUI chat interface."""
    from dotenv import load_dotenv

    load_dotenv()

    # Configure torch environment (env vars only, no imports - defer torch import until analysis)
    from src.models.torch_config import configure_torch_env

    configure_torch_env()

    # Configure logging - suppress all stderr output
    logger.remove()

    # File logging - always capture INFO+ for debugging
    log_file = Path("~/.ai-casino/tui.log").expanduser()
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        level="INFO",
        rotation="10 MB",
        retention="3 days",
        filter=sanitize_log_record,
    )

    # Suppress all standard library logging to stderr
    logging.basicConfig(
        level=logging.ERROR,
        handlers=[logging.FileHandler(log_file)],
        format="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s",
    )

    # Redirect stderr to log file to catch library output
    stderr_backup = sys.stderr

    with contextlib.ExitStack() as stack:
        log_file_handle = stack.enter_context(log_file.open("a", encoding="utf-8"))
        sys.stderr = log_file_handle

        try:
            # NOTE: nest_asyncio removed - breaks Python 3.14 + anyio/httpcore
            # Textual handles its own event loop

            from src.tui.app import TradingChatApp

            app = TradingChatApp()
            app.run()
        except Exception as e:
            typer.echo(f"TUI error: {e}")
            logger.exception("TUI failed")
            raise typer.Exit(1) from e
        finally:
            # Restore stderr before ExitStack closes file
            sys.stderr = stderr_backup
