"""Chat subcommand for TUI mode."""

import logging
import os
import sys
from pathlib import Path

import typer
from loguru import logger


def chat() -> None:
    """Launch interactive TUI chat interface."""
    from dotenv import load_dotenv

    load_dotenv()

    # Disable parallelism to prevent subprocess fd conflicts with Textual
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    # Suppress transformers/torch warnings
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

    # Force torch to use fork-safe settings
    import torch

    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(1)

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
    )

    # Suppress all standard library logging to stderr
    logging.basicConfig(
        level=logging.ERROR,
        handlers=[logging.FileHandler(log_file)],
        format="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s",
    )

    # Redirect stderr to log file to catch library output
    stderr_backup = sys.stderr
    sys.stderr = open(log_file, "a", encoding="utf-8")  # noqa: SIM115, PTH123

    try:
        # NOTE: nest_asyncio removed - breaks Python 3.14 + anyio/httpcore
        # Textual handles its own event loop

        from src.tui.app import TradingChatApp

        app = TradingChatApp()
        app.run()
    except Exception as e:
        # Restore stderr for error reporting
        sys.stderr.close()
        sys.stderr = stderr_backup
        typer.echo(f"TUI error: {e}")
        logger.exception("TUI failed")
        raise typer.Exit(1) from e
    finally:
        # Always restore stderr
        if sys.stderr != stderr_backup:
            sys.stderr.close()
            sys.stderr = stderr_backup
