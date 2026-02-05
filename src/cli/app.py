"""Typer CLI application entry point."""

import sys

import typer

from src.cli.analyze import analyze
from src.cli.chat import chat as chat_cmd
from src.cli.daemon import daemon, trump_daemon
from src.cli.optimize import optimize

app = typer.Typer(
    name="aicasino",
    help="AI Casino - Multi-agent stock trading system",
    no_args_is_help=False,
)

app.command()(analyze)
app.command()(optimize)
app.command()(daemon)
app.command(name="trump-daemon")(trump_daemon)
app.command(name="chat")(chat_cmd)


def main() -> None:
    """CLI entry point - defaults to chat mode."""
    if len(sys.argv) == 1:
        chat_cmd()
    else:
        app()


if __name__ == "__main__":
    main()
