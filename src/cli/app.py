"""Typer CLI application entry point."""

import typer

from src.cli.analyze import analyze
from src.cli.chat import chat
from src.cli.daemon import daemon
from src.cli.optimize import optimize

app = typer.Typer(
    name="casino",
    help="AI Casino - Multi-agent stock trading system",
    no_args_is_help=True,
)

app.command()(analyze)
app.command()(optimize)
app.command()(daemon)
app.command()(chat)


def main() -> None:
    """CLI entry point."""
    app()


if __name__ == "__main__":
    main()
