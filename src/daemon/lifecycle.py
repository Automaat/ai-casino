"""Daemon lifecycle management - startup, shutdown, signal handling."""

from __future__ import annotations

import asyncio
import contextlib
import signal

import uvicorn
from loguru import logger
from rich.console import Console

from src.daemon.factory import DaemonComponents

console = Console()


class DaemonLifecycle:
    """Manage daemon lifecycle: startup, shutdown, signal handling."""

    def __init__(self, components: DaemonComponents) -> None:
        """Initialize lifecycle manager.

        Args:
            components: Daemon components
        """
        self.components = components
        self.running = False
        self._api_server: uvicorn.Server | None = None
        self._api_task: asyncio.Task | None = None

    async def startup(self) -> None:
        """Execute startup sequence: DB migrations, signal handlers, API server."""
        self.running = True

        # Initialize database (run migrations)
        try:
            if self.components.config.database.enable_persistence:
                database_engine = self.components.container.database_engine()
                await database_engine.ensure_migrated()
                logger.info("Database migrations applied successfully")
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            if self.components.config.database.enable_persistence:
                raise  # Fail fast if persistence enabled but DB unavailable

        # Set up signal handlers
        def shutdown_handler(sig: int, _frame: object) -> None:
            logger.info(f"Received signal {sig}, shutting down...")
            self.request_shutdown()

        signal.signal(signal.SIGINT, shutdown_handler)
        signal.signal(signal.SIGTERM, shutdown_handler)

        # Print startup info
        console.print("\n[bold green]Daemon started[/bold green]")
        console.print(f"Watchlist: {', '.join(self.components.config.watchlist)}")
        console.print(f"Interval: {self.components.config.interval_minutes} minutes")
        console.print(f"Market hours only: {self.components.config.market_hours_only}")
        console.print(f"Auto trade: {self.components.config.auto_trade}")
        console.print()

        # Start API server if enabled
        if self.components.config.api.enabled:
            self._start_api_server()

    async def shutdown(self) -> None:
        """Execute shutdown: stop API server, save state."""
        # Stop API server before saving state
        await self._stop_api_server()

        # Save final state
        self.components.state.save(self.components.config.state.state_file)

        console.print("\n[bold yellow]Daemon stopped[/bold yellow]")
        logger.info("Daemon shutdown complete")

    def request_shutdown(self) -> None:
        """Request graceful shutdown (called by signal handlers)."""
        self.running = False
        if self._api_server:
            self._api_server.should_exit = True

    def _start_api_server(self) -> None:
        """Start embedded API server as background task."""
        try:
            from src.daemon.api import create_api_app

            app = create_api_app(self.components)
            config = uvicorn.Config(
                app,
                host=self.components.config.api.host,
                port=self.components.config.api.port,
                log_level="info",
                access_log=False,
            )
            self._api_server = uvicorn.Server(config)

            def _log_api_task_result(t: asyncio.Task) -> None:
                if t.cancelled():
                    # Task was cancelled as part of shutdown; no error to log.
                    return
                exc = t.exception()
                if exc is not None:
                    # Log real server crashes with traceback.
                    logger.opt(exception=exc).error("API server crashed")

            self._api_task = asyncio.create_task(self._api_server.serve())
            self._api_task.add_done_callback(_log_api_task_result)

            logger.info(
                f"API server started at http://{self.components.config.api.host}:"
                f"{self.components.config.api.port}"
            )
            console.print(
                f"[bold cyan]API server: http://{self.components.config.api.host}:"
                f"{self.components.config.api.port}[/bold cyan]"
            )
        except Exception as e:
            logger.error(f"Failed to start API server: {e}")
            self._api_server = None
            self._api_task = None

    async def _stop_api_server(self) -> None:
        """Stop embedded API server gracefully."""
        if self._api_server and self._api_task:
            try:
                logger.info("Stopping API server...")
                self._api_server.should_exit = True
                await asyncio.wait_for(self._api_task, timeout=5.0)
                logger.info("API server stopped")
            except TimeoutError:
                logger.warning("API server shutdown timed out, cancelling task")
                self._api_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._api_task
            except Exception as e:
                logger.error(f"Error stopping API server: {e}")
