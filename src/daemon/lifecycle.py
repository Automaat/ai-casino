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
        self._watcher_tasks: list[asyncio.Task] = []

    def __repr__(self) -> str:
        """Return string representation."""
        return f"DaemonLifecycle(running={self.running})"

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

        # Start event watchers if enabled
        self._start_watchers()

    async def shutdown(self) -> None:
        """Execute shutdown: stop watchers, stop API server, wait for background tasks, save state."""
        # Stop watchers first
        await self._stop_watchers()

        # Stop API server before saving state
        await self._stop_api_server()

        # Wait for position manager background tasks to complete
        if self.components.position_manager:
            try:
                await self.components.position_manager.wait_for_pending_tasks(timeout_seconds=5.0)
            except Exception as e:
                logger.warning(f"Error waiting for position persistence tasks: {e}")

        # Wait for other state manager background tasks to complete
        if self.components.state.discovery:
            try:
                await self.components.state.discovery.wait_for_pending_tasks(timeout_seconds=5.0)
            except Exception as e:
                logger.warning(f"Error waiting for discovery state persistence tasks: {e}")

        if self.components.state.snapshots:
            try:
                await self.components.state.snapshots.wait_for_pending_tasks(timeout_seconds=5.0)
            except Exception as e:
                logger.warning(f"Error waiting for snapshot state persistence tasks: {e}")

        if self.components.state.trading:
            try:
                await self.components.state.trading.wait_for_pending_tasks(timeout_seconds=5.0)
            except Exception as e:
                logger.warning(f"Error waiting for trading state persistence tasks: {e}")

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

    def _start_watchers(self) -> None:
        """Start event watchers as background tasks."""
        watchers = []

        if self.components.news_watcher:
            watchers.append(("NewsWatcher", self.components.news_watcher))

        if self.components.social_watcher:
            watchers.append(("SocialWatcher", self.components.social_watcher))

        if not watchers:
            return

        for name, watcher in watchers:

            def _log_watcher_task_result(t: asyncio.Task, watcher_name: str = name) -> None:
                if t.cancelled():
                    return
                exc = t.exception()
                if exc is not None:
                    logger.opt(exception=exc).error(f"{watcher_name} crashed")

            task = asyncio.create_task(watcher.run())
            task.add_done_callback(_log_watcher_task_result)
            self._watcher_tasks.append(task)
            logger.info(f"{name} started as background task")

        console.print(f"[bold cyan]Event watchers: {len(watchers)} active[/bold cyan]")

    async def _stop_watchers(self) -> None:
        """Stop event watchers gracefully."""
        if not self._watcher_tasks:
            return

        try:
            logger.info("Stopping event watchers...")

            # Signal watchers to stop
            if self.components.news_watcher:
                self.components.news_watcher.running = False
            if self.components.social_watcher:
                self.components.social_watcher.running = False

            # Wait for tasks to complete (up to 5 seconds)
            done, pending = await asyncio.wait(self._watcher_tasks, timeout=5.0)
            if pending:
                logger.warning(f"{len(pending)} watcher tasks did not complete in time, cancelling them")
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
            logger.info("Event watchers stopped")
        except Exception as e:
            logger.error(f"Error stopping watchers: {e}")
        finally:
            # Clear references to watcher tasks after shutdown attempt
            self._watcher_tasks.clear()
