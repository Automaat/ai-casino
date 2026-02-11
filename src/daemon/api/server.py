"""Threaded uvicorn server to isolate API from daemon event loop."""

from __future__ import annotations

import asyncio
import threading
from typing import TYPE_CHECKING

import uvicorn
from loguru import logger

if TYPE_CHECKING:
    from fastapi import FastAPI


class ThreadedApiServer:
    """Run uvicorn in a dedicated thread with its own event loop.

    Prevents daemon analysis cycles from starving API request handlers.
    """

    def __init__(self, app: FastAPI, host: str, port: int) -> None:
        """Initialize threaded server.

        Args:
            app: FastAPI application
            host: Bind address
            port: Bind port
        """
        config = uvicorn.Config(app, host=host, port=port, log_level="info", access_log=False)
        self._server = uvicorn.Server(config)
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start uvicorn in a background daemon thread."""
        # Prevent double-start: if the server thread is already running, do nothing.
        if self._thread is not None and self._thread.is_alive():
            logger.warning("API server thread already running; start() call ignored")
            return
        self._thread = threading.Thread(target=self._run, daemon=True, name="api-server")
        self._thread.start()
        logger.info("API server thread started")

    def _run(self) -> None:
        """Thread entry: create isolated event loop and serve."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._server.serve())
        except Exception as e:
            logger.opt(exception=True).error(f"API server crashed: {e}")
        finally:
            loop.close()

    def stop(self, timeout: float = 5.0) -> bool:
        """Signal shutdown and join the server thread.

        Args:
            timeout: Max seconds to wait for thread to finish

        Returns:
            True if thread exited cleanly, False if timeout or not running
        """
        self._server.should_exit = True
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning("API server thread did not exit within timeout")
                return False
            return True
        return False  # Thread was never started or already dead

    @property
    def should_exit(self) -> bool:
        """Proxy for uvicorn shutdown flag."""
        return self._server.should_exit

    @should_exit.setter
    def should_exit(self, value: bool) -> None:
        self._server.should_exit = value

    def __repr__(self) -> str:
        """String representation."""
        alive = self._thread.is_alive() if self._thread else False
        return f"ThreadedApiServer(alive={alive})"
