"""FastAPI application factory for daemon monitoring API."""

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.daemon.api.routers import (
    config_router,
    cost_analytics_router,
    execution_router,
    health_router,
    portfolio_router,
    signal_analytics_router,
    state_router,
    supervisor_router,
    trading_router,
    validation_router,
    websocket_router,
)

if TYPE_CHECKING:
    from src.daemon.factory import DaemonComponents


def create_api_app(components: DaemonComponents) -> FastAPI:
    """Create FastAPI app with components reference.

    Args:
        components: DaemonComponents instance

    Returns:
        FastAPI app
    """
    app = FastAPI(
        title="AI Casino Daemon API",
        description="Read-only monitoring API for trading daemon",
        version="0.1.0",
    )

    # Store components and start time in app state
    app.state.components = components
    app.state.start_time = datetime.now(UTC)

    # CORS middleware - configured to allow WebSocket connections
    # Note: Starlette CORSMiddleware has issues with WebSocket when allow_credentials=True
    # Using allow_credentials=False and validating origin manually in WebSocket handler
    app.add_middleware(
        CORSMiddleware,
        allow_origins=components.config.api.cors_origins,
        allow_credentials=False,  # Must be False for WebSocket to work
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(health_router)
    app.include_router(config_router)
    app.include_router(state_router)
    app.include_router(trading_router)
    app.include_router(portfolio_router)
    app.include_router(execution_router)
    app.include_router(supervisor_router)
    app.include_router(validation_router)
    app.include_router(cost_analytics_router)
    app.include_router(signal_analytics_router)
    app.include_router(websocket_router)

    logger.info("FastAPI app created")
    return app
