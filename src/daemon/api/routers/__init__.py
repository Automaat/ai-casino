"""API routers for daemon monitoring endpoints."""

from src.daemon.api.routers.config import router as config_router
from src.daemon.api.routers.execution import router as execution_router
from src.daemon.api.routers.health import router as health_router
from src.daemon.api.routers.portfolio import router as portfolio_router
from src.daemon.api.routers.state import router as state_router
from src.daemon.api.routers.trading import router as trading_router
from src.daemon.api.routers.websocket import router as websocket_router

__all__ = [
    "config_router",
    "execution_router",
    "health_router",
    "portfolio_router",
    "state_router",
    "trading_router",
    "websocket_router",
]
