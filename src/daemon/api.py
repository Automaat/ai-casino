"""Embedded FastAPI server for daemon monitoring."""

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from src.daemon.runner import DaemonRunner


class HealthResponse(BaseModel):
    """Health endpoint response."""

    status: str = Field(description="Health status (healthy/degraded)")
    uptime_seconds: float = Field(description="Daemon uptime in seconds")
    running: bool = Field(description="Whether daemon is running")
    last_run: str | None = Field(description="Last analysis run timestamp")


class StateSummaryResponse(BaseModel):
    """State summary endpoint response."""

    total_analyses: int = Field(description="Total analyses performed")
    total_trades: int = Field(description="Total trades executed")
    error_count: int = Field(description="Total errors recorded")
    degradation_tier: str = Field(description="Current degradation tier")
    trading_mode: str = Field(description="Current trading mode (paper/live)")


class ConfigResponse(BaseModel):
    """Config endpoint response."""

    watchlist: list[str] = Field(description="Symbols being monitored")
    interval_minutes: int = Field(description="Analysis interval in minutes")
    market_hours_only: bool = Field(description="Whether restricted to market hours")
    auto_trade: bool = Field(description="Whether auto-trading is enabled")
    trading_mode: str = Field(description="Current trading mode (paper/live)")
    pre_market_enabled: bool = Field(description="Whether pre-market trading is enabled")


def create_api_app(runner: "DaemonRunner") -> FastAPI:
    """Create FastAPI app with runner reference.

    Args:
        runner: DaemonRunner instance

    Returns:
        FastAPI app
    """
    app = FastAPI(
        title="AI Casino Daemon API",
        description="Read-only monitoring API for trading daemon",
        version="0.1.0",
    )

    # Store runner and start time in app state
    app.state.runner = runner
    app.state.start_time = datetime.now(UTC)

    # CORS middleware - only allow dashboard origin
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:8050"],
        allow_credentials=True,
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """Get daemon health status."""
        runner: DaemonRunner = app.state.runner
        uptime = (datetime.now(UTC) - app.state.start_time).total_seconds()

        # Determine health status from degradation tier
        degradation_tier = "FULL"
        if runner.state.degradation_history:
            degradation_tier = runner.state.degradation_history[-1].tier

        status = "healthy" if degradation_tier == "FULL" else "degraded"

        return HealthResponse(
            status=status,
            uptime_seconds=uptime,
            running=runner.running,
            last_run=runner.state.last_run.isoformat() if runner.state.last_run else None,
        )

    @app.get("/state/summary", response_model=StateSummaryResponse)
    async def state_summary() -> StateSummaryResponse:
        """Get daemon state summary."""
        runner: DaemonRunner = app.state.runner

        # Get current degradation tier
        degradation_tier = "FULL"
        if runner.state.degradation_history:
            degradation_tier = runner.state.degradation_history[-1].tier

        return StateSummaryResponse(
            total_analyses=runner.state.total_analyses,
            total_trades=runner.state.total_trades,
            error_count=len(runner.state.errors),
            degradation_tier=degradation_tier,
            trading_mode=runner.state.current_trading_mode,
        )

    @app.get("/config", response_model=ConfigResponse)
    async def config() -> ConfigResponse:
        """Get daemon configuration (no secrets)."""
        runner: DaemonRunner = app.state.runner

        return ConfigResponse(
            watchlist=runner.config.watchlist,
            interval_minutes=runner.config.interval_minutes,
            market_hours_only=runner.config.market_hours_only,
            auto_trade=runner.config.auto_trade,
            trading_mode=runner.state.current_trading_mode,
            pre_market_enabled=runner.config.schedule.enable_pre_market,
        )

    logger.info("FastAPI app created")
    return app
