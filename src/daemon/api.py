"""Embedded FastAPI server for daemon monitoring."""

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
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


class AnalysisRecordResponse(BaseModel):
    """Single analysis for API."""

    symbol: str
    timestamp: datetime
    signal: str
    confidence: float
    executed_trade: bool
    trading_session: str
    is_paper_trade: bool


class AnalysesResponse(BaseModel):
    """Analyses endpoint response."""

    analyses: list[AnalysisRecordResponse]
    total_count: int = Field(description="Total in history")
    returned_count: int = Field(description="Returned")


class PositionResponse(BaseModel):
    """Active position (excludes internal fields)."""

    symbol: str
    entry_price: float
    current_qty: float
    current_stop_loss: float
    entry_timestamp: datetime
    entry_signal: str
    entry_confidence: float
    days_held: int
    trailing_stop_activated: bool
    breakeven_activated: bool
    profit_targets: list[float]


class PositionsResponse(BaseModel):
    """Positions endpoint response."""

    positions: list[PositionResponse]
    count: int


class WatchlistResponse(BaseModel):
    """Watchlist endpoint response."""

    symbols: list[str]
    count: int
    sources: dict[str, int] = Field(description="Breakdown: config/broker/screening")


class RiskReportResponse(BaseModel):
    """Risk report endpoint response."""

    timestamp: datetime
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    cdar_95: float
    max_drawdown: float
    risk_status: str


class DegradationResponse(BaseModel):
    """Degradation endpoint response."""

    tier: str
    unavailable_services: list[str]
    confidence_adjustment: float
    halt_reason: str | None


class GamePlanResponse(BaseModel):
    """Game plan endpoint response."""

    date: str
    priority_symbols: list[str]
    risk_stance: str
    sector_focus: list[str]
    reasoning: str
    confidence: float
    generated_at: str


class EventResponse(BaseModel):
    """Events endpoint response."""

    events: list[dict]
    returned_count: int


def create_api_app(runner: "DaemonRunner") -> FastAPI:  # noqa: C901, PLR0915
    """Create FastAPI app with runner reference.

    Complexity acceptable for FastAPI route registration pattern.

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
        allow_methods=["GET", "OPTIONS"],
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

    @app.get("/analyses", response_model=AnalysesResponse)
    async def get_analyses(limit: int = 50, symbol: str | None = None) -> AnalysesResponse:
        """Get analysis history."""
        runner: DaemonRunner = app.state.runner
        limit = max(0, min(limit, 500))

        analyses = list(reversed(runner.state.analyses))

        if symbol:
            analyses = [a for a in analyses if a.symbol == symbol]

        analyses = analyses[:limit]

        return AnalysesResponse(
            analyses=[AnalysisRecordResponse(**a.model_dump()) for a in analyses],
            total_count=runner.state.total_analyses,
            returned_count=len(analyses),
        )

    @app.get("/positions", response_model=PositionsResponse)
    async def get_positions() -> PositionsResponse:
        """Get active positions."""
        runner: DaemonRunner = app.state.runner

        from src.daemon.positions import PositionRecord

        positions = []
        for symbol, pos_dict in runner.state.active_positions.items():
            try:
                pos = PositionRecord.model_validate(pos_dict)
                positions.append(
                    PositionResponse(
                        symbol=pos.symbol,
                        entry_price=pos.entry_price,
                        current_qty=pos.current_qty,
                        current_stop_loss=pos.current_stop_loss,
                        entry_timestamp=pos.entry_timestamp,
                        entry_signal=pos.entry_signal,
                        entry_confidence=pos.entry_confidence,
                        days_held=pos.days_held,
                        trailing_stop_activated=pos.trailing_stop_activated,
                        breakeven_activated=pos.breakeven_activated,
                        profit_targets=pos.profit_targets,
                    )
                )
            except Exception as e:
                logger.error(f"Failed to parse position {symbol}: {e}")
                continue

        return PositionsResponse(positions=positions, count=len(positions))

    @app.get("/watchlist", response_model=WatchlistResponse)
    async def get_watchlist() -> WatchlistResponse:
        """Get merged watchlist."""
        runner: DaemonRunner = app.state.runner

        # Acceptable coupling for API layer - private method access
        symbols = runner._get_merged_watchlist()  # noqa: SLF001

        config_count = len([s for s in runner.config.watchlist if s in symbols])

        broker_count = 0
        try:
            broker_symbols = set(runner.state.active_positions.keys())
            broker_count = len([s for s in broker_symbols if s in symbols])
        except Exception as e:
            logger.warning(f"Unable to derive broker symbols for watchlist: {e}")

        screening_count = 0
        if runner.config.screening.enabled and runner.state.screening_history:
            latest = runner.state.screening_history[-1]
            screening_count = len([s for s in latest.top_symbols if s in symbols])

        return WatchlistResponse(
            symbols=symbols,
            count=len(symbols),
            sources={
                "config": config_count,
                "broker": broker_count,
                "screening": screening_count,
            },
        )

    @app.get("/risk", response_model=RiskReportResponse | None)
    async def get_risk() -> RiskReportResponse | None:
        """Get latest risk report."""
        runner: DaemonRunner = app.state.runner

        if not runner.state.risk_report_history:
            return None

        latest = runner.state.risk_report_history[-1]

        return RiskReportResponse(
            timestamp=latest.timestamp,
            var_95=latest.var_95,
            var_99=latest.var_99,
            cvar_95=latest.cvar_95,
            cvar_99=latest.cvar_99,
            cdar_95=latest.cdar_95,
            max_drawdown=latest.max_drawdown,
            risk_status=latest.risk_status,
        )

    @app.get("/degradation", response_model=DegradationResponse)
    async def get_degradation() -> DegradationResponse:
        """Get current degradation status."""
        runner: DaemonRunner = app.state.runner

        if not runner.state.degradation_history:
            return DegradationResponse(
                tier="FULL",
                unavailable_services=[],
                confidence_adjustment=1.0,
                halt_reason=None,
            )

        latest = runner.state.degradation_history[-1]

        return DegradationResponse(
            tier=latest.tier,
            unavailable_services=latest.unavailable_services,
            confidence_adjustment=latest.confidence_adjustment,
            halt_reason=latest.halt_reason,
        )

    @app.get("/game-plan", response_model=GamePlanResponse | None)
    async def get_game_plan() -> GamePlanResponse | None:
        """Get latest game plan (if enabled and generated)."""
        import json
        from pathlib import Path

        runner: DaemonRunner = app.state.runner

        if not runner.config.game_plan.enabled or not runner.state.game_plan_history:
            return None

        latest = runner.state.game_plan_history[-1]
        plan_dir = Path(runner.config.game_plan.plan_dir).expanduser()
        plan_file = plan_dir / f"{latest.timestamp.date()}.json"

        if not plan_file.exists():
            logger.warning(f"Game plan file not found: {plan_file}")
            return None

        try:
            with plan_file.open() as f:
                plan_data = json.load(f)

            return GamePlanResponse(
                date=plan_data["date"],
                priority_symbols=plan_data["priority_symbols"],
                risk_stance=plan_data["risk_stance"],
                sector_focus=plan_data["sector_focus"],
                reasoning=plan_data["reasoning"],
                confidence=plan_data["confidence"],
                generated_at=plan_data["generated_at"],
            )
        except Exception as e:
            logger.error(f"Failed to load game plan: {e}")
            return None

    @app.get("/events", response_model=EventResponse)
    async def get_events(limit: int = 100) -> EventResponse:
        """Get event history."""
        runner: DaemonRunner = app.state.runner

        if not runner.event_bus:
            return EventResponse(events=[], returned_count=0)

        limit = max(0, min(limit, 500))

        events = runner.event_bus.get_history(limit=limit)

        events_dict = [e.model_dump(mode="json") for e in events]

        return EventResponse(events=events_dict, returned_count=len(events_dict))

    @app.websocket("/ws/events")
    async def websocket_events(websocket: WebSocket) -> None:
        """Stream real-time events to dashboard."""
        runner: DaemonRunner = app.state.runner

        if not runner.event_bus:
            await websocket.close(code=1011, reason="EventBus not available")
            return

        await websocket.accept()
        logger.info(f"WebSocket connected: {websocket.client}")

        subscriber_id, queue = await runner.event_bus.subscribe()

        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                    event_dict = event.model_dump(mode="json")
                    await websocket.send_json(event_dict)
                except TimeoutError:
                    # Ping client to detect disconnect
                    try:
                        await websocket.send_json({"type": "ping"})
                    except Exception:
                        logger.info("Client disconnected during ping")
                        break

        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected: {websocket.client}")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            await runner.event_bus.unsubscribe(subscriber_id)
            logger.info(f"Unsubscribed: {subscriber_id}")

    logger.info("FastAPI app created")
    return app
