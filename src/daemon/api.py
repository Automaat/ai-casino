"""Embedded FastAPI server for daemon monitoring."""

import asyncio
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
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
    rsi: float | None = None
    macd_hist: float | None = None


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
    current_price: float


class PositionsResponse(BaseModel):
    """Positions endpoint response."""

    positions: list[PositionResponse]
    count: int


class WatchlistResponse(BaseModel):
    """Watchlist endpoint response."""

    symbols: list[str]
    count: int
    sources: dict[str, int] = Field(description="Breakdown: config/broker/screening")


class SnapshotRecord(BaseModel):
    """Portfolio snapshot record."""

    timestamp: datetime
    portfolio_value: float
    balance: float
    total_exposure: float


class SnapshotsResponse(BaseModel):
    """Snapshots endpoint response."""

    snapshots: list[SnapshotRecord]
    count: int


class RebalanceAllocation(BaseModel):
    """Rebalance allocation record."""

    symbol: str
    target_weight: float
    current_weight: float
    delta: float
    action: str


class RebalanceResponse(BaseModel):
    """Rebalance endpoint response."""

    timestamp: datetime
    method: str
    allocations: list[RebalanceAllocation]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float


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


class RiskHistoryResponse(BaseModel):
    """Risk report history."""

    reports: list[RiskReportResponse]
    count: int


class SectorRotationResponse(BaseModel):
    """Sector rotation analysis."""

    timestamp: datetime
    leading_sectors: list[str]
    lagging_sectors: list[str]
    sector_strengths: dict[str, float]
    sector_momenta: dict[str, str]
    flagged_positions: list[str]


class CorrelationMatrixResponse(BaseModel):
    """Correlation matrix."""

    timestamp: datetime
    num_positions: int
    correlation_matrix: dict[str, dict[str, float]]
    symbols: list[str]
    max_correlation: float
    avg_correlation: float


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

        broker_positions = {}
        if runner.broker:
            try:
                account_info = await asyncio.to_thread(runner.broker.get_account_info)
                broker_positions = account_info.positions
            except Exception as e:
                logger.warning(f"Failed to fetch broker positions: {e}")

        positions = []
        for symbol, pos_dict in runner.state.active_positions.items():
            try:
                pos = PositionRecord.model_validate(pos_dict)

                current_price = pos.entry_price
                if symbol in broker_positions:
                    broker_pos = broker_positions[symbol]
                    current_price = (
                        broker_pos.market_value / broker_pos.qty if broker_pos.qty > 0 else pos.entry_price
                    )

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
                        current_price=current_price,
                    )
                )
            except Exception as e:
                logger.error(f"Failed to parse position {symbol}: {e}")
                continue

        return PositionsResponse(positions=positions, count=len(positions))

    @app.get("/portfolio/snapshots", response_model=SnapshotsResponse)
    async def get_snapshots(days: int = 30) -> SnapshotsResponse:
        """Get portfolio snapshots history."""
        from src.database.connection import get_session
        from src.database.repositories.snapshot import PortfolioSnapshotRepository

        # Clamp days to prevent abuse
        days = max(1, min(days, 365))

        start = datetime.now(UTC) - timedelta(days=days)
        end = datetime.now(UTC)

        try:
            async with get_session() as session:
                repo = PortfolioSnapshotRepository(session)
                snapshots = await repo.get_by_date_range(start, end)

                snapshot_records = [
                    SnapshotRecord(
                        timestamp=s.timestamp,
                        portfolio_value=s.portfolio_value,
                        balance=s.balance,
                        total_exposure=s.total_exposure,
                    )
                    for s in snapshots
                ]

                return SnapshotsResponse(snapshots=snapshot_records, count=len(snapshot_records))
        except Exception as e:
            logger.error(f"Failed to fetch snapshots: {e}")
            raise HTTPException(status_code=500, detail="Failed to fetch portfolio snapshots") from e

    @app.get("/portfolio/rebalance", response_model=RebalanceResponse | None)
    async def get_rebalance() -> RebalanceResponse | None:
        """Get latest portfolio rebalance data."""
        runner: DaemonRunner = app.state.runner

        if not runner.state.portfolio_rebalancing_history:
            return None

        latest = runner.state.portfolio_rebalancing_history[-1]

        broker_positions = {}
        total_portfolio_value = 0.0
        if runner.broker:
            try:
                account_info = await asyncio.to_thread(runner.broker.get_account_info)
                broker_positions = account_info.positions
                total_portfolio_value = account_info.portfolio_value
            except Exception as e:
                logger.warning(f"Failed to fetch broker positions for rebalance: {e}")

        allocations = []
        for symbol, target_weight in latest.target_weights.items():
            current_weight = 0.0
            if symbol in broker_positions and total_portfolio_value > 0:
                current_weight = broker_positions[symbol].market_value / total_portfolio_value

            delta = current_weight - target_weight
            action = "REDUCE" if delta > 0 else "INCREASE" if delta < 0 else "HOLD"

            allocations.append(
                RebalanceAllocation(
                    symbol=symbol,
                    target_weight=target_weight,
                    current_weight=current_weight,
                    delta=delta,
                    action=action,
                )
            )

        return RebalanceResponse(
            timestamp=latest.timestamp,
            method=latest.method,
            allocations=allocations,
            expected_return=latest.expected_return,
            expected_volatility=latest.expected_volatility,
            sharpe_ratio=latest.sharpe_ratio,
        )

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

    @app.get("/risk/history", response_model=RiskHistoryResponse)
    async def get_risk_history() -> RiskHistoryResponse:
        """Get historical risk reports."""
        runner: DaemonRunner = app.state.runner
        reports = [
            RiskReportResponse(
                timestamp=r.timestamp,
                var_95=r.var_95,
                var_99=r.var_99,
                cvar_95=r.cvar_95,
                cvar_99=r.cvar_99,
                cdar_95=r.cdar_95,
                max_drawdown=r.max_drawdown,
                risk_status=r.risk_status,
            )
            for r in runner.state.risk_report_history
        ]
        return RiskHistoryResponse(reports=reports, count=len(reports))

    @app.get("/sector-rotation/latest", response_model=SectorRotationResponse | None)
    async def get_sector_rotation() -> SectorRotationResponse | None:
        """Get latest sector rotation analysis."""
        runner: DaemonRunner = app.state.runner
        if not runner.state.sector_rotation_history:
            return None
        latest = runner.state.sector_rotation_history[-1]
        return SectorRotationResponse(
            timestamp=latest.timestamp,
            leading_sectors=latest.leading_sectors,
            lagging_sectors=latest.lagging_sectors,
            sector_strengths=latest.sector_strengths,
            sector_momenta=latest.sector_momenta,
            flagged_positions=latest.flagged_positions,
        )

    @app.get("/correlation/latest", response_model=CorrelationMatrixResponse | None)
    async def get_correlation_matrix() -> CorrelationMatrixResponse | None:
        """Get latest correlation matrix."""
        from src.data.market import MarketDataFetcher
        from src.metrics.correlation import CorrelationAuditor

        auditor = CorrelationAuditor(MarketDataFetcher())
        audit_result = auditor.load_latest()

        if not audit_result:
            return None

        symbols = sorted(audit_result.correlation_matrix.keys())
        return CorrelationMatrixResponse(
            timestamp=audit_result.audit_date,
            num_positions=audit_result.num_positions,
            correlation_matrix=audit_result.correlation_matrix,
            symbols=symbols,
            max_correlation=audit_result.max_correlation,
            avg_correlation=audit_result.avg_correlation,
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
