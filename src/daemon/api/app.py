"""FastAPI application factory for daemon monitoring API."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from contextvars import ContextVar
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.daemon.api.models import (
    AnalysesResponse,
    AnalysisRecordResponse,
    ConfigResponse,
    CorrelationMatrixResponse,
    DegradationHistoryResponse,
    DegradationResponse,
    EventResponse,
    ExecutionMetricsListResponse,
    FullConfigResponse,
    GamePlanResponse,
    HealthResponse,
    MarketEventsResponse,
    PositionResponse,
    PositionsResponse,
    RebalanceAllocation,
    RebalanceResponse,
    RiskHistoryResponse,
    RiskReportResponse,
    SectorRotationResponse,
    SnapshotRecord,
    SnapshotsResponse,
    StateSummaryResponse,
    WatchlistResponse,
)

if TYPE_CHECKING:
    from src.daemon.factory import DaemonComponents

_broker_cache: ContextVar[dict[str, Any] | None] = ContextVar("_broker_cache", default=None)


def _mask_sensitive_field(value: str | None) -> str:
    """Mask sensitive API key (show first 4 + last 4 chars when long enough).

    Args:
        value: API key or None

    Returns:
        Masked string (e.g., "sk-1234...xy89") for values with length >= 8,
        "***" for shorter non-empty values, or "Not set" when value is falsy.
    """
    if not value:
        return "Not set"
    if len(value) < 8:
        return "***"
    return f"{value[:4]}...{value[-4:]}"


@asynccontextmanager
async def get_broker_account_info_cached(
    components: DaemonComponents,
) -> AsyncIterator[dict[str, Any] | None]:
    """Request-scoped cached broker account info.

    Args:
        components: DaemonComponents instance

    Yields:
        Broker account info dict or None if broker unavailable
    """
    if not components.broker:
        yield None
        return

    token = _broker_cache.set({})
    cache: dict[str, Any] = _broker_cache.get()  # type: ignore[assignment]
    cache_key = "account_info"

    try:
        if cache_key not in cache:
            try:
                from src.data.broker import BrokerAccountInfo

                account_info: BrokerAccountInfo = await asyncio.to_thread(components.broker.get_account_info)
                cache[cache_key] = {
                    "positions": account_info.positions,
                    "portfolio_value": account_info.portfolio_value,
                }
            except Exception as e:
                logger.warning(f"Failed to fetch broker account info: {e}")
                cache[cache_key] = None

        yield cache[cache_key]
    finally:
        _broker_cache.reset(token)


def create_api_app(components: DaemonComponents) -> FastAPI:  # noqa: C901, PLR0915
    """Create FastAPI app with components reference.

    Complexity acceptable for FastAPI route registration pattern.

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

    # CORS middleware - allow configured dashboard origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=components.config.api.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """Get daemon health status."""
        components: DaemonComponents = app.state.components
        uptime = (datetime.now(UTC) - app.state.start_time).total_seconds()

        # Determine health status from degradation tier
        degradation_tier = "FULL"
        if components.state.degradation_history:
            degradation_tier = components.state.degradation_history[-1].tier

        status = "healthy" if degradation_tier == "FULL" else "degraded"

        return HealthResponse(
            status=status,
            uptime_seconds=uptime,
            running=components.running,
            last_run=components.state.last_run.isoformat() if components.state.last_run else None,
        )

    @app.get("/state/summary", response_model=StateSummaryResponse)
    async def state_summary() -> StateSummaryResponse:
        """Get daemon state summary."""
        components: DaemonComponents = app.state.components

        # Get current degradation tier
        degradation_tier = "FULL"
        if components.state.degradation_history:
            degradation_tier = components.state.degradation_history[-1].tier

        return StateSummaryResponse(
            total_analyses=components.state.total_analyses,
            total_trades=components.state.total_trades,
            error_count=len(components.state.errors),
            degradation_tier=degradation_tier,
            trading_mode=components.state.current_trading_mode,
        )

    @app.get("/config", response_model=ConfigResponse)
    async def config() -> ConfigResponse:
        """Get daemon configuration (no secrets)."""
        components: DaemonComponents = app.state.components

        return ConfigResponse(
            watchlist=components.config.watchlist,
            interval_minutes=components.config.interval_minutes,
            market_hours_only=components.config.market_hours_only,
            auto_trade=components.config.auto_trade,
            trading_mode=components.state.current_trading_mode,
            pre_market_enabled=components.config.schedule.enable_pre_market,
        )

    @app.get("/config/full", response_model=FullConfigResponse)
    async def config_full() -> FullConfigResponse:
        """Get full daemon configuration with masked sensitive fields."""
        components: DaemonComponents = app.state.components
        cfg = components.config

        # Mask API keys
        masked_api_keys = {
            "alpha_vantage_api_key": _mask_sensitive_field(cfg.api_keys.alpha_vantage_api_key),
            "marketaux_api_key": _mask_sensitive_field(cfg.api_keys.marketaux_api_key),
            "finnhub_api_key": _mask_sensitive_field(cfg.api_keys.finnhub_api_key),
            "alpaca_api_key": _mask_sensitive_field(cfg.api_keys.alpaca_api_key),
            "alpaca_secret_key": _mask_sensitive_field(cfg.api_keys.alpaca_secret_key),
            "alpaca_paper_api_key": _mask_sensitive_field(cfg.api_keys.alpaca_paper_api_key),
            "alpaca_paper_secret_key": _mask_sensitive_field(cfg.api_keys.alpaca_paper_secret_key),
            "reddit_client_id": _mask_sensitive_field(cfg.api_keys.reddit_client_id),
            "reddit_client_secret": _mask_sensitive_field(cfg.api_keys.reddit_client_secret),
            "reddit_user_agent": _mask_sensitive_field(cfg.api_keys.reddit_user_agent),
            "anthropic_api_key": _mask_sensitive_field(cfg.api_keys.anthropic_api_key),
            "openai_api_key": _mask_sensitive_field(cfg.api_keys.openai_api_key),
            "openai_api_base": _mask_sensitive_field(cfg.api_keys.openai_api_base),
        }

        # Mask telegram secrets
        notifications_dict = cfg.notifications.model_dump()
        notifications_dict["telegram"]["bot_token"] = _mask_sensitive_field(
            cfg.notifications.telegram.bot_token
        )
        notifications_dict["telegram"]["chat_id"] = _mask_sensitive_field(cfg.notifications.telegram.chat_id)

        return FullConfigResponse(
            watchlist=cfg.watchlist,
            interval_minutes=cfg.interval_minutes,
            market_hours_only=cfg.market_hours_only,
            auto_trade=cfg.auto_trade,
            max_concurrent_analyses=cfg.max_concurrent_analyses,
            trading_mode=components.state.current_trading_mode,
            paper_trading=cfg.paper_trading.model_dump(),
            schedule=cfg.schedule.model_dump(),
            state=cfg.state.model_dump(),
            journal=cfg.journal.model_dump(),
            health=cfg.health.model_dump(),
            optimization=cfg.optimization.model_dump(),
            screening=cfg.screening.model_dump(),
            prefetch=cfg.prefetch.model_dump(),
            sector_rotation=cfg.sector_rotation.model_dump(),
            earnings_calendar=cfg.earnings_calendar.model_dump(),
            peer_analysis=cfg.peer_analysis.model_dump(),
            correlation_audit=cfg.correlation_audit.model_dump(),
            reporting=cfg.reporting.model_dump(),
            risk_limits=cfg.risk_limits.model_dump(),
            rebalancing=cfg.rebalancing.model_dump(),
            signal_tracking=cfg.signal_tracking.model_dump(),
            pre_trade_backtesting=cfg.pre_trade_backtesting.model_dump(),
            game_plan=cfg.game_plan.model_dump(),
            position_management=cfg.position_management.model_dump(),
            monte_carlo=cfg.monte_carlo.model_dump(),
            notifications=notifications_dict,
            analysis_orchestration=cfg.analysis_orchestration.model_dump(),
            news_watcher=cfg.news_watcher.model_dump(),
            social_watcher=cfg.social_watcher.model_dump(),
            filings_watcher=cfg.filings_watcher.model_dump(),
            anomaly_watcher=cfg.anomaly_watcher.model_dump(),
            api=cfg.api.model_dump(),
            llm=cfg.llm.model_dump(),
            api_keys=masked_api_keys,
        )

    @app.get("/analyses", response_model=AnalysesResponse)
    async def get_analyses(limit: int = 50, symbol: str | None = None) -> AnalysesResponse:
        """Get analysis history."""
        components: DaemonComponents = app.state.components
        limit = max(0, min(limit, 500))

        analyses = list(reversed(components.state.analyses))

        if symbol:
            analyses = [a for a in analyses if a.symbol == symbol]

        analyses = analyses[:limit]

        return AnalysesResponse(
            analyses=[AnalysisRecordResponse(**a.model_dump()) for a in analyses],
            total_count=components.state.total_analyses,
            returned_count=len(analyses),
        )

    @app.get("/positions", response_model=PositionsResponse)
    async def get_positions() -> PositionsResponse:
        """Get active positions."""
        components: DaemonComponents = app.state.components

        from src.daemon.positions import PositionRecord

        async with get_broker_account_info_cached(components) as account_info:
            broker_positions = account_info["positions"] if account_info else {}

            positions = []
            for symbol, pos_dict in components.state.active_positions.items():
                try:
                    pos = PositionRecord.model_validate(pos_dict)

                    current_price = pos.entry_price
                    if symbol in broker_positions:
                        broker_pos = broker_positions[symbol]
                        current_price = (
                            broker_pos.market_value / broker_pos.qty
                            if broker_pos.qty > 0
                            else pos.entry_price
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
        import os

        from src.database.connection import get_session
        from src.database.engine import MissingDatabaseURLError
        from src.database.repositories.snapshot import PortfolioSnapshotRepository

        # Return empty if DATABASE_URL not configured
        if not os.getenv("DATABASE_URL"):
            logger.debug("DATABASE_URL not set, returning empty snapshots")
            return SnapshotsResponse(snapshots=[], count=0)

        # Clamp days to prevent abuse
        days = max(1, min(days, 365))

        start = datetime.now(UTC) - timedelta(days=days)
        end = datetime.now(UTC)

        try:
            async with get_session() as session:
                repo = PortfolioSnapshotRepository(session)
                if days > 7:
                    snapshots = await repo.get_by_date_range_sampled(start, end, max_points=100)
                else:
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
        except MissingDatabaseURLError:
            logger.debug("DATABASE_URL not configured, returning empty snapshots")
            return SnapshotsResponse(snapshots=[], count=0)
        except Exception as e:
            logger.error(f"Failed to fetch snapshots: {e}")
            raise HTTPException(status_code=500, detail="Failed to fetch portfolio snapshots") from e

    @app.get("/portfolio/rebalance", response_model=RebalanceResponse | None)
    async def get_rebalance() -> RebalanceResponse | None:
        """Get latest portfolio rebalance data."""
        components: DaemonComponents = app.state.components

        if not components.state.portfolio_rebalancing_history:
            return None

        latest = components.state.portfolio_rebalancing_history[-1]

        async with get_broker_account_info_cached(components) as account_info:
            broker_positions = account_info["positions"] if account_info else {}
            total_portfolio_value = account_info["portfolio_value"] if account_info else 0.0

            allocations = []
            for allocation in latest.allocations:
                current_weight = 0.0
                if allocation.symbol in broker_positions and total_portfolio_value > 0:
                    current_weight = broker_positions[allocation.symbol].market_value / total_portfolio_value

                delta = current_weight - allocation.weight
                action = "REDUCE" if delta > 0 else "INCREASE" if delta < 0 else "HOLD"

                allocations.append(
                    RebalanceAllocation(
                        symbol=allocation.symbol,
                        target_weight=allocation.weight,
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
        components: DaemonComponents = app.state.components

        symbols = components.broker_manager.get_merged_watchlist()

        config_count = len([s for s in components.config.watchlist if s in symbols])

        broker_count = 0
        try:
            broker_symbols = set(components.state.active_positions.keys())
            broker_count = len([s for s in broker_symbols if s in symbols])
        except Exception as e:
            logger.warning(f"Unable to derive broker symbols for watchlist: {e}")

        screening_count = 0
        if components.config.screening.enabled and components.state.screening_history:
            latest = components.state.screening_history[-1]
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
        components: DaemonComponents = app.state.components

        if not components.state.risk_report_history:
            return None

        latest = components.state.risk_report_history[-1]

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
        components: DaemonComponents = app.state.components
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
            for r in components.state.risk_report_history
        ]
        return RiskHistoryResponse(reports=reports, count=len(reports))

    @app.get("/sector-rotation/latest", response_model=SectorRotationResponse | None)
    async def get_sector_rotation() -> SectorRotationResponse | None:
        """Get latest sector rotation analysis."""
        components: DaemonComponents = app.state.components
        if not components.state.sector_rotation_history:
            return None
        latest = components.state.sector_rotation_history[-1]
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
        from src.metrics.correlation import CorrelationAuditor

        # Use runner's configured output_dir for consistency with daemon
        # market_fetcher=None because load_latest() only reads from disk
        auditor = CorrelationAuditor(
            market_fetcher=None,
            output_dir=components.config.correlation_audit.output_dir,
        )
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
        components: DaemonComponents = app.state.components

        if not components.state.degradation_history:
            return DegradationResponse(
                tier="FULL",
                unavailable_services=[],
                confidence_adjustment=1.0,
                halt_reason=None,
            )

        latest = components.state.degradation_history[-1]

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

        components: DaemonComponents = app.state.components

        if not components.config.game_plan.enabled or not components.state.game_plan_history:
            return None

        latest = components.state.game_plan_history[-1]
        plan_dir = Path(components.config.game_plan.plan_dir).expanduser()
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
        components: DaemonComponents = app.state.components

        if not components.event_bus:
            return EventResponse(events=[], returned_count=0)

        limit = max(0, min(limit, 500))

        events = components.event_bus.get_history(limit=limit)

        events_dict = [e.model_dump(mode="json") for e in events]

        return EventResponse(events=events_dict, returned_count=len(events_dict))

    @app.get("/events/market", response_model=MarketEventsResponse)
    async def get_market_events(limit: int = 100) -> MarketEventsResponse:
        """Get market event signals (news, social, anomaly)."""
        components: DaemonComponents = app.state.components
        limit = max(0, min(limit, 500))

        if limit <= 0:
            events = []
        else:
            events = components.state.market_events[-limit:] if components.state.market_events else []

        return MarketEventsResponse(events=events, returned_count=len(events))

    @app.get("/events/degradation-history", response_model=DegradationHistoryResponse)
    async def get_degradation_history(limit: int = 50) -> DegradationHistoryResponse:
        """Get degradation history for timeline."""
        components: DaemonComponents = app.state.components
        limit = max(0, min(limit, 200))

        if limit <= 0:
            history = []
        else:
            history = (
                components.state.degradation_history[-limit:] if components.state.degradation_history else []
            )

        return DegradationHistoryResponse(
            records=[r.model_dump(mode="json") for r in history],
            count=len(history),
        )

    @app.get("/api/execution-metrics", response_model=ExecutionMetricsListResponse)
    async def get_execution_metrics(limit: int = 50) -> ExecutionMetricsListResponse:
        """Get recent execution metrics from JSONL.

        Args:
            limit: Max number of metrics to return (clamped to 1-500)

        Returns:
            ExecutionMetricsListResponse with list of metrics
        """
        import json
        from pathlib import Path

        limit = max(1, min(limit, 500))
        metrics_file = Path("logs/execution_metrics.jsonl").expanduser()

        if not metrics_file.exists():
            return ExecutionMetricsListResponse(metrics=[], count=0)

        metrics = []
        try:
            # Read last N lines efficiently (read backwards)
            with metrics_file.open("rb") as f:
                f.seek(0, 2)
                file_size = f.tell()
                if file_size == 0:
                    return ExecutionMetricsListResponse(metrics=[], count=0)

                # Read file in chunks from end
                buffer_size = 8192
                lines = []
                buffer = b""
                pos = file_size

                while pos > 0 and len(lines) < limit:
                    chunk_size = min(buffer_size, pos)
                    pos -= chunk_size
                    f.seek(pos)
                    chunk = f.read(chunk_size)
                    buffer = chunk + buffer

                    # Extract complete lines
                    while b"\n" in buffer and len(lines) < limit:
                        buffer, line = buffer.rsplit(b"\n", 1)
                        if line:
                            lines.insert(0, line)

                # Parse JSONL
                for line in lines[-limit:]:
                    try:
                        metric = json.loads(line)
                        metrics.append(metric)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Malformed JSONL line: {e}")
                        continue

                # Reverse to get newest first
                metrics.reverse()

        except Exception as e:
            logger.error(f"Failed to read execution metrics: {e}")
            raise HTTPException(status_code=500, detail="Failed to read execution metrics") from e

        return ExecutionMetricsListResponse(metrics=metrics, count=len(metrics))

    @app.get("/api/execution-metrics/{workflow_id}", response_model=dict)
    async def get_execution_metric_detail(workflow_id: str) -> dict:
        """Get single workflow execution detail.

        Args:
            workflow_id: Workflow ID to fetch

        Returns:
            WorkflowExecutionMetrics as dict
        """
        import json
        from pathlib import Path

        metrics_file = Path("logs/execution_metrics.jsonl").expanduser()

        if not metrics_file.exists():
            raise HTTPException(status_code=404, detail="Execution metrics file not found")

        try:
            with metrics_file.open() as f:
                for line in f:
                    try:
                        metric = json.loads(line)
                        if metric.get("workflow_id") == workflow_id:
                            return metric
                    except json.JSONDecodeError:
                        continue

            raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to fetch workflow detail: {e}")
            raise HTTPException(status_code=500, detail="Failed to fetch workflow detail") from e

    @app.websocket("/ws/events")
    async def websocket_events(websocket: WebSocket) -> None:
        """Stream real-time events to dashboard."""
        components: DaemonComponents = app.state.components

        if not components.event_bus:
            await websocket.close(code=1011, reason="EventBus not available")
            return

        await websocket.accept()
        logger.info(f"WebSocket connected: {websocket.client}")

        subscriber_id, queue = await components.event_bus.subscribe()

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
            await components.event_bus.unsubscribe(subscriber_id)
            logger.info(f"Unsubscribed: {subscriber_id}")

    logger.info("FastAPI app created")
    return app
