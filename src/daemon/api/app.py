"""FastAPI application factory for daemon monitoring API."""

import asyncio
import queue
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from contextvars import ContextVar
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.daemon.api.models import (
    ActiveExecutionGraphsResponse,
    AnalysesResponse,
    AnalysisRecordResponse,
    ConfigResponse,
    CorrelationMatrixResponse,
    DegradationHistoryResponse,
    DegradationResponse,
    EventResponse,
    ExecutionGraphDetailResponse,
    ExecutionGraphHistoryResponse,
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
    ServiceCheck,
    ServiceHealthResponse,
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
                logger.opt(exception=True).warning(f"Failed to fetch broker account info: {e}")
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

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """Get daemon health status."""
        components: DaemonComponents = app.state.components
        uptime = (datetime.now(UTC) - app.state.start_time).total_seconds()

        # Determine health status from degradation tier
        degradation_tier = "FULL"
        last_run = None

        try:
            degradation_history = await components.state.get_degradation_history(limit=1)
            if degradation_history:
                degradation_tier = degradation_history[-1].tier

            last_run = await components.state.get_last_run()
        except Exception:
            # DB temporarily unavailable due to concurrent operations - still healthy
            pass

        status = "healthy" if degradation_tier == "FULL" else "degraded"

        return HealthResponse(
            status=status,
            uptime_seconds=uptime,
            daemon_running=components.running,
            last_run=last_run.isoformat() if last_run else None,
        )

    @app.get("/state/summary", response_model=StateSummaryResponse)
    async def state_summary() -> StateSummaryResponse:
        """Get daemon state summary."""
        components: DaemonComponents = app.state.components

        # Get current degradation tier
        degradation_tier = "FULL"
        degradation_history = await components.state.get_degradation_history(limit=1)
        if degradation_history:
            degradation_tier = degradation_history[-1].tier

        # Calculate positions count
        active_positions = await components.state.get_active_positions()
        positions_count = len(active_positions)

        # Win rate calculation - not available in current state (would need trades history)
        win_rate = None

        # Get recent analyses (last 50), convert to dicts
        all_analyses = await components.state.get_analyses(limit=50)
        recent_analyses = [
            analysis if isinstance(analysis, dict) else analysis.model_dump(mode="json")
            for analysis in all_analyses
        ]

        total_analyses = await components.state.get_total_analyses()
        total_trades = await components.state.get_total_trades()
        errors = await components.state.get_errors()
        trading_mode = await components.state.get_current_trading_mode()

        return StateSummaryResponse(
            total_analyses=total_analyses,
            recent_analyses=recent_analyses,
            total_trades=total_trades,
            positions_count=positions_count,
            win_rate=win_rate,
            error_count=len(errors),
            degradation_tier=degradation_tier,
            trading_mode=trading_mode,
        )

    @app.get("/config", response_model=ConfigResponse)
    async def config() -> ConfigResponse:
        """Get daemon configuration (no secrets)."""
        components: DaemonComponents = app.state.components

        trading_mode = await components.state.get_current_trading_mode()
        return ConfigResponse(
            watchlist=components.config.watchlist,
            interval_minutes=components.config.interval_minutes,
            market_hours_only=components.config.market_hours_only,
            auto_trade=components.config.auto_trade,
            trading_mode=trading_mode,
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
            trading_mode=await components.state.get_current_trading_mode(),
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

        all_analyses = await components.state.get_analyses(limit=1000)
        analyses = list(reversed(all_analyses))

        if symbol:
            analyses = [a for a in analyses if a.symbol == symbol]

        analyses = analyses[:limit]

        total_analyses = await components.state.get_total_analyses()
        return AnalysesResponse(
            analyses=[AnalysisRecordResponse(**a.model_dump()) for a in analyses],
            total_count=total_analyses,
            returned_count=len(analyses),
        )

    @app.get("/positions", response_model=PositionsResponse)
    async def get_positions() -> PositionsResponse:
        """Get active positions."""
        components: DaemonComponents = app.state.components

        from src.daemon.positions import PositionRecord

        async with get_broker_account_info_cached(components) as account_info:
            broker_positions = account_info["positions"] if account_info else {}

            active_positions = await components.state.get_active_positions()
            positions = []
            for symbol, pos_dict in dict(active_positions).items():
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
                    logger.opt(exception=True).error(f"Failed to parse position {symbol}: {e}")
                    continue

            return PositionsResponse(positions=positions, count=len(positions))

    @app.get("/portfolio/snapshots", response_model=SnapshotsResponse)
    async def get_snapshots(days: int = 30) -> SnapshotsResponse:
        """Get portfolio snapshots history."""
        from src.database.connection import get_session
        from src.database.engine import MissingDatabaseURLError
        from src.database.repositories.snapshot import PortfolioSnapshotRepository

        components: DaemonComponents = app.state.components

        # Clamp days to prevent abuse
        days = max(1, min(days, 365))

        start = datetime.now(UTC) - timedelta(days=days)
        end = datetime.now(UTC)

        # Check if database persistence is enabled
        database_enabled = components.config.database.enable_persistence

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

                # Check if we have any trade history
                total_trades = await components.state.get_total_trades()
                has_trades = len(snapshot_records) > 0 or total_trades > 0

                return SnapshotsResponse(
                    snapshots=snapshot_records,
                    count=len(snapshot_records),
                    database_enabled=database_enabled,
                    has_trades=has_trades,
                )
        except MissingDatabaseURLError:
            logger.debug("DATABASE_URL not configured, returning empty snapshots")
            total_trades = await components.state.get_total_trades()
            has_trades = total_trades > 0
            return SnapshotsResponse(
                snapshots=[],
                count=0,
                database_enabled=database_enabled,
                has_trades=has_trades,
            )
        except Exception as e:
            logger.opt(exception=True).error(f"Failed to fetch snapshots: {e}")
            raise HTTPException(status_code=500, detail="Failed to fetch portfolio snapshots") from e

    @app.get("/portfolio/rebalance", response_model=RebalanceResponse)
    async def get_rebalance() -> RebalanceResponse:
        """Get latest portfolio rebalance data."""
        components: DaemonComponents = app.state.components

        # Check if rebalancing is enabled
        rebalancing_enabled = components.config.rebalancing.enabled

        # If disabled or no history, return status-only response
        rebalancing_history = await components.state.get_rebalancing_history(limit=1)
        if not rebalancing_enabled or not rebalancing_history:
            return RebalanceResponse(enabled=rebalancing_enabled)

        latest = rebalancing_history[-1]

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
                enabled=rebalancing_enabled,
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

        symbols = await components.broker_manager.get_merged_watchlist()

        config_count = len([s for s in components.config.watchlist if s in symbols])

        broker_count = 0
        try:
            active_positions = await components.state.get_active_positions()
            broker_symbols = set(dict(active_positions).keys())
            broker_count = len([s for s in broker_symbols if s in symbols])
        except Exception as e:
            logger.opt(exception=True).warning(f"Unable to derive broker symbols for watchlist: {e}")

        screening_count = 0
        screening_history = await components.state.get_screening_history(limit=1)
        if components.config.screening.enabled and screening_history:
            latest = screening_history[-1]
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

        risk_history = await components.state.get_risk_report_history(limit=1)
        if not risk_history:
            return None

        latest = risk_history[-1]

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
        all_reports = await components.state.get_risk_report_history(limit=1000)
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
            for r in all_reports
        ]
        return RiskHistoryResponse(reports=reports, count=len(reports))

    @app.get("/sector-rotation/latest", response_model=SectorRotationResponse | None)
    async def get_sector_rotation() -> SectorRotationResponse | None:
        """Get latest sector rotation analysis."""
        components: DaemonComponents = app.state.components
        rotation_history = await components.state.get_sector_rotation_history(limit=1)
        if not rotation_history:
            return None
        latest = rotation_history[-1]
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

        degradation_history = await components.state.get_degradation_history(limit=1)
        if not degradation_history:
            return DegradationResponse(
                tier="FULL",
                unavailable_services=[],
                confidence_adjustment=1.0,
                halt_reason=None,
            )

        latest = degradation_history[-1]

        return DegradationResponse(
            tier=latest.tier,
            unavailable_services=latest.unavailable_services,
            confidence_adjustment=latest.confidence_adjustment,
            halt_reason=latest.halt_reason,
        )

    @app.get("/health/services", response_model=ServiceHealthResponse)
    async def get_service_health() -> ServiceHealthResponse:
        """Get individual service health checks."""
        import json
        from pathlib import Path

        components: DaemonComponents = app.state.components

        def _read_health_report() -> dict[str, Any]:
            """Read the latest health report from disk, with a safe fallback."""
            health_dir = Path(components.config.health.health_dir).expanduser()
            reports = sorted(health_dir.glob("health-*.json"))
            if not reports:
                return {"overall_status": "HEALTHY", "service_checks": []}

            latest_file = reports[-1]
            try:
                return json.loads(latest_file.read_text())
            except Exception as e:
                logger.opt(exception=True).warning(
                    f"Failed to read or parse health report {latest_file}: {e}"
                )
                return {"overall_status": "HEALTHY", "service_checks": []}

        # Read health report in thread to avoid blocking
        report_data = await asyncio.to_thread(_read_health_report)

        try:
            raw_checks = report_data.get("service_checks", [])
            if not isinstance(raw_checks, list):
                msg = "service_checks is not a list"
                raise TypeError(msg)

            # Convert ServiceCheckResult-like dicts to ServiceCheck models
            service_checks = [
                ServiceCheck(
                    service=check["service"],
                    status=check["status"],
                    message=check["message"],
                    duration_ms=check["duration_ms"],
                    checked_at=check["checked_at"],
                )
                for check in raw_checks
            ]

            overall_status = report_data.get("overall_status", "HEALTHY")
            return ServiceHealthResponse(
                overall_status=overall_status,
                service_checks=service_checks,
            )
        except Exception as e:
            logger.opt(exception=True).warning(
                f"Invalid health report format, using fallback health status: {e}"
            )
            return ServiceHealthResponse(
                overall_status="HEALTHY",
                service_checks=[],
            )

    @app.get("/game-plan", response_model=GamePlanResponse | None)
    async def get_game_plan() -> GamePlanResponse | None:
        """Get latest game plan (if enabled and generated)."""
        import json
        from pathlib import Path

        components: DaemonComponents = app.state.components

        game_plan_history = await components.state.get_game_plan_history(limit=1)
        if not components.config.game_plan.enabled or not game_plan_history:
            return None

        latest = game_plan_history[-1]

        def _load_plan_file() -> dict | None:
            plan_dir = Path(components.config.game_plan.plan_dir).expanduser()
            plan_file = plan_dir / f"{latest.timestamp.date()}.json"

            if not plan_file.exists():
                logger.warning(f"Game plan file not found: {plan_file}")
                return None

            with plan_file.open() as f:
                return json.load(f)

        try:
            plan_data = await asyncio.to_thread(_load_plan_file)
            if not plan_data:
                return None

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
            logger.opt(exception=True).error(f"Failed to load game plan: {e}")
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
            market_events = await components.state.get_market_events(limit=limit)
            events = market_events

        return MarketEventsResponse(events=events, returned_count=len(events))

    @app.get("/events/degradation-history", response_model=DegradationHistoryResponse)
    async def get_degradation_history(limit: int = 50) -> DegradationHistoryResponse:
        """Get degradation history for timeline."""
        components: DaemonComponents = app.state.components
        limit = max(0, min(limit, 200))

        if limit <= 0:
            history = []
        else:
            history = await components.state.get_degradation_history(limit=limit)

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

        def _read_metrics() -> list[dict]:
            metrics_file = Path("logs/execution_metrics.jsonl").expanduser()

            if not metrics_file.exists():
                return []

            metrics = []
            # Read last N lines efficiently (read backwards)
            with metrics_file.open("rb") as f:
                f.seek(0, 2)
                file_size = f.tell()
                if file_size == 0:
                    return []

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
                        logger.opt(exception=True).warning(f"Malformed JSONL line: {e}")
                        continue

                # Reverse to get newest first
                metrics.reverse()

            return metrics

        try:
            metrics = await asyncio.to_thread(_read_metrics)
        except Exception as e:
            logger.opt(exception=True).error(f"Failed to read execution metrics: {e}")
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

        def _find_metric() -> tuple[dict | None, bool]:
            """Find metric and return (metric, file_exists)."""
            metrics_file = Path("logs/execution_metrics.jsonl").expanduser()

            if not metrics_file.exists():
                return None, False

            with metrics_file.open() as f:
                for line in f:
                    try:
                        metric = json.loads(line)
                        if metric.get("workflow_id") == workflow_id:
                            return metric, True
                    except json.JSONDecodeError:
                        continue

            return None, True

        try:
            result, file_exists = await asyncio.to_thread(_find_metric)
            if result is None:
                detail = (
                    "Execution metrics file not found"
                    if not file_exists
                    else f"Workflow {workflow_id} not found"
                )
                raise HTTPException(status_code=404, detail=detail)
            return result
        except HTTPException:
            raise
        except Exception as e:
            logger.opt(exception=True).error(f"Failed to fetch workflow detail: {e}")
            raise HTTPException(status_code=500, detail="Failed to fetch workflow detail") from e

    @app.get("/api/execution/active", response_model=ActiveExecutionGraphsResponse)
    async def get_active_execution_graphs() -> ActiveExecutionGraphsResponse:
        """Get active execution graphs from in-memory trackers."""
        components: DaemonComponents = app.state.components

        active_trackers = await components.state.get_active_execution_trackers()
        graphs = [tracker.graph.model_dump(mode="json") for tracker in active_trackers.values()]

        return ActiveExecutionGraphsResponse(graphs=graphs, count=len(graphs))

    @app.get("/api/execution/{workflow_id}", response_model=ExecutionGraphDetailResponse)
    async def get_execution_graph(workflow_id: str) -> ExecutionGraphDetailResponse:
        """Get execution graph by workflow ID.

        Search order: active trackers → in-memory history → database

        Args:
            workflow_id: Workflow ID to fetch

        Returns:
            ExecutionGraphDetailResponse with graph data and source
        """
        from src.database.connection import get_session
        from src.database.repositories.execution_graph import ExecutionGraphRepository

        components: DaemonComponents = app.state.components

        # Check active trackers
        active_trackers = await components.state.get_active_execution_trackers()
        if workflow_id in active_trackers:
            tracker = active_trackers[workflow_id]
            return ExecutionGraphDetailResponse(
                workflow_id=workflow_id,
                graph=tracker.graph.model_dump(mode="json"),
                source="active",
            )

        # Check in-memory history
        execution_history = await components.state.get_execution_graph_history(limit=1000)
        for graph in execution_history:
            if str(graph.workflow_id) == workflow_id:
                return ExecutionGraphDetailResponse(
                    workflow_id=workflow_id,
                    graph=graph.model_dump(mode="json"),
                    source="memory",
                )

        # Check database
        try:
            async with get_session() as session:
                repo = ExecutionGraphRepository(session)
                graph = await repo.get_by_workflow_id(workflow_id)

                if graph:
                    return ExecutionGraphDetailResponse(
                        workflow_id=workflow_id,
                        graph=graph.model_dump(mode="json"),
                        source="database",
                    )
        except Exception as e:
            logger.opt(exception=True).error(f"Failed to fetch from DB: {e}")

        raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")

    @app.get("/api/execution/history", response_model=ExecutionGraphHistoryResponse)
    async def get_execution_graph_history(
        limit: int = 50,
        symbol: str | None = None,
        days: int | None = None,
    ) -> ExecutionGraphHistoryResponse:
        """Get paginated execution graph history.

        Args:
            limit: Max results (1-500)
            symbol: Filter by symbol
            days: Filter by last N days

        Returns:
            ExecutionGraphHistoryResponse with graphs and metadata
        """
        from src.database.connection import get_session
        from src.database.engine import MissingDatabaseURLError
        from src.database.repositories.execution_graph import ExecutionGraphRepository

        limit = max(1, min(limit, 500))

        try:
            async with get_session() as session:
                repo = ExecutionGraphRepository(session)

                if days:
                    from datetime import UTC, datetime, timedelta

                    start = datetime.now(UTC) - timedelta(days=days)
                    end = datetime.now(UTC)
                    graphs = await repo.get_by_date_range(start, end, symbol, limit)
                else:
                    graphs = await repo.list_recent(limit, symbol)

                return ExecutionGraphHistoryResponse(
                    graphs=[g.model_dump(mode="json") for g in graphs],
                    count=len(graphs),
                    database_enabled=True,
                )
        except MissingDatabaseURLError:
            logger.debug("Database not configured, returning empty history")
            return ExecutionGraphHistoryResponse(
                graphs=[],
                count=0,
                database_enabled=False,
            )
        except Exception as e:
            logger.opt(exception=True).error(f"Failed to fetch history: {e}")
            raise HTTPException(status_code=500, detail="Failed to fetch execution graph history") from e

    @app.websocket("/ws/events")
    async def websocket_events(websocket: WebSocket) -> None:
        """Stream real-time events to dashboard."""
        components: DaemonComponents = app.state.components

        # Accept connection first (FastAPI CORSMiddleware doesn't handle WebSocket properly)
        await websocket.accept()

        # Validate origin after accepting
        origin = websocket.headers.get("origin")
        allowed_origins = components.config.api.cors_origins

        if not components.event_bus:
            logger.warning("WebSocket rejected - EventBus not available")
            await websocket.close(code=1011, reason="EventBus not available")
            return

        # Explicitly check for None origin (security monitoring)
        if origin is None:
            logger.warning("WebSocket rejected - null origin (potential security issue)")
            await websocket.close(code=1008, reason="Invalid origin")
            return

        if origin not in allowed_origins:
            logger.warning(f"WebSocket rejected - invalid origin: {origin}")
            await websocket.close(code=1008, reason="Invalid origin")
            return

        logger.info(f"WebSocket connected from {origin}")

        subscriber_id, event_queue = await components.event_bus.subscribe()

        try:
            while True:
                try:
                    event = await asyncio.to_thread(event_queue.get, block=True, timeout=30.0)
                    event_dict = event.model_dump(mode="json")
                    await websocket.send_json(event_dict)
                except queue.Empty:
                    # Ping client to detect disconnect
                    try:
                        await websocket.send_json({"type": "ping"})
                    except WebSocketDisconnect:
                        logger.info("Client disconnected during ping")
                        break
                    except Exception as e:
                        logger.opt(exception=True).error(f"Unexpected error during ping: {e}")
                        break

        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected: {websocket.client}")
        except Exception as e:
            logger.opt(exception=True).error(f"WebSocket error: {e}")
        finally:
            await components.event_bus.unsubscribe(subscriber_id)
            logger.info(f"Unsubscribed: {subscriber_id}")

    logger.info("FastAPI app created")
    return app
