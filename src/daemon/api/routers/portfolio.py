"""Portfolio, positions, and risk endpoints."""

from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Request
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from src.daemon.api.dependencies import get_db_session
from src.daemon.api.models import (
    CorrelationMatrixResponse,
    MetricsSnapshot,
    PositionManagementActionResponse,
    PositionResponse,
    PositionsResponse,
    PositionTimelineResponse,
    RebalanceAllocation,
    RebalanceCalculation,
    RebalanceHistoryEntry,
    RebalanceResponse,
    RebalancingHistoryResponse,
    RiskHistoryResponse,
    RiskReportResponse,
    SectorAttributionResponse,
    SectorContributionDetail,
    SectorRotationResponse,
    SnapshotRecord,
    SnapshotsResponse,
)
from src.daemon.api.routers.shared import get_broker_account_info_cached, get_components

router = APIRouter(tags=["portfolio"])


@router.get("/positions", response_model=PositionsResponse)
async def get_positions(request: Request) -> PositionsResponse:
    """Get active positions."""
    components = get_components(request)

    from src.daemon.positions import PositionRecord

    async with get_broker_account_info_cached(components) as account_info:
        broker_positions = account_info["positions"] if account_info else {}
        account_equity = account_info["balance"] if account_info else None

        active_positions = await components.state.get_active_positions()
        positions = []
        for symbol, pos_dict in dict(active_positions).items():
            try:
                pos = PositionRecord.model_validate(pos_dict)

                current_price = pos.entry_price
                broker_unrealized_pnl = None
                if symbol in broker_positions:
                    broker_pos = broker_positions[symbol]
                    current_price = (
                        broker_pos.market_value / broker_pos.qty if broker_pos.qty > 0 else pos.entry_price
                    )
                    broker_unrealized_pnl = broker_pos.unrealized_pnl

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
                        broker_unrealized_pnl=broker_unrealized_pnl,
                    )
                )
            except Exception as e:
                logger.opt(exception=True).error(f"Failed to parse position {symbol}: {e}")
                continue

        return PositionsResponse(positions=positions, count=len(positions), account_equity=account_equity)


@router.get("/portfolio/snapshots", response_model=SnapshotsResponse)
async def get_snapshots(request: Request, days: int = 30) -> SnapshotsResponse:
    """Get portfolio snapshots history."""
    from src.database.connection import get_session
    from src.database.engine import MissingDatabaseURLError
    from src.database.repositories.snapshot import PortfolioSnapshotRepository

    components = get_components(request)

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


@router.get("/portfolio/rebalance", response_model=RebalanceResponse)
async def get_rebalance(
    request: Request, session: AsyncSession = Depends(get_db_session)
) -> RebalanceResponse:
    """Get latest portfolio rebalance data."""
    components = get_components(request)

    # Check if rebalancing is enabled
    rebalancing_enabled = components.config.rebalancing.enabled

    # If disabled or no history, return status-only response
    rebalancing_history = await components.state.get_rebalancing_history(limit=1, session=session)
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


@router.get("/portfolio/rebalancing/history", response_model=RebalancingHistoryResponse)
async def get_rebalancing_history(
    request: Request, session: AsyncSession = Depends(get_db_session)
) -> RebalancingHistoryResponse:
    """Get rebalancing history with deviation metrics."""
    components = get_components(request)

    # Check if rebalancing is enabled
    rebalancing_enabled = components.config.rebalancing.enabled
    rebalance_threshold = components.config.rebalancing.rebalance_threshold

    # Get current portfolio value
    current_portfolio_value = 0.0
    async with get_broker_account_info_cached(components) as account_info:
        current_portfolio_value = account_info["portfolio_value"] if account_info else 0.0
        broker_positions = account_info["positions"] if account_info else {}

    # Get current metrics from previous rebalancing record if exists
    current_metrics = None
    prior_records = await components.state.get_rebalancing_history(limit=2, session=session)
    if len(prior_records) >= 2:
        # Records are desc by timestamp, so index 1 is the previous record
        prior_record = prior_records[1]
        current_metrics = MetricsSnapshot(
            expected_return=prior_record.expected_return,
            expected_volatility=prior_record.expected_volatility,
            sharpe_ratio=prior_record.sharpe_ratio,
        )

    # If disabled, return status-only response
    if not rebalancing_enabled:
        return RebalancingHistoryResponse(
            enabled=False,
            current_portfolio_value=current_portfolio_value,
            rebalance_threshold=rebalance_threshold,
            current_metrics=current_metrics,
        )

    # Get rebalancing history
    rebalancing_records = await components.state.get_rebalancing_history(limit=30, session=session)

    if not rebalancing_records:
        return RebalancingHistoryResponse(
            enabled=True,
            current_portfolio_value=current_portfolio_value,
            rebalance_threshold=rebalance_threshold,
            current_metrics=current_metrics,
        )

    # Build latest calculation with full allocations (records are desc by timestamp)
    latest_record = rebalancing_records[0]
    total_portfolio_value = current_portfolio_value

    allocations = []
    for allocation in latest_record.allocations:
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

    latest = RebalanceCalculation(
        timestamp=latest_record.timestamp,
        method=latest_record.method,
        allocations=allocations,
        expected_return=latest_record.expected_return,
        expected_volatility=latest_record.expected_volatility,
        sharpe_ratio=latest_record.sharpe_ratio,
    )

    # Build history with deviation metrics
    history = []
    for record in rebalancing_records:
        # Calculate deviation metrics from allocations
        deviations = [abs(alloc.delta) for alloc in record.allocations]
        avg_deviation = sum(deviations) / len(deviations) if deviations else 0.0
        max_deviation = max(deviations) if deviations else 0.0

        history.append(
            RebalanceHistoryEntry(
                timestamp=record.timestamp,
                method=record.method,
                avg_deviation_pct=avg_deviation * 100,
                max_deviation_pct=max_deviation * 100,
                metrics=MetricsSnapshot(
                    expected_return=record.expected_return,
                    expected_volatility=record.expected_volatility,
                    sharpe_ratio=record.sharpe_ratio,
                ),
            )
        )

    return RebalancingHistoryResponse(
        enabled=True,
        current_portfolio_value=current_portfolio_value,
        rebalance_threshold=rebalance_threshold,
        current_metrics=current_metrics,
        latest=latest,
        history=history,
    )


@router.get("/risk", response_model=RiskReportResponse | None)
async def get_risk(
    request: Request, session: AsyncSession = Depends(get_db_session)
) -> RiskReportResponse | None:
    """Get latest risk report."""
    components = get_components(request)

    risk_history = await components.state.get_risk_report_history(limit=1, session=session)
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


@router.get("/risk/history", response_model=RiskHistoryResponse)
async def get_risk_history(
    request: Request, session: AsyncSession = Depends(get_db_session)
) -> RiskHistoryResponse:
    """Get historical risk reports."""
    components = get_components(request)
    all_reports = await components.state.get_risk_report_history(limit=1000, session=session)
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


@router.get("/sector-rotation/latest", response_model=SectorRotationResponse | None)
async def get_sector_rotation(
    request: Request, session: AsyncSession = Depends(get_db_session)
) -> SectorRotationResponse | None:
    """Get latest sector rotation analysis."""
    components = get_components(request)
    rotation_history = await components.state.get_sector_rotation_history(limit=1, session=session)
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


@router.get("/sector-attribution/latest", response_model=SectorAttributionResponse | None)
async def get_sector_attribution(
    request: Request, session: AsyncSession = Depends(get_db_session)
) -> SectorAttributionResponse | None:
    """Get latest sector attribution analysis."""
    components = get_components(request)
    record = await components.state.get_sector_attribution_latest(session=session)
    if not record:
        return None

    contributions = [
        SectorContributionDetail(
            sector=str(c["sector"]),
            sector_etf=str(c["sector_etf"]),
            total_value=float(c["total_value"]),
            portfolio_weight=float(c["portfolio_weight"]),
            benchmark_weight=float(c["benchmark_weight"]),
            over_under_weight=float(c["over_under_weight"]),
            pnl=float(c["pnl"]),
            return_pct=float(c["return_pct"]),
            position_count=int(c["position_count"]),
        )
        for c in record.contributions
    ]

    return SectorAttributionResponse(
        timestamp=record.timestamp,
        contributions=contributions,
        total_portfolio_value=record.total_portfolio_value,
        benchmark_name=record.benchmark_name,
    )


@router.get("/correlation/latest", response_model=CorrelationMatrixResponse | None)
async def get_correlation_matrix(request: Request) -> CorrelationMatrixResponse | None:
    """Get latest correlation matrix."""
    from src.metrics.correlation import CorrelationAuditor

    components = get_components(request)

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


@router.get("/positions/{symbol}/timeline", response_model=PositionTimelineResponse)
async def get_position_timeline(symbol: str, request: Request) -> PositionTimelineResponse:
    """Get position timeline with management actions."""
    components = get_components(request)

    # Check database enabled
    database_enabled = components.config.database.enable_persistence

    # If database disabled, return empty timeline
    if not database_enabled:
        # Still need position from state for basic info
        position = await components.state.positions.get_position(symbol)
        if not position:
            raise HTTPException(status_code=404, detail=f"Position {symbol} not found")

        return PositionTimelineResponse(
            symbol=position.symbol,
            entry_price=position.entry_price,
            current_price=position.entry_price,
            current_qty=position.current_qty,
            entry_timestamp=position.entry_timestamp,
            days_held=position.days_held,
            actions=[],
            count=0,
            database_enabled=False,
        )

    # Get position from state
    position = await components.state.positions.get_position(symbol)
    if not position:
        raise HTTPException(status_code=404, detail=f"Position {symbol} not found")

    # Get current price from broker
    current_price = position.entry_price
    async with get_broker_account_info_cached(components) as account_info:
        if account_info and symbol in account_info["positions"]:
            broker_pos = account_info["positions"][symbol]
            current_price = (
                broker_pos.market_value / broker_pos.qty if broker_pos.qty > 0 else position.entry_price
            )

    # Get actions
    actions = await components.state.positions.get_recent_actions(symbol=symbol, limit=500)

    # Convert to response models
    action_responses = [
        PositionManagementActionResponse(
            action_type=action.action_type,
            timestamp=action.timestamp,
            old_stop_loss=action.old_stop_loss,
            new_stop_loss=action.new_stop_loss,
            qty_sold=action.qty_sold,
            price=action.price,
            reason=action.reason,
            executed=action.executed,
            order_id=action.order_id,
        )
        for action in actions
    ]

    return PositionTimelineResponse(
        symbol=position.symbol,
        entry_price=position.entry_price,
        current_price=current_price,
        current_qty=position.current_qty,
        entry_timestamp=position.entry_timestamp,
        days_held=position.days_held,
        actions=action_responses,
        count=len(action_responses),
        database_enabled=database_enabled,
    )
