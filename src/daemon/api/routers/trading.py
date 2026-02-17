"""Trading and analysis endpoints."""

import asyncio
import json
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, Request
from loguru import logger
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from src.daemon.api.dependencies import get_db_session
from src.daemon.api.models import (
    ActiveDiscoveryCandidate,
    ActiveDiscoveryResponse,
    ActiveDiscoverySourceDetail,
    AnalysesResponse,
    AnalysisRecordResponse,
    DiscoveryInsightsResponse,
    DiscoveryRecord,
    DiscoverySourceBreakdown,
    DiscoverySuccessMetrics,
    EnrichedTradeResponse,
    GamePlanResponse,
    TradeResponse,
    TradesResponse,
    WatchlistResponse,
)
from src.daemon.api.routers.shared import get_components

router = APIRouter(tags=["trading"])


class TradeQueryParams(BaseModel):
    """Trade query filter parameters."""

    limit: int = 100
    symbol: str | None = None
    status: str | None = None
    window: Literal["all", "30d", "7d"] = "all"


@router.get("/analyses", response_model=AnalysesResponse)
async def get_analyses(request: Request, limit: int = 50, symbol: str | None = None) -> AnalysesResponse:
    """Get analysis history."""
    components = get_components(request)
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


@router.get("/watchlist", response_model=WatchlistResponse)
async def get_watchlist(
    request: Request, session: AsyncSession = Depends(get_db_session)
) -> WatchlistResponse:
    """Get merged watchlist."""
    components = get_components(request)

    symbols = await components.broker_manager.get_merged_watchlist(session=session)

    config_count = len([s for s in components.config.watchlist if s in symbols])

    broker_count = 0
    try:
        active_positions = await components.state.get_active_positions()
        broker_symbols = set(dict(active_positions).keys())
        broker_count = len([s for s in broker_symbols if s in symbols])
    except Exception as e:
        logger.opt(exception=True).warning(f"Unable to derive broker symbols for watchlist: {e}")

    screening_count = 0
    screening_history = await components.state.get_screening_history(limit=1, session=session)
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


@router.get("/game-plan", response_model=GamePlanResponse | None)
async def get_game_plan(request: Request) -> GamePlanResponse | None:
    """Get latest game plan (if enabled and generated)."""
    components = get_components(request)

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


@router.get("/trades", response_model=TradesResponse)
async def get_trades(
    request: Request,
    session: AsyncSession = Depends(get_db_session),
    params: TradeQueryParams = Depends(),
) -> TradesResponse:
    """Get trade history.

    Args:
        request: FastAPI request object
        session: Database session
        params: Query parameters for filtering

    Returns:
        TradesResponse with filtered trades
    """
    components = get_components(request)
    limit = max(0, min(params.limit, 500))

    from src.database.repositories.trade import TradeRepository

    repo = TradeRepository(session)

    # Get trades by window
    trades = await repo.get_by_window(params.window)

    # Apply filters
    if params.symbol:
        trades = [t for t in trades if t.symbol == params.symbol]
    if params.status:
        trades = [t for t in trades if t.status == params.status.upper()]

    # Apply limit
    trades = trades[:limit]

    # Get total count
    total_count = await repo.count_all()

    return TradesResponse(
        trades=[
            TradeResponse(
                id=str(t.id),
                timestamp=t.timestamp,
                symbol=t.symbol,
                action=t.action.value,
                entry_price=t.entry_price,
                exit_price=t.exit_price,
                shares=t.shares,
                confidence=t.confidence,
                risk_level=t.risk_level,
                status=t.status,
                pnl=t.pnl,
                pnl_percent=t.pnl_percent,
                strategy_name=t.strategy_name,
                is_paper_trade=t.is_paper_trade,
                closed_at=t.closed_at,
            )
            for t in trades
        ],
        total_count=total_count,
        returned_count=len(trades),
        database_enabled=components.config.database.enable_persistence,
    )


@router.get("/trades/{trade_id}", response_model=EnrichedTradeResponse)
async def get_trade_detail(
    trade_id: str, session: AsyncSession = Depends(get_db_session)
) -> EnrichedTradeResponse:
    """Get single trade with analysis reasoning.

    Args:
        trade_id: Trade UUID
        session: Database session

    Returns:
        EnrichedTradeResponse with trade and matched analysis

    Raises:
        HTTPException: 404 if trade not found
    """
    from src.database.repositories.analysis import AnalysisRecordRepository
    from src.database.repositories.trade import TradeRepository

    trade_repo = TradeRepository(session)
    analysis_repo = AnalysisRecordRepository(session)

    trade = await trade_repo.get_by_id(trade_id)
    if not trade:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail=f"Trade {trade_id} not found")

    # Find matching analysis (same symbol, timestamp within 1 minute)
    analyses = await analysis_repo.get_by_symbol(trade.symbol)
    matched_analysis = None
    matched_time_diff: float | None = None

    for analysis in analyses:
        time_diff = abs((analysis.timestamp - trade.timestamp).total_seconds())
        if time_diff <= 60 and (matched_time_diff is None or time_diff < matched_time_diff):
            matched_analysis = analysis
            matched_time_diff = time_diff

    return EnrichedTradeResponse(
        trade=TradeResponse(
            id=str(trade.id),
            timestamp=trade.timestamp,
            symbol=trade.symbol,
            action=trade.action.value,
            entry_price=trade.entry_price,
            exit_price=trade.exit_price,
            shares=trade.shares,
            confidence=trade.confidence,
            risk_level=trade.risk_level,
            status=trade.status,
            pnl=trade.pnl,
            pnl_percent=trade.pnl_percent,
            strategy_name=trade.strategy_name,
            is_paper_trade=trade.is_paper_trade,
            closed_at=trade.closed_at,
        ),
        analysis=(
            AnalysisRecordResponse(
                symbol=matched_analysis.symbol,
                timestamp=matched_analysis.timestamp,
                signal=matched_analysis.signal,
                confidence=matched_analysis.confidence,
                executed_trade=matched_analysis.executed_trade,
                trading_session=matched_analysis.trading_session.value,
                is_paper_trade=matched_analysis.is_paper_trade,
                rsi=matched_analysis.rsi,
                macd_hist=matched_analysis.macd_hist,
                reasoning=matched_analysis.reasoning,
                technical_analysis_reasoning=matched_analysis.technical_analysis_reasoning,
                sentiment_analysis_reasoning=matched_analysis.sentiment_analysis_reasoning,
                news_analysis_reasoning=matched_analysis.news_analysis_reasoning,
            )
            if matched_analysis
            else None
        ),
    )


@router.get("/discovery/insights", response_model=DiscoveryInsightsResponse)
async def get_discovery_insights(
    request: Request, session: AsyncSession = Depends(get_db_session)
) -> DiscoveryInsightsResponse:
    """Get discovery insights dashboard data (last 30 days, up to 1000 records)."""
    components = get_components(request)

    # Fetch discovery history from last 30 days
    discoveries = await components.state.get_discovery_history(limit=1000, session=session)

    if not discoveries:
        return DiscoveryInsightsResponse(
            source_breakdown=[],
            success_metrics=DiscoverySuccessMetrics(
                total_discovered=0, added_to_watchlist=0, received_signal=0, signal_rate=0.0
            ),
            recent_discoveries=[],
            avg_composite_score=0.0,
            total_discoveries=0,
        )

    # Calculate source breakdown
    source_counts: dict[str, int] = {}
    for discovery in discoveries:
        for source in discovery.sources:
            source_name = source.value if hasattr(source, "value") else str(source)
            source_counts[source_name] = source_counts.get(source_name, 0) + 1

    total_source_occurrences = sum(source_counts.values())
    source_breakdown = [
        DiscoverySourceBreakdown(
            source=source,
            count=count,
            percentage=(
                round((count / total_source_occurrences) * 100, 1) if total_source_occurrences > 0 else 0.0
            ),
        )
        for source, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True)
    ]

    # Calculate success metrics
    total_discovered = len(discoveries)
    added_to_watchlist = sum(1 for d in discoveries if d.added_to_watchlist)
    received_signal = sum(1 for d in discoveries if d.first_signal is not None)
    signal_rate = round((received_signal / total_discovered) * 100, 1) if total_discovered > 0 else 0.0

    success_metrics = DiscoverySuccessMetrics(
        total_discovered=total_discovered,
        added_to_watchlist=added_to_watchlist,
        received_signal=received_signal,
        signal_rate=signal_rate,
    )

    # Recent discoveries with outcomes
    recent_discoveries = [
        DiscoveryRecord(
            symbol=d.symbol,
            discovered_at=d.discovered_at,
            composite_score=d.composite_score,
            sources=[s.value if hasattr(s, "value") else str(s) for s in d.sources],
            added_to_watchlist=d.added_to_watchlist,
            first_signal=d.first_signal,
            first_signal_date=d.first_signal_date,
            outcome_7d=d.outcome_7d,
            outcome_30d=d.outcome_30d,
        )
        for d in sorted(discoveries, key=lambda x: x.discovered_at, reverse=True)[:50]
    ]

    # Calculate average composite score
    avg_composite_score = round(sum(d.composite_score for d in discoveries) / len(discoveries), 3)

    return DiscoveryInsightsResponse(
        source_breakdown=source_breakdown,
        success_metrics=success_metrics,
        recent_discoveries=recent_discoveries,
        avg_composite_score=avg_composite_score,
        total_discoveries=total_discovered,
    )


@router.get("/discovery/active", response_model=ActiveDiscoveryResponse)
async def get_active_discovery(
    request: Request,
    session: AsyncSession = Depends(get_db_session),
    source_filter: Literal["all", "batch", "continuous"] = "all",
) -> ActiveDiscoveryResponse:
    """Get active discovery candidates (currently tracked with TTL).

    Args:
        request: FastAPI request
        session: Database session
        source_filter: Filter by source type (all, batch, continuous)

    Returns:
        ActiveDiscoveryResponse with filtered candidates
    """
    from datetime import UTC, datetime

    components = get_components(request)

    # Get active discovery candidates
    active_candidates = await components.state.get_active_discovery_candidates(session=session)

    # Define source categories
    batch_sources = {"technical_screening", "earnings_upcoming", "sector_rotation"}
    continuous_sources = {"reddit_trending", "volume_spike", "price_gap", "news_trending", "pre_market"}

    # Filter candidates by source type
    if source_filter == "batch":
        active_candidates = [
            c for c in active_candidates if any(s.value in batch_sources for s in c.sources)
        ]
    elif source_filter == "continuous":
        active_candidates = [
            c for c in active_candidates if any(s.value in continuous_sources for s in c.sources)
        ]

    # Get last discovery timestamp
    last_discovery = await components.state.get_last_discovery(session=session)

    # Convert to response models
    now = datetime.now(UTC)
    candidates = []
    for candidate in active_candidates:
        time_remaining = int((candidate.ttl_expires_at - now).total_seconds() / 60)

        sources = [
            ActiveDiscoverySourceDetail(
                source_type=str(source.value),
                weight=candidate.source_scores.get(str(source.value), 0.0),
            )
            for source in candidate.sources
        ]

        candidates.append(
            ActiveDiscoveryCandidate(
                symbol=candidate.symbol,
                discovered_at=candidate.discovery_timestamp,
                composite_score=candidate.composite_score,
                sources=sources,
                ttl_expires_at=candidate.ttl_expires_at,
                time_remaining_minutes=max(0, time_remaining),
            )
        )

    return ActiveDiscoveryResponse(
        candidates=candidates,
        count=len(candidates),
        last_discovery=last_discovery,
    )
