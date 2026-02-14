"""State and event endpoints."""

from fastapi import APIRouter, Request
from loguru import logger

from src.daemon.api.models import (
    DegradationHistoryResponse,
    DegradationResponse,
    EventResponse,
    MarketEventsResponse,
    StateSummaryResponse,
)
from src.daemon.api.routers.shared import get_components

router = APIRouter(tags=["state"])


@router.get("/state/summary", response_model=StateSummaryResponse)
async def state_summary(request: Request) -> StateSummaryResponse:
    """Get daemon state summary."""
    components = get_components(request)

    try:
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
    except Exception as e:
        # DB errors - return minimal safe response
        logger.opt(exception=True).warning(f"Failed to fetch state summary: {e}")
        return StateSummaryResponse(
            total_analyses=0,
            recent_analyses=[],
            total_trades=0,
            positions_count=0,
            win_rate=None,
            error_count=0,
            degradation_tier="FULL",
            trading_mode=components.config.trading_mode.value,
        )


@router.get("/degradation", response_model=DegradationResponse)
async def get_degradation(request: Request) -> DegradationResponse:
    """Get current degradation status."""
    components = get_components(request)

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


@router.get("/events", response_model=EventResponse)
async def get_events(request: Request, limit: int = 100) -> EventResponse:
    """Get event history."""
    components = get_components(request)

    if not components.event_bus:
        return EventResponse(events=[], returned_count=0)

    limit = max(0, min(limit, 500))

    events = components.event_bus.get_history(limit=limit)

    events_dict = [e.model_dump(mode="json") for e in events]

    return EventResponse(events=events_dict, returned_count=len(events_dict))


@router.get("/events/market", response_model=MarketEventsResponse)
async def get_market_events(request: Request, limit: int = 100) -> MarketEventsResponse:
    """Get market event signals (news, social, anomaly)."""
    components = get_components(request)
    limit = max(0, min(limit, 500))

    if limit <= 0:
        events = []
    else:
        market_events = await components.state.get_market_events(limit=limit)
        events = market_events

    return MarketEventsResponse(events=events, returned_count=len(events))


@router.get("/events/degradation-history", response_model=DegradationHistoryResponse)
async def get_degradation_history(request: Request, limit: int = 50) -> DegradationHistoryResponse:
    """Get degradation history for timeline."""
    components = get_components(request)
    limit = max(0, min(limit, 200))

    if limit <= 0:
        history = []
    else:
        history = await components.state.get_degradation_history(limit=limit)

    return DegradationHistoryResponse(
        records=[r.model_dump(mode="json") for r in history],
        count=len(history),
    )
