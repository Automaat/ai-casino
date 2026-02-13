"""Trading and analysis endpoints."""

import asyncio
import json
from pathlib import Path

from fastapi import APIRouter, Request
from loguru import logger

from src.daemon.api.models import (
    AnalysesResponse,
    AnalysisRecordResponse,
    GamePlanResponse,
    WatchlistResponse,
)
from src.daemon.api.routers.shared import get_components

router = APIRouter(tags=["trading"])


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
async def get_watchlist(request: Request) -> WatchlistResponse:
    """Get merged watchlist."""
    components = get_components(request)

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
