"""Screening endpoints."""

from collections import defaultdict

from fastapi import APIRouter, Depends, Query, Request
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from src.daemon.api.dependencies import get_db_session
from src.daemon.api.models import (
    ScreeningCandidateResponse,
    ScreeningHistoryResponse,
    ScreeningInsightsResponse,
    ScreeningRecordResponse,
)
from src.daemon.api.routers.shared import get_components

router = APIRouter(tags=["screening"], prefix="/screening")


@router.get("/history", response_model=ScreeningHistoryResponse)
async def get_screening_history(
    request: Request,
    session: AsyncSession = Depends(get_db_session),
    limit: int = Query(default=30, ge=1, le=100),
) -> ScreeningHistoryResponse:
    """Get screening history with candidates."""
    components = get_components(request)
    records = await components.state.get_screening_history(limit=limit, session=session)

    if not records:
        return ScreeningHistoryResponse(
            records=[],
            total_count=0,
            latest_screening=None,
        )

    response_records = []
    for record in records:
        if not record.id:
            logger.warning(f"Screening record missing id: {record.timestamp}")
            continue

        candidates = [
            ScreeningCandidateResponse(
                symbol=c.symbol,
                name=c.name,
                sector=c.sector,
                score=c.score,
                signal=c.signal.value if hasattr(c.signal, "value") else str(c.signal),
                metrics=c.metrics,
                reason=c.reason,
            )
            for c in record.candidates
        ]

        response_records.append(
            ScreeningRecordResponse(
                id=record.id,
                timestamp=record.timestamp,
                criteria=record.criteria,
                universe=record.universe,
                top_symbols=record.top_symbols,
                candidates=candidates,
                screened_at=record.screened_at,
                candidate_count=len(candidates),
            )
        )

    latest_screening = max(r.timestamp for r in records) if records else None

    return ScreeningHistoryResponse(
        records=response_records,
        total_count=len(response_records),
        latest_screening=latest_screening,
    )


@router.get("/insights", response_model=ScreeningInsightsResponse)
async def get_screening_insights(
    request: Request,
    session: AsyncSession = Depends(get_db_session),
) -> ScreeningInsightsResponse:
    """Get screening analytics and insights."""
    components = get_components(request)
    records = await components.state.get_screening_history(limit=100, session=session)

    if not records:
        return ScreeningInsightsResponse(
            total_screenings=0,
            latest_screening_date=None,
            criteria_breakdown={},
            sector_distribution={},
            avg_score=0.0,
            top_signals={},
        )

    # Criteria breakdown
    criteria_breakdown: dict[str, int] = defaultdict(int)
    for record in records:
        criteria_breakdown[record.criteria] += 1

    # Latest screening for sector distribution and signals
    latest = records[-1]
    sector_counts: dict[str, int] = defaultdict(int)
    signal_counts: dict[str, int] = defaultdict(int)
    total_score = 0.0

    for candidate in latest.candidates:
        sector_counts[candidate.sector] += 1
        signal = candidate.signal.value if hasattr(candidate.signal, "value") else str(candidate.signal)
        signal_counts[signal] += 1
        total_score += candidate.score

    # Top 5 sectors
    sorted_sectors = sorted(sector_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    sector_distribution = dict(sorted_sectors)

    # Avg score
    avg_score = round(total_score / len(latest.candidates), 3) if latest.candidates else 0.0

    return ScreeningInsightsResponse(
        total_screenings=len(records),
        latest_screening_date=latest.timestamp,
        criteria_breakdown=dict(criteria_breakdown),
        sector_distribution=sector_distribution,
        avg_score=avg_score,
        top_signals=dict(signal_counts),
    )
