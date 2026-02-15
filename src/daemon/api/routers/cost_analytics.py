"""Cost analytics endpoints."""

from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException, Query
from loguru import logger

from src.daemon.api.models import (
    CostAnalyticsSummaryResponse,
    CostByDimensionListResponse,
    CostByDimensionResponse,
    CostTrendPointResponse,
    CostTrendsResponse,
)
from src.metrics.analytics import CostAnalyticsService

router = APIRouter(prefix="/cost-analytics", tags=["cost-analytics"])


def _get_service() -> CostAnalyticsService:
    """Get cost analytics service instance.

    Returns:
        CostAnalyticsService instance
    """
    return CostAnalyticsService()


@router.get("/summary", response_model=CostAnalyticsSummaryResponse)
async def get_summary(
    start_date: str = Query(..., description="Start date in ISO format (YYYY-MM-DD)"),
    end_date: str = Query(..., description="End date in ISO format (YYYY-MM-DD)"),
) -> CostAnalyticsSummaryResponse:
    """Get cost analytics summary for date range.

    Args:
        start_date: Start date in ISO format
        end_date: End date in ISO format

    Returns:
        Cost analytics summary
    """
    try:
        start = datetime.fromisoformat(start_date).replace(tzinfo=UTC)
        end = datetime.fromisoformat(end_date).replace(hour=23, minute=59, second=59, tzinfo=UTC)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {e}") from e

    try:
        service = _get_service()
        summary = await service.get_summary(start, end)

        return CostAnalyticsSummaryResponse(
            total_cost_usd=summary.total_cost_usd,
            total_tokens=summary.total_tokens,
            total_executions=summary.total_executions,
            avg_cost_per_execution=summary.avg_cost_per_execution,
            avg_cost_per_signal=summary.avg_cost_per_signal,
            forecast_30d_usd=summary.forecast_30d_usd,
            date_range=(summary.date_range[0].isoformat(), summary.date_range[1].isoformat()),
        )
    except Exception as e:
        logger.opt(exception=True).error(f"Failed to get cost summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to get cost summary") from e


@router.get("/trends", response_model=CostTrendsResponse)
async def get_trends(
    period: str = Query(..., description="Time period: daily or weekly"),
    start_date: str = Query(..., description="Start date in ISO format"),
    end_date: str = Query(..., description="End date in ISO format"),
) -> CostTrendsResponse:
    """Get cost trends over time.

    Args:
        period: Time period (daily or weekly)
        start_date: Start date
        end_date: End date

    Returns:
        Cost trends data
    """
    if period not in ("daily", "weekly"):
        raise HTTPException(status_code=400, detail="Period must be 'daily' or 'weekly'")

    try:
        start = datetime.fromisoformat(start_date).replace(tzinfo=UTC)
        end = datetime.fromisoformat(end_date).replace(hour=23, minute=59, second=59, tzinfo=UTC)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {e}") from e

    try:
        service = _get_service()
        trends = await service.get_trends(period, start, end)

        return CostTrendsResponse(
            trends=[
                CostTrendPointResponse(
                    timestamp=t.timestamp,
                    cost_usd=t.cost_usd,
                    tokens=t.tokens,
                    execution_count=t.execution_count,
                )
                for t in trends
            ],
            count=len(trends),
        )
    except Exception as e:
        logger.opt(exception=True).error(f"Failed to get cost trends: {e}")
        raise HTTPException(status_code=500, detail="Failed to get cost trends") from e


@router.get("/by-symbol", response_model=CostByDimensionListResponse)
async def get_by_symbol(
    start_date: str = Query(..., description="Start date in ISO format"),
    end_date: str = Query(..., description="End date in ISO format"),
) -> CostByDimensionListResponse:
    """Get cost breakdown by symbol.

    Args:
        start_date: Start date
        end_date: End date

    Returns:
        Cost by symbol breakdown
    """
    try:
        start = datetime.fromisoformat(start_date).replace(tzinfo=UTC)
        end = datetime.fromisoformat(end_date).replace(hour=23, minute=59, second=59, tzinfo=UTC)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {e}") from e

    try:
        service = _get_service()
        data = await service.get_by_symbol(start, end)

        return CostByDimensionListResponse(
            data=[
                CostByDimensionResponse(
                    dimension_value=d.dimension_value,
                    cost_usd=d.cost_usd,
                    tokens=d.tokens,
                    execution_count=d.execution_count,
                    percentage=d.percentage,
                )
                for d in data
            ],
            count=len(data),
        )
    except Exception as e:
        logger.opt(exception=True).error(f"Failed to get cost by symbol: {e}")
        raise HTTPException(status_code=500, detail="Failed to get cost by symbol") from e


@router.get("/by-agent", response_model=CostByDimensionListResponse)
async def get_by_agent(
    start_date: str = Query(..., description="Start date in ISO format"),
    end_date: str = Query(..., description="End date in ISO format"),
) -> CostByDimensionListResponse:
    """Get cost breakdown by agent.

    Args:
        start_date: Start date
        end_date: End date

    Returns:
        Cost by agent breakdown
    """
    try:
        start = datetime.fromisoformat(start_date).replace(tzinfo=UTC)
        end = datetime.fromisoformat(end_date).replace(hour=23, minute=59, second=59, tzinfo=UTC)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {e}") from e

    try:
        service = _get_service()
        data = await service.get_by_agent(start, end)

        return CostByDimensionListResponse(
            data=[
                CostByDimensionResponse(
                    dimension_value=d.dimension_value,
                    cost_usd=d.cost_usd,
                    tokens=d.tokens,
                    execution_count=d.execution_count,
                    percentage=d.percentage,
                )
                for d in data
            ],
            count=len(data),
        )
    except Exception as e:
        logger.opt(exception=True).error(f"Failed to get cost by agent: {e}")
        raise HTTPException(status_code=500, detail="Failed to get cost by agent") from e


@router.get("/by-model", response_model=CostByDimensionListResponse)
async def get_by_model(
    start_date: str = Query(..., description="Start date in ISO format"),
    end_date: str = Query(..., description="End date in ISO format"),
) -> CostByDimensionListResponse:
    """Get cost breakdown by model.

    Args:
        start_date: Start date
        end_date: End date

    Returns:
        Cost by model breakdown
    """
    try:
        start = datetime.fromisoformat(start_date).replace(tzinfo=UTC)
        end = datetime.fromisoformat(end_date).replace(hour=23, minute=59, second=59, tzinfo=UTC)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {e}") from e

    try:
        service = _get_service()
        data = await service.get_by_model(start, end)

        return CostByDimensionListResponse(
            data=[
                CostByDimensionResponse(
                    dimension_value=d.dimension_value,
                    cost_usd=d.cost_usd,
                    tokens=d.tokens,
                    execution_count=d.execution_count,
                    percentage=d.percentage,
                )
                for d in data
            ],
            count=len(data),
        )
    except Exception as e:
        logger.opt(exception=True).error(f"Failed to get cost by model: {e}")
        raise HTTPException(status_code=500, detail="Failed to get cost by model") from e
