"""Signal analytics endpoints."""

from dataclasses import asdict
from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException, Query
from loguru import logger

from src.daemon.api.models import (
    AccuracyByTypeListResponse,
    AccuracyByTypeResponse,
    CalibrationBucketResponse,
    CalibrationCurveResponse,
    ExecutionRateListResponse,
    ExecutionRateResponse,
    SankeyFlowResponse,
    SignalFlowSummaryResponse,
    TimingAnalysisResponse,
)
from src.metrics.signal_analytics import SignalAnalyticsService

router = APIRouter(prefix="/api/signal-analytics", tags=["signal-analytics"])


class _ServiceHolder:
    """Holder for singleton service instance to avoid global statement."""

    instance: SignalAnalyticsService | None = None


def _get_service() -> SignalAnalyticsService:
    """Get signal analytics service singleton instance.

    Returns:
        SignalAnalyticsService instance (shared across all requests)
    """
    if _ServiceHolder.instance is None:
        _ServiceHolder.instance = SignalAnalyticsService()
    return _ServiceHolder.instance


@router.get("/summary", response_model=SignalFlowSummaryResponse)
async def get_summary(
    start_date: str = Query(..., description="Start date in ISO format (YYYY-MM-DD)"),
    end_date: str = Query(..., description="End date in ISO format (YYYY-MM-DD)"),
    horizon: str = Query("5d", description="Time horizon for profitability (1d/5d/20d)"),
) -> SignalFlowSummaryResponse:
    """Get signal flow summary for date range.

    Args:
        start_date: Start date in ISO format
        end_date: End date in ISO format
        horizon: Time horizon for profitability

    Returns:
        Signal flow summary
    """
    if horizon not in ("1d", "5d", "20d"):
        raise HTTPException(status_code=400, detail="Horizon must be '1d', '5d', or '20d'")

    try:
        start = datetime.fromisoformat(start_date).replace(tzinfo=UTC)
        end = datetime.fromisoformat(end_date).replace(hour=23, minute=59, second=59, tzinfo=UTC)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {e}") from e

    try:
        service = _get_service()
        summary = await service.get_flow_summary(start, end, horizon)

        return SignalFlowSummaryResponse(
            total_signals=summary.total_signals,
            total_buy_signals=summary.total_buy_signals,
            total_sell_signals=summary.total_sell_signals,
            execution_rate=summary.execution_rate,
            executed_count=summary.executed_count,
            not_executed_count=summary.not_executed_count,
            profitable_count=summary.profitable_count,
            unprofitable_count=summary.unprofitable_count,
            overall_accuracy=summary.overall_accuracy,
            avg_confidence=summary.avg_confidence,
            date_range=(summary.date_range[0].isoformat(), summary.date_range[1].isoformat()),
        )
    except Exception as e:
        logger.opt(exception=True).error(f"Failed to get signal summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to get signal summary") from e


@router.get("/sankey", response_model=SankeyFlowResponse)
async def get_sankey(
    start_date: str = Query(..., description="Start date in ISO format (YYYY-MM-DD)"),
    end_date: str = Query(..., description="End date in ISO format (YYYY-MM-DD)"),
    horizon: str = Query("5d", description="Time horizon for profitability (1d/5d/20d)"),
) -> SankeyFlowResponse:
    """Get Sankey flow data for signal visualization.

    Args:
        start_date: Start date in ISO format
        end_date: End date in ISO format
        horizon: Time horizon for profitability

    Returns:
        Sankey flow data (nodes and links)
    """
    if horizon not in ("1d", "5d", "20d"):
        raise HTTPException(status_code=400, detail="Horizon must be '1d', '5d', or '20d'")

    try:
        start = datetime.fromisoformat(start_date).replace(tzinfo=UTC)
        end = datetime.fromisoformat(end_date).replace(hour=23, minute=59, second=59, tzinfo=UTC)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {e}") from e

    try:
        service = _get_service()
        data = await service.get_sankey_data(start, end, horizon)

        return SankeyFlowResponse(
            nodes=[asdict(node) for node in data.nodes],
            links=[asdict(link) for link in data.links],
        )
    except Exception as e:
        logger.opt(exception=True).error(f"Failed to get Sankey data: {e}")
        raise HTTPException(status_code=500, detail="Failed to get Sankey data") from e


@router.get("/accuracy-by-type", response_model=AccuracyByTypeListResponse)
async def get_accuracy_by_type(
    start_date: str = Query(..., description="Start date in ISO format (YYYY-MM-DD)"),
    end_date: str = Query(..., description="End date in ISO format (YYYY-MM-DD)"),
    horizon: str = Query("5d", description="Time horizon (1d/5d/20d)"),
) -> AccuracyByTypeListResponse:
    """Get accuracy breakdown by signal type.

    Args:
        start_date: Start date in ISO format
        end_date: End date in ISO format
        horizon: Time horizon (1d, 5d, or 20d)

    Returns:
        Accuracy by type breakdown
    """
    if horizon not in ("1d", "5d", "20d"):
        raise HTTPException(status_code=400, detail="Horizon must be '1d', '5d', or '20d'")

    try:
        start = datetime.fromisoformat(start_date).replace(tzinfo=UTC)
        end = datetime.fromisoformat(end_date).replace(hour=23, minute=59, second=59, tzinfo=UTC)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {e}") from e

    try:
        service = _get_service()
        data = await service.get_accuracy_by_type(start, end, horizon)

        return AccuracyByTypeListResponse(
            data=[
                AccuracyByTypeResponse(
                    signal_type=d.signal_type,
                    horizon=d.horizon,
                    hit_rate=d.hit_rate,
                    executed_count=d.executed_count,
                    total_count=d.total_count,
                )
                for d in data
            ],
            count=len(data),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.opt(exception=True).error(f"Failed to get accuracy by type: {e}")
        raise HTTPException(status_code=500, detail="Failed to get accuracy by type") from e


@router.get("/calibration", response_model=CalibrationCurveResponse)
async def get_calibration(
    start_date: str = Query(..., description="Start date in ISO format (YYYY-MM-DD)"),
    end_date: str = Query(..., description="End date in ISO format (YYYY-MM-DD)"),
    horizon: str = Query("5d", description="Time horizon (1d/5d/20d)"),
) -> CalibrationCurveResponse:
    """Get calibration curve data (confidence vs actual accuracy).

    Args:
        start_date: Start date in ISO format
        end_date: End date in ISO format
        horizon: Time horizon (1d, 5d, or 20d)

    Returns:
        Calibration curve data
    """
    if horizon not in ("1d", "5d", "20d"):
        raise HTTPException(status_code=400, detail="Horizon must be '1d', '5d', or '20d'")

    try:
        start = datetime.fromisoformat(start_date).replace(tzinfo=UTC)
        end = datetime.fromisoformat(end_date).replace(hour=23, minute=59, second=59, tzinfo=UTC)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {e}") from e

    try:
        service = _get_service()
        data = await service.get_calibration_curves(start, end, horizon)

        return CalibrationCurveResponse(
            buckets=[
                CalibrationBucketResponse(
                    confidence_bucket=b.confidence_bucket,
                    expected_confidence=b.expected_confidence,
                    actual_accuracy=b.actual_accuracy,
                    sample_count=b.sample_count,
                )
                for b in data.buckets
            ]
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.opt(exception=True).error(f"Failed to get calibration data: {e}")
        raise HTTPException(status_code=500, detail="Failed to get calibration data") from e


@router.get("/timing", response_model=TimingAnalysisResponse)
async def get_timing(
    start_date: str = Query(..., description="Start date in ISO format (YYYY-MM-DD)"),
    end_date: str = Query(..., description="End date in ISO format (YYYY-MM-DD)"),
) -> TimingAnalysisResponse:
    """Get signal timing analysis (signal → execution delay).

    Args:
        start_date: Start date in ISO format
        end_date: End date in ISO format

    Returns:
        Timing analysis data
    """
    try:
        start = datetime.fromisoformat(start_date).replace(tzinfo=UTC)
        end = datetime.fromisoformat(end_date).replace(hour=23, minute=59, second=59, tzinfo=UTC)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {e}") from e

    try:
        service = _get_service()
        data = await service.get_timing_analysis(start, end)

        return TimingAnalysisResponse(
            avg_execution_delay_hours=data.avg_execution_delay_hours,
            by_confidence_bucket=data.by_confidence_bucket,
        )
    except Exception as e:
        logger.opt(exception=True).error(f"Failed to get timing analysis: {e}")
        raise HTTPException(status_code=500, detail="Failed to get timing analysis") from e


@router.get("/execution-rate", response_model=ExecutionRateListResponse)
async def get_execution_rate(
    start_date: str = Query(..., description="Start date in ISO format (YYYY-MM-DD)"),
    end_date: str = Query(..., description="End date in ISO format (YYYY-MM-DD)"),
) -> ExecutionRateListResponse:
    """Get execution rate by confidence bucket.

    Args:
        start_date: Start date in ISO format
        end_date: End date in ISO format

    Returns:
        Execution rate breakdown
    """
    try:
        start = datetime.fromisoformat(start_date).replace(tzinfo=UTC)
        end = datetime.fromisoformat(end_date).replace(hour=23, minute=59, second=59, tzinfo=UTC)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {e}") from e

    try:
        service = _get_service()
        data = await service.get_execution_rate_by_confidence(start, end)

        return ExecutionRateListResponse(
            data=[
                ExecutionRateResponse(
                    confidence_bucket=d.confidence_bucket,
                    execution_rate=d.execution_rate,
                    executed_count=d.executed_count,
                    total_count=d.total_count,
                )
                for d in data
            ],
            count=len(data),
        )
    except Exception as e:
        logger.opt(exception=True).error(f"Failed to get execution rate: {e}")
        raise HTTPException(status_code=500, detail="Failed to get execution rate") from e
