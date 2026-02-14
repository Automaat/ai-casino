"""Paper trading validation endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Request
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from src.daemon.api.dependencies import get_db_session
from src.daemon.api.models import PaperTradingValidationResponse, ValidationCriterionResponse
from src.daemon.api.routers.shared import get_components

router = APIRouter(tags=["validation"])


@router.get("/validation/paper-trading", response_model=PaperTradingValidationResponse)
async def get_paper_trading_validation(
    request: Request, session: AsyncSession = Depends(get_db_session)
) -> PaperTradingValidationResponse:
    """Get paper trading validation status and progress toward live promotion.

    Returns:
        Paper trading validation status with criteria progress
    """
    components = get_components(request)

    # Create metrics tracker for this request (avoid mutating shared component)
    from src.database.repositories.trade import TradeRepository
    from src.metrics.tracker import create_metrics_tracker

    trade_repo = TradeRepository(session)
    metrics_tracker = create_metrics_tracker(trade_repository=trade_repo)

    # Create validator
    from src.daemon.paper_trading_validator import PaperTradingValidator

    validator = PaperTradingValidator(
        config=components.config.paper_trading,
        state=components.state,
        metrics_tracker=metrics_tracker,
    )

    # Assess readiness
    try:
        report = await validator.assess_readiness()

        return PaperTradingValidationResponse(
            ready_for_live=report.ready_for_live,
            assessment_date=report.assessment_date,
            paper_trading_duration_days=report.paper_trading_duration_days,
            total_paper_trades=report.total_paper_trades,
            criteria=[
                ValidationCriterionResponse(
                    name=c.name,
                    passed=c.passed,
                    current_value=c.current_value,
                    threshold=c.threshold,
                    message=c.message,
                )
                for c in report.criteria
            ],
            recommendations=report.recommendations,
        )
    except Exception as e:
        logger.opt(exception=True).error(f"Failed to assess paper trading readiness: {e}")
        raise HTTPException(status_code=500, detail="Failed to assess paper trading readiness") from e
