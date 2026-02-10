"""Risk assessment stage implementation."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from src.agents.risk import RiskManagementAgent

from src.agents.risk import RiskAssessment
from src.workflows.models.risk import RiskAssessmentInput, RiskAssessmentOutput


async def assess_risk(
    input_data: RiskAssessmentInput,
    risk_manager: "RiskManagementAgent",
) -> RiskAssessmentOutput:
    """Assess risk for trading decision (async, thread-offloaded).

    Args:
        input_data: Risk assessment input with decision, account info, and market data
        risk_manager: Risk management agent

    Returns:
        RiskAssessmentOutput with risk assessment
    """
    logger.info("Assessing risk for trading decision")

    # Extract daily data and current price using helpers
    daily_data = input_data.get_daily_data()
    current_price = input_data.get_current_price()

    # Ensure account_info is present
    if input_data.account_info is None:
        msg = "account_info is None, cannot assess risk"
        raise ValueError(msg)

    def _sync_assess_risk() -> RiskAssessment:
        """Synchronous risk assessment wrapper for thread execution."""
        return risk_manager.assess(
            symbol=input_data.symbol,
            action=input_data.final_decision.action,
            current_price=current_price,
            account_info=input_data.account_info,  # type: ignore[arg-type]
            market_data=daily_data,
            decision_confidence=input_data.final_decision.confidence,
            broker_positions=input_data.broker_positions,
            portfolio_value=input_data.portfolio_value,
            target_portfolio_weight=input_data.target_portfolio_weight,
            backtest_validation=input_data.backtest_validation,
            degradation_context=input_data.degradation_context,
            broker_api_failed=input_data.broker_api_failed,
        )

    risk_assessment = await asyncio.to_thread(_sync_assess_risk)
    return RiskAssessmentOutput(risk_assessment=risk_assessment)
