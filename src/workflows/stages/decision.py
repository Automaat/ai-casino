"""Decision stage implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from src.agents.trader import TraderAgent
    from src.metrics.execution import ExecutionMetricsCollector

from src.workflows.models.decision import DecisionInput, DecisionOutput
from src.workflows.stages.strategy_selection import _timed_agent_call


async def make_decision(
    input_data: DecisionInput,
    trader: TraderAgent,
    collector: ExecutionMetricsCollector | None = None,
) -> DecisionOutput:
    """Make final trading decision.

    Args:
        input_data: Decision input with all analyses and context
        trader: Trader agent for decision making
        collector: Optional metrics collector

    Returns:
        DecisionOutput with final decision
    """
    from src.agents.trader import TradingDecision
    from src.strategies.signal import Signal

    logger.info("Making final trading decision")

    # Check for missing critical analyses
    missing_critical = []
    if not input_data.technical:
        missing_critical.append("technical")
    if not input_data.sentiment:
        missing_critical.append("sentiment")
    if not input_data.news:
        missing_critical.append("news")

    missing_research = []
    if not input_data.bullish:
        missing_research.append("bullish")
    if not input_data.bearish:
        missing_research.append("bearish")

    # If critical analyses missing, return conservative HOLD decision
    if missing_critical or missing_research:
        missing_str = ", ".join(missing_critical + missing_research)
        logger.warning(
            f"Cannot make informed decision for {input_data.symbol}: missing analyses ({missing_str}). "
            "Returning HOLD with degraded confidence."
        )
        # Extract position info for correct display_action (WAIT vs HOLD)
        positions = input_data.account_info.positions if input_data.account_info else {}
        position_qty = positions.get(input_data.symbol)

        return DecisionOutput(
            final_decision=TradingDecision(
                action=Signal.HOLD,
                confidence=0.1,
                risk_level="HIGH",
                reasoning=[
                    f"Insufficient data to make informed trading decision: missing {missing_str} analyses",
                    "Supervisor routing skipped these analyses due to data unavailability",
                    "Conservative HOLD recommended until data becomes available",
                ],
                owns_position=input_data.owns_position,
                position_qty=position_qty,
            )
        )

    # Extract account info for position awareness
    positions = input_data.account_info.positions if input_data.account_info else {}
    position_qty = positions.get(input_data.symbol)

    # Call trader agent with all inputs
    decision = await _timed_agent_call(
        "trader",
        trader.decide(
            input_data.symbol,
            input_data.technical,
            input_data.sentiment,
            input_data.news,
            input_data.fundamental,
            input_data.bullish,
            input_data.bearish,
            comparative=input_data.comparative,
            owns_position=input_data.owns_position,
            position_qty=position_qty,
            sector_context=input_data.context.sector_rotation,
            earnings_context=input_data.context.earnings,
            peer_analysis_context=input_data.context.peer_analysis,
            backtest_validation=input_data.backtest_validation,
            game_plan_context=input_data.context.game_plan,
            position_context=input_data.context.position,
            degradation_context=input_data.degradation_context,
            recent_trades_context=input_data.context.recent_trades,
        ),
        collector,
    )

    return DecisionOutput(final_decision=decision)
