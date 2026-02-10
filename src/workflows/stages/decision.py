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
    trader: "TraderAgent",
    collector: "ExecutionMetricsCollector | None" = None,
) -> DecisionOutput:
    """Make final trading decision.

    Args:
        input_data: Decision input with all analyses and context
        trader: Trader agent for decision making
        collector: Optional metrics collector

    Returns:
        DecisionOutput with final decision
    """
    logger.info("Making final trading decision")

    # Ensure critical analyses are present
    if not input_data.technical or not input_data.sentiment or not input_data.news:
        msg = "Missing critical analyses (technical, sentiment, news)"
        raise ValueError(msg)
    if not input_data.bullish or not input_data.bearish:
        msg = "Missing research analyses (bullish, bearish)"
        raise ValueError(msg)

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
        ),
        collector,
    )

    return DecisionOutput(final_decision=decision)
