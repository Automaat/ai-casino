"""Strategy selection and validation stage implementation."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Coroutine
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, TypeVar

from loguru import logger

if TYPE_CHECKING:
    from src.agents.meta import MetaAgent
    from src.backtesting import VectorBTRunner
    from src.daemon.config import PreTradeBacktestingConfig
    from src.metrics.execution import ExecutionMetricsCollector

from src.metrics.execution import current_agent
from src.workflows.models.backtest import BacktestValidationOutput
from src.workflows.models.strategy import StrategySelectionInput, StrategySelectionOutput
from src.workflows.types import BacktestValidation

T = TypeVar("T")


async def _timed_agent_call(
    agent_name: str,
    coro: Coroutine[Any, Any, T],
    collector: ExecutionMetricsCollector | None,
) -> T:
    """Wrap an agent coroutine with timing and context var tracking.

    Args:
        agent_name: Agent name for metrics
        coro: Coroutine to execute
        collector: Optional metrics collector

    Returns:
        Result of coroutine execution
    """
    if collector is None:
        return await coro
    token = current_agent.set(agent_name)
    start = time.perf_counter()
    try:
        return await coro
    finally:
        collector.record_agent_timing(agent_name, (time.perf_counter() - start) * 1000)
        current_agent.reset(token)


async def select_strategy(
    input_data: StrategySelectionInput,
    meta_agent: MetaAgent | None,
    default_strategy: Any,
    use_ensemble: bool,
    collector: ExecutionMetricsCollector | None,
) -> StrategySelectionOutput:
    """Select trading strategy via meta-agent or fallback.

    Args:
        input_data: Strategy selection input with symbol and market data
        meta_agent: Optional meta-agent for dynamic strategy selection
        default_strategy: Default strategy instance (MomentumStrategy or EnsembleStrategy)
        use_ensemble: Use ensemble strategy as default
        collector: Optional metrics collector

    Returns:
        StrategySelectionOutput with selected strategy and analysis
    """
    if meta_agent:
        # Extract daily DataFrame for meta-agent using helper
        daily_data = input_data.get_daily_data()

        selection = await _timed_agent_call(
            "meta_agent",
            meta_agent.select_strategy(input_data.symbol, daily_data),
            collector,
        )
        return StrategySelectionOutput(
            strategy_instance=selection.strategy_instance,
            strategy_name=selection.strategy_name,
            regime_analysis=selection.regime_analysis,
            strategy_selection=selection,
        )

    # No meta-agent, use default strategy
    name = "ensemble" if use_ensemble else "momentum"
    return StrategySelectionOutput(
        strategy_instance=default_strategy,
        strategy_name=name,
        regime_analysis=None,
        strategy_selection=None,
    )


async def validate_strategy_with_backtest(
    symbol: str,
    strategy: Any,
    strategy_name: str,
    input_data: StrategySelectionInput,
    pre_trade_backtest_config: PreTradeBacktestingConfig | None,
    vectorbt_runner: VectorBTRunner | None,
    collector: ExecutionMetricsCollector | None,  # noqa: ARG001
) -> BacktestValidationOutput:
    """Run pre-trade backtesting validation on selected strategy.

    Args:
        symbol: Stock ticker
        strategy: Strategy instance
        strategy_name: Strategy name for logging
        input_data: Strategy selection input with market data
        pre_trade_backtest_config: Optional backtesting configuration
        vectorbt_runner: Optional VectorBT runner instance
        collector: Optional metrics collector

    Returns:
        BacktestValidationOutput with validation result
    """
    if not pre_trade_backtest_config or not pre_trade_backtest_config.enabled:
        return BacktestValidationOutput(backtest_validation=None, warnings=[])

    if not vectorbt_runner:
        logger.warning("Backtesting enabled but VectorBTRunner not initialized")
        return BacktestValidationOutput(backtest_validation=None, warnings=[])

    logger.info(f"Running pre-trade backtest for {symbol} ({strategy_name})")

    try:
        end_date = datetime.now(UTC)
        start_date = end_date - timedelta(days=pre_trade_backtest_config.lookback_days)

        backtest_result = await asyncio.to_thread(
            vectorbt_runner.run_backtest,
            symbol,
            start_date,
            end_date,
            strategy,
        )

        failure_reasons = []
        if backtest_result.sharpe_ratio < pre_trade_backtest_config.min_sharpe_threshold:
            min_sharpe = pre_trade_backtest_config.min_sharpe_threshold
            failure_reasons.append(f"Sharpe {backtest_result.sharpe_ratio:.2f} < {min_sharpe}")
        if abs(backtest_result.max_drawdown) > pre_trade_backtest_config.max_drawdown_threshold:
            failure_reasons.append(
                f"Max drawdown {abs(backtest_result.max_drawdown):.1%} > "
                f"{pre_trade_backtest_config.max_drawdown_threshold:.1%}"
            )

        passed = len(failure_reasons) == 0
        confidence_adjustment = 1.0 if passed else pre_trade_backtest_config.confidence_penalty_multiplier

        validation = BacktestValidation(
            symbol=symbol,
            strategy_name=strategy_name,
            passed=passed,
            sharpe_ratio=backtest_result.sharpe_ratio,
            max_drawdown=backtest_result.max_drawdown,
            total_return=backtest_result.total_return,
            win_rate=backtest_result.win_rate,
            profit_factor=backtest_result.profit_factor,
            total_trades=backtest_result.total_trades,
            lookback_days=pre_trade_backtest_config.lookback_days,
            failure_reasons=failure_reasons,
            confidence_adjustment=confidence_adjustment,
        )

        warnings = []
        if not passed:
            warning = f"Backtest FAILED ({strategy_name}): {'; '.join(failure_reasons)}"
            logger.warning(warning)
            warnings.append(warning)

        logger.info(
            f"Backtest {'PASSED' if passed else 'FAILED'}: "
            f"Sharpe={backtest_result.sharpe_ratio:.2f}, MaxDD={abs(backtest_result.max_drawdown):.1%}"
        )

        return BacktestValidationOutput(backtest_validation=validation, warnings=warnings)

    except Exception as e:
        logger.warning(f"Backtest validation error: {e}, continuing without validation")
        warning = f"Backtest error: {e}"
        return BacktestValidationOutput(backtest_validation=None, warnings=[warning])
