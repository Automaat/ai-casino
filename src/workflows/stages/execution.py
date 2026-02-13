"""Trade execution stage implementation."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from src.daemon.notifications import NotificationService
    from src.data.broker import AlpacaBroker
    from src.data.market import MarketDataFetcher
    from src.database.repositories.execution_metric import ExecutionMetricRepository
    from src.database.repositories.snapshot import PortfolioSnapshotRepository

from src.agents.risk import AccountInfo, RiskAssessment
from src.agents.trader import TradingDecision
from src.data.broker import BrokerAPIError, OrderStatus
from src.strategies.session import TradingSession
from src.workflows.models.execution import TradeExecutionInput, TradeExecutionOutput


async def _get_current_price(
    symbol: str,
    market_fetcher: MarketDataFetcher,
) -> Decimal:
    """Get current market price for slippage calculation.

    Args:
        symbol: Stock ticker
        market_fetcher: Market data fetcher instance

    Returns:
        Current market price as Decimal
    """
    try:
        # Fetch latest close price using 1 day period
        market_data = await asyncio.to_thread(market_fetcher.fetch_daily, symbol, 1)
    except Exception as e:
        logger.opt(exception=True).error(f"Failed to get current price for {symbol}: {e}")
        raise

    if market_data.data.empty:
        msg = f"No price data available for {symbol}"
        raise ValueError(msg)

    return Decimal(str(market_data.data["Close"].iloc[-1]))


async def _capture_requested_price(
    symbol: str,
    market_fetcher: MarketDataFetcher | None,
    execution_metric_repository: ExecutionMetricRepository | None,
) -> Decimal | None:
    """Capture requested price for slippage tracking.

    Args:
        symbol: Stock ticker
        market_fetcher: Market data fetcher
        execution_metric_repository: Execution metric repository

    Returns:
        Requested price or None if capture failed
    """
    if not market_fetcher or not execution_metric_repository:
        return None

    try:
        requested_price = await _get_current_price(symbol, market_fetcher)
        logger.debug(f"Market price at submission: {requested_price}")
        return requested_price
    except Exception as e:
        logger.opt(exception=True).warning(f"Failed to capture requested price: {e}")
        return None


async def _track_execution_metric(
    order: OrderStatus,
    requested_price: Decimal,
    execution_metric_repository: ExecutionMetricRepository,
) -> None:
    """Track execution metric in database.

    Args:
        order: Order status from broker
        requested_price: Requested price at submission
        execution_metric_repository: Repository for metrics
    """
    try:
        from src.metrics.execution_metric import ExecutionMetric

        metric = ExecutionMetric.from_order_status(order, requested_price)
        await execution_metric_repository.create(metric)
        logger.info(f"Tracked execution: {order.order_id} (slippage: {metric.slippage_bps}bps)")
    except Exception as e:
        # Non-blocking: don't fail trade if metric write fails
        logger.opt(exception=True).error(f"Failed to track execution metric: {e}")


async def execute_trade(
    input_data: TradeExecutionInput,
    broker: AlpacaBroker | None,
    market_fetcher: MarketDataFetcher | None = None,
    execution_metric_repository: ExecutionMetricRepository | None = None,
) -> TradeExecutionOutput:
    """Execute trade via broker (async, thread-offloaded).

    Args:
        input_data: Trade execution input with decision and risk assessment
        broker: Optional Alpaca broker for trade execution
        market_fetcher: Optional market data fetcher for current price
        execution_metric_repository: Optional repository for metrics persistence

    Returns:
        TradeExecutionOutput with order status
    """
    if not broker:
        msg = "Cannot execute trade without broker"
        raise ValueError(msg)

    action = input_data.final_decision.action
    warnings: list[str] = []

    # Block trades during pre-market session
    if input_data.trading_session != TradingSession.REGULAR:
        logger.warning(
            f"Trade blocked: {action.value} {input_data.symbol} - "
            f"trades only allowed during REGULAR session (current: {input_data.trading_session.value})"
        )
        return TradeExecutionOutput(order_status=None, warnings=warnings)

    # Check risk approval
    if not input_data.risk_assessment.validation.approved:
        return TradeExecutionOutput(order_status=None, warnings=warnings)

    # Capture requested price BEFORE order submission (for slippage tracking)
    requested_price = await _capture_requested_price(
        input_data.symbol,
        market_fetcher,
        execution_metric_repository,
    )

    # Capture broker for use in nested function (type narrowing)
    broker_client = broker

    def _sync_submit_order() -> OrderStatus | None:
        """Synchronous order submission wrapper for thread execution."""
        try:
            stop_loss_price = (
                input_data.risk_assessment.stop_loss.stop_loss_price
                if input_data.risk_assessment.stop_loss
                else None
            )
            qty = (
                int(input_data.risk_assessment.position_sizing.recommended_shares)
                if input_data.risk_assessment.position_sizing
                else 0
            )
            order = broker_client.submit_order(
                symbol=input_data.symbol,
                qty=qty,
                side=action.value.lower(),
                stop_loss_price=stop_loss_price,
            )
            stop_loss_str = f"{stop_loss_price:.2f}" if stop_loss_price is not None else "None"
            logger.info(
                f"Executed {action.value}: {input_data.symbol} x{order.qty} (stop-loss={stop_loss_str})"
            )
            return order
        except BrokerAPIError as e:
            logger.critical(
                f"BROKER API FAILURE during order submission for {input_data.symbol} "
                f"with action {action.value}: {e}"
            )
            warnings.append(f"Order submission failed: {e}")
            return None
        except Exception as e:
            logger.opt(exception=True).error(
                f"Unexpected error submitting order for {input_data.symbol}: {e}"
            )
            warnings.append(f"Order submission error: {e}")
            return None

    order = await asyncio.to_thread(_sync_submit_order)

    # Track execution metrics in postgres if repository available and order filled
    if execution_metric_repository and order and requested_price:
        await _track_execution_metric(order, requested_price, execution_metric_repository)

    return TradeExecutionOutput(order_status=order, warnings=warnings)


async def create_portfolio_snapshot(
    symbol: str,  # noqa: ARG001
    account_info: AccountInfo | None,
    snapshot_repository: PortfolioSnapshotRepository | None,
) -> None:
    """Capture portfolio snapshot after trade execution.

    Args:
        symbol: Stock symbol
        account_info: Account info with positions
        snapshot_repository: Optional snapshot repository
    """
    from src.database.repositories.snapshot import PortfolioSnapshot

    if not snapshot_repository or not account_info:
        return

    try:
        snapshot = PortfolioSnapshot(
            timestamp=datetime.now(UTC),
            balance=account_info.balance,
            available_cash=account_info.available_cash,
            total_exposure=account_info.total_exposure,
            portfolio_value=account_info.balance,
            positions={k: float(v) for k, v in account_info.positions.items()},
            trigger="TRADE",
        )
        await snapshot_repository.create(snapshot)
        logger.info("Captured portfolio snapshot (trigger=TRADE)")
    except Exception as e:
        logger.opt(exception=True).error(f"Failed to capture portfolio snapshot: {e}")


async def notify_trade_execution(
    symbol: str,
    final_decision: TradingDecision,
    risk_assessment: RiskAssessment,
    notification_service: NotificationService | None,
) -> None:
    """Send risk rejection notification.

    Args:
        symbol: Stock symbol
        final_decision: Final trading decision
        risk_assessment: Risk assessment
        notification_service: Optional notification service
    """
    from src.daemon.config import NotificationTrigger
    from src.daemon.notifications import NotificationMessage

    if not notification_service:
        return

    if not risk_assessment or not final_decision:
        return  # Nothing to notify if missing data

    message = NotificationMessage(
        trigger=NotificationTrigger.RISK_REJECTION,
        title=f"Trade Blocked: {symbol}",
        body=risk_assessment.validation.reasoning,
        metadata={
            "symbol": symbol,
            "signal": final_decision.action.value,
            "price": risk_assessment.current_price,
            "confidence": final_decision.confidence,
            "rejection_reason": risk_assessment.validation.reasoning,
            "risk_score": risk_assessment.validation.risk_score,
        },
        timestamp=datetime.now(UTC),
    )

    await notification_service.notify(NotificationTrigger.RISK_REJECTION, message)
