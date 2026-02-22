"""Trade execution stage implementation."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from loguru import logger
from result import Err

if TYPE_CHECKING:
    from src.data.broker import AlpacaBroker
    from src.data.market import MarketDataFetcher
    from src.database.repositories.execution_metric import ExecutionMetricRepository
    from src.database.repositories.snapshot import PortfolioSnapshotRepository
    from src.v1.notifications.service import NotificationService
    from src.v1.trades.service import TradingService

from src.agents.risk import AccountInfo, RiskAssessment
from src.agents.trader import TradingDecision
from src.data.broker import OrderStatus
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
    trading_service: TradingService | None = None,
    market_fetcher: MarketDataFetcher | None = None,
    execution_metric_repository: ExecutionMetricRepository | None = None,
) -> TradeExecutionOutput:
    """Execute trade via TradingService (preferred) or direct broker fallback.

    Args:
        input_data: Trade execution input with decision and risk assessment
        broker: Optional Alpaca broker for trade execution
        trading_service: Optional TradingService for unified execution
        market_fetcher: Optional market data fetcher for current price
        execution_metric_repository: Optional repository for metrics persistence

    Returns:
        TradeExecutionOutput with order status
    """
    if not broker and not trading_service:
        msg = "Cannot execute trade without broker or trading_service"
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

    # Use TradingService if available
    if trading_service:
        return await _execute_via_service(
            input_data, trading_service, market_fetcher, execution_metric_repository
        )

    # Fallback: direct broker submission
    return await _execute_via_broker(
        input_data, broker, warnings, market_fetcher, execution_metric_repository
    )


async def _execute_via_service(
    input_data: TradeExecutionInput,
    trading_service: TradingService,
    market_fetcher: MarketDataFetcher | None,
    execution_metric_repository: ExecutionMetricRepository | None,
) -> TradeExecutionOutput:
    """Execute trade through TradingService."""
    from src.v1.trades.models import MIN_RATIONALE_LENGTH, TradeAction, TradeRequest

    stop_loss_price = (
        input_data.risk_assessment.stop_loss.stop_loss_price if input_data.risk_assessment.stop_loss else None
    )
    qty = (
        int(input_data.risk_assessment.position_sizing.recommended_shares)
        if input_data.risk_assessment.position_sizing
        else 0
    )
    rationale = (
        " ".join(input_data.final_decision.reasoning) if input_data.final_decision.reasoning else "workflow"
    )

    # Capture requested price BEFORE order submission (for slippage tracking)
    requested_price = await _capture_requested_price(
        input_data.symbol, market_fetcher, execution_metric_repository
    )

    request = TradeRequest(
        symbol=input_data.symbol,
        action=TradeAction(input_data.final_decision.action.value.upper()),
        quantity=max(qty, 1),
        confidence=input_data.final_decision.confidence,
        rationale=rationale if len(rationale) >= MIN_RATIONALE_LENGTH else f"Workflow trade: {rationale}",
        stop_loss_price=stop_loss_price,
        strategy_name="workflow",
    )

    result = await trading_service.execute(request)

    # Convert TradeResult back to OrderStatus for workflow compatibility
    order: OrderStatus | None = None
    warnings: list[str] = []

    if result.executed and result.order_id:
        order = OrderStatus(
            order_id=result.order_id,
            symbol=result.symbol,
            qty=float(result.quantity),
            filled_qty=float(result.quantity),
            side=result.action.value.lower(),
            status=result.status,
            submitted_at=result.submitted_at or datetime.now(UTC),
            filled_at=result.submitted_at,
            filled_avg_price=result.filled_avg_price,
        )

        # Track slippage metrics
        if execution_metric_repository and requested_price:
            await _track_execution_metric(order, requested_price, execution_metric_repository)
    elif result.rejection:
        warnings.append(result.rejection.message)

    return TradeExecutionOutput(order_status=order, warnings=warnings)


async def _execute_via_broker(
    input_data: TradeExecutionInput,
    broker: AlpacaBroker | None,
    warnings: list[str],
    market_fetcher: MarketDataFetcher | None,
    execution_metric_repository: ExecutionMetricRepository | None,
) -> TradeExecutionOutput:
    """Execute trade directly via broker (legacy path)."""
    if not broker:
        msg = "Cannot execute trade without broker"
        raise ValueError(msg)

    action = input_data.final_decision.action

    # Capture requested price BEFORE order submission (for slippage tracking)
    requested_price = await _capture_requested_price(
        input_data.symbol, market_fetcher, execution_metric_repository
    )

    broker_client = broker

    def _sync_submit_order() -> OrderStatus | None:
        """Synchronous order submission wrapper for thread execution."""
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
        order_result = broker_client.submit_order(
            symbol=input_data.symbol,
            qty=qty,
            side=action.value.lower(),
            stop_loss_price=stop_loss_price,
        )
        if isinstance(order_result, Err):
            logger.critical(
                f"BROKER API FAILURE during order submission for {input_data.symbol} "
                f"with action {action.value}: {order_result.err_value}"
            )
            warnings.append(f"Order submission failed: {order_result.err_value}")
            return None
        order = order_result.ok()
        stop_loss_str = f"{stop_loss_price:.2f}" if stop_loss_price is not None else "None"
        logger.info(f"Executed {action.value}: {input_data.symbol} x{order.qty} (stop-loss={stop_loss_str})")
        return order

    order = await asyncio.to_thread(_sync_submit_order)

    # Track execution metrics
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
    from src.v1.notifications.models import NotificationMessage, NotificationSeverity

    if not notification_service:
        return

    if not risk_assessment or not final_decision:
        return

    message = NotificationMessage(
        title=f"Trade Blocked: {symbol}",
        body=risk_assessment.validation.reasoning,
        severity=NotificationSeverity.WARNING,
        metadata={
            "symbol": symbol,
            "signal": final_decision.action.value,
            "price": risk_assessment.current_price,
            "confidence": final_decision.confidence,
            "reason": risk_assessment.validation.reasoning,
            "risk_score": risk_assessment.validation.risk_score,
        },
        timestamp=datetime.now(UTC),
    )

    await notification_service.notify(message)
