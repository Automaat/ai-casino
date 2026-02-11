"""Alpaca broker integration for paper trading.

This module provides a thin wrapper around the Alpaca API for executing
paper trades and fetching account information.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import TYPE_CHECKING

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest, StopLossRequest, StopOrderRequest
from loguru import logger
from pydantic import BaseModel

if TYPE_CHECKING:
    from alpaca.trading.models import Clock, Order, Position, TradeAccount

    from src.cache.historical import HistoricalCache


def round_price_for_broker(price: float) -> float:
    """Round price to valid broker increment.

    Alpaca requires:
    - Stocks >= $1.00: $0.01 increment (2 decimals)
    - Stocks < $1.00: $0.0001 increment (4 decimals)

    Args:
        price: Price to round

    Returns:
        Rounded price compliant with broker API
    """
    if price >= 1.0:
        return round(price, 2)
    return round(price, 4)


class BrokerAPIError(Exception):
    """Broker API communication failure."""


class BrokerPosition(BaseModel):
    """Broker position information."""

    symbol: str
    qty: float
    market_value: float
    avg_entry_price: float
    unrealized_pnl: float
    unrealized_pnl_percent: float


class BrokerAccountInfo(BaseModel):
    """Broker account information."""

    balance: float
    available_cash: float
    positions: dict[str, BrokerPosition]
    total_exposure: float
    portfolio_value: float


class OrderStatus(BaseModel):
    """Order status information."""

    order_id: str
    symbol: str
    qty: float
    filled_qty: float
    side: str
    status: str
    submitted_at: datetime
    filled_at: datetime | None
    filled_avg_price: float | None


class AlpacaBroker:
    """Alpaca broker client for paper trading."""

    @staticmethod
    def get_credentials(trading_mode: str) -> tuple[str | None, str | None]:
        """Get credentials based on trading mode.

        Args:
            trading_mode: Trading mode ("paper" or "live")

        Returns:
            Tuple of (api_key, secret_key)
        """
        if trading_mode == "paper":
            return (
                os.getenv("ALPACA_PAPER_API_KEY") or os.getenv("ALPACA_API_KEY"),
                os.getenv("ALPACA_PAPER_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY"),
            )
        # live
        return (
            os.getenv("ALPACA_API_KEY"),
            os.getenv("ALPACA_SECRET_KEY"),
        )

    def __init__(
        self,
        api_key: str | None = None,
        secret_key: str | None = None,
        paper: bool = True,
        historical_cache: HistoricalCache | None = None,
    ) -> None:
        """Initialize Alpaca broker client.

        Args:
            api_key: Alpaca API key (from env if not provided)
            secret_key: Alpaca secret key (from env if not provided)
            paper: Whether to use paper trading (default True)
            historical_cache: Optional permanent cache for order fills
        """
        self.api_key = api_key or os.getenv("ALPACA_API_KEY")
        self.secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY")
        self.paper = paper
        self._cache = historical_cache

        if not self.api_key or not self.secret_key:
            msg = "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set"
            raise ValueError(msg)

        self.client = TradingClient(
            api_key=self.api_key,
            secret_key=self.secret_key,
            paper=self.paper,
        )
        logger.info(f"Initialized AlpacaBroker (paper={self.paper})")

    def get_account_info(self) -> BrokerAccountInfo:
        """Fetch current account information.

        Returns:
            BrokerAccountInfo with balance, cash, positions, and exposure
        """
        try:
            account: TradeAccount = self.client.get_account()  # type: ignore[assignment]
            positions_raw: list[Position] = self.client.get_all_positions()  # type: ignore[assignment]

            positions = {}
            total_exposure = 0.0

            for pos in positions_raw:
                market_value = float(pos.market_value) if pos.market_value else 0.0
                unrealized_pl = float(pos.unrealized_pl) if pos.unrealized_pl else 0.0
                unrealized_plpc = float(pos.unrealized_plpc) if pos.unrealized_plpc else 0.0

                positions[pos.symbol] = BrokerPosition(
                    symbol=pos.symbol,
                    qty=float(pos.qty or 0),
                    market_value=market_value,
                    avg_entry_price=float(pos.avg_entry_price or 0.0),
                    unrealized_pnl=unrealized_pl,
                    unrealized_pnl_percent=unrealized_plpc,
                )
                total_exposure += market_value

            return BrokerAccountInfo(
                balance=float(account.equity) if account.equity else 0.0,
                available_cash=float(account.buying_power) if account.buying_power else 0.0,
                positions=positions,
                total_exposure=total_exposure,
                portfolio_value=float(account.portfolio_value) if account.portfolio_value else 0.0,
            )
        except Exception as e:
            msg = f"Failed to fetch account info: {e}"
            logger.error(msg)
            raise BrokerAPIError(msg) from e

    def submit_order(
        self, symbol: str, qty: int, side: str, stop_loss_price: float | None = None
    ) -> OrderStatus:
        """Submit market order with optional stop loss.

        Args:
            symbol: Stock ticker symbol
            qty: Number of shares to trade
            side: Order side ("buy" or "sell")
            stop_loss_price: Optional stop loss price

        Returns:
            OrderStatus with order details
        """
        if qty <= 0:
            msg = f"Order quantity must be positive, got {qty}"
            raise ValueError(msg)

        normalized_side = side.lower()
        if normalized_side == "buy":
            order_side = OrderSide.BUY
        elif normalized_side == "sell":
            order_side = OrderSide.SELL
        else:
            msg = f"Invalid order side: {side!r}. Expected 'buy' or 'sell'."
            raise ValueError(msg)

        try:
            order_data = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=TimeInForce.DAY,
            )

            if stop_loss_price is not None:
                if stop_loss_price <= 0:
                    msg = f"Stop loss price must be positive, got {stop_loss_price}"
                    raise ValueError(msg)
                # Round stop loss price to valid broker increment
                stop_loss_price = round_price_for_broker(stop_loss_price)
                order_data.order_class = OrderClass.OTO
                order_data.stop_loss = StopLossRequest(stop_price=stop_loss_price)

            order: Order = self.client.submit_order(order_data=order_data)  # type: ignore[assignment]

            logger.info(f"Submitted order: {side.upper()} {qty} {symbol}")

            order_status = OrderStatus(
                order_id=str(order.id),
                symbol=order.symbol or "",
                qty=float(order.qty or 0),
                filled_qty=float(order.filled_qty or 0),
                side=order.side.value if order.side else "unknown",
                status=order.status.value,
                submitted_at=order.submitted_at,
                filled_at=order.filled_at,
                filled_avg_price=float(order.filled_avg_price) if order.filled_avg_price else None,
            )

            if self._cache:
                self._cache.store_order_fill(order_status)  # type: ignore[arg-type]

            return order_status
        except Exception as e:
            msg = f"Failed to submit order: {e}"
            logger.error(msg)
            raise BrokerAPIError(msg) from e

    def get_order_status(self, order_id: str) -> OrderStatus:
        """Get status of an existing order.

        Args:
            order_id: Order ID to query

        Returns:
            OrderStatus with current order details
        """
        try:
            order: Order = self.client.get_order_by_id(order_id=order_id)  # type: ignore[assignment]

            return OrderStatus(
                order_id=str(order.id),
                symbol=order.symbol or "",
                qty=float(order.qty or 0),
                filled_qty=float(order.filled_qty or 0),
                side=order.side.value if order.side else "unknown",
                status=order.status.value,
                submitted_at=order.submitted_at,
                filled_at=order.filled_at,
                filled_avg_price=float(order.filled_avg_price) if order.filled_avg_price else None,
            )
        except Exception as e:
            logger.error(f"Failed to get order status: {e}")
            raise

    def submit_stop_order(self, symbol: str, qty: int, stop_price: float) -> OrderStatus:
        """Submit stop order to protect existing long position.

        Args:
            symbol: Stock ticker symbol
            qty: Number of shares to sell
            stop_price: Price to trigger sell order

        Returns:
            OrderStatus with order details
        """
        if qty <= 0:
            msg = f"Order quantity must be positive, got {qty}"
            raise ValueError(msg)

        if stop_price <= 0:
            msg = f"Stop price must be positive, got {stop_price}"
            raise ValueError(msg)

        # Round stop price to valid broker increment
        stop_price = round_price_for_broker(stop_price)

        try:
            order_data = StopOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.GTC,
                stop_price=stop_price,
            )

            order: Order = self.client.submit_order(order_data=order_data)  # type: ignore[assignment]

            price_fmt = f"${stop_price:.4f}" if stop_price < 1.0 else f"${stop_price:.2f}"
            logger.info(f"Submitted stop order: SELL {qty} {symbol} @ {price_fmt}")

            return OrderStatus(
                order_id=str(order.id),
                symbol=order.symbol or "",
                qty=float(order.qty or 0),
                filled_qty=float(order.filled_qty or 0),
                side=order.side.value if order.side else "unknown",
                status=order.status.value,
                submitted_at=order.submitted_at,
                filled_at=order.filled_at,
                filled_avg_price=float(order.filled_avg_price) if order.filled_avg_price else None,
            )
        except Exception as e:
            msg = f"Failed to submit order: {e}"
            logger.error(msg)
            raise BrokerAPIError(msg) from e

    def cancel_order(self, order_id: str) -> None:
        """Cancel an existing order.

        Args:
            order_id: Order ID to cancel
        """
        try:
            self.client.cancel_order_by_id(order_id=order_id)
            logger.info(f"Cancelled order: {order_id}")
        except Exception as e:
            logger.error(f"Failed to cancel order: {e}")
            raise

    def is_market_open(self) -> bool:
        """Check if market is currently open.

        Returns:
            True if market is open for trading
        """
        try:
            clock: Clock = self.client.get_clock()  # type: ignore[assignment]
            return clock.is_open
        except Exception as e:
            logger.warning(f"Failed to check market status: {e}")
            return False

    def __repr__(self) -> str:
        """Return string representation."""
        return f"AlpacaBroker(paper={self.paper})"
