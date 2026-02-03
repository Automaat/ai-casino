"""Alpaca broker integration for paper trading.

This module provides a thin wrapper around the Alpaca API for executing
paper trades and fetching account information.
"""

import os
from datetime import datetime

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest, StopLossRequest
from loguru import logger
from pydantic import BaseModel


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

    def __init__(
        self,
        api_key: str | None = None,
        secret_key: str | None = None,
        base_url: str | None = None,
        paper: bool = True,
    ) -> None:
        """Initialize Alpaca broker client.

        Args:
            api_key: Alpaca API key (from env if not provided)
            secret_key: Alpaca secret key (from env if not provided)
            base_url: Alpaca base URL (from env if not provided)
            paper: Whether to use paper trading (default True)
        """
        self.api_key = api_key or os.getenv("ALPACA_API_KEY")
        self.secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY")
        self.base_url = base_url or os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        self.paper = paper

        if not self.api_key or not self.secret_key:
            msg = "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set"
            raise ValueError(msg)

        self.client = TradingClient(api_key=self.api_key, secret_key=self.secret_key, paper=self.paper)
        logger.info(f"Initialized AlpacaBroker (paper={self.paper})")

    def get_account_info(self) -> BrokerAccountInfo:
        """Fetch current account information.

        Returns:
            BrokerAccountInfo with balance, cash, positions, and exposure
        """
        try:
            account = self.client.get_account()
            positions_raw = self.client.get_all_positions()

            positions = {}
            total_exposure = 0.0

            for pos in positions_raw:
                positions[pos.symbol] = BrokerPosition(
                    symbol=pos.symbol,
                    qty=float(pos.qty),
                    market_value=float(pos.market_value),
                    avg_entry_price=float(pos.avg_entry_price),
                    unrealized_pnl=float(pos.unrealized_pnl),
                    unrealized_pnl_percent=float(pos.unrealized_plpc),
                )
                total_exposure += float(pos.market_value)

            return BrokerAccountInfo(
                balance=float(account.equity),
                available_cash=float(account.buying_power),
                positions=positions,
                total_exposure=total_exposure,
                portfolio_value=float(account.portfolio_value),
            )
        except Exception as e:
            logger.error(f"Failed to fetch account info: {e}")
            raise

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
        try:
            order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL

            order_data = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=TimeInForce.DAY,
            )

            if stop_loss_price is not None:
                order_data.order_class = "bracket"
                order_data.stop_loss = StopLossRequest(stop_price=stop_loss_price)

            order = self.client.submit_order(order_data=order_data)

            logger.info(f"Submitted order: {side.upper()} {qty} {symbol}")

            return OrderStatus(
                order_id=str(order.id),
                symbol=order.symbol,
                qty=float(order.qty),
                filled_qty=float(order.filled_qty or 0),
                side=order.side.value,
                status=order.status.value,
                submitted_at=order.submitted_at,
                filled_at=order.filled_at,
                filled_avg_price=float(order.filled_avg_price) if order.filled_avg_price else None,
            )
        except Exception as e:
            logger.error(f"Failed to submit order: {e}")
            raise

    def get_order_status(self, order_id: str) -> OrderStatus:
        """Get status of an existing order.

        Args:
            order_id: Order ID to query

        Returns:
            OrderStatus with current order details
        """
        try:
            order = self.client.get_order_by_id(order_id=order_id)

            return OrderStatus(
                order_id=str(order.id),
                symbol=order.symbol,
                qty=float(order.qty),
                filled_qty=float(order.filled_qty or 0),
                side=order.side.value,
                status=order.status.value,
                submitted_at=order.submitted_at,
                filled_at=order.filled_at,
                filled_avg_price=float(order.filled_avg_price) if order.filled_avg_price else None,
            )
        except Exception as e:
            logger.error(f"Failed to get order status: {e}")
            raise

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

    def __repr__(self) -> str:
        """Return string representation."""
        return f"AlpacaBroker(paper={self.paper})"
