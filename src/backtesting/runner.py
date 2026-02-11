"""Backtesting runner and result models."""

from datetime import UTC, datetime

import pandas as pd
import yfinance as yf
from loguru import logger
from pydantic import BaseModel, ConfigDict

from backtesting import Backtest  # type: ignore[import-untyped]
from src.backtesting.strategies import MomentumBacktestStrategy
from src.metrics.tracker import TradeRecord
from src.strategies.signal import Signal


class BacktestResult(BaseModel):
    """Backtest execution result."""

    symbol: str
    start_date: datetime
    end_date: datetime
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    avg_return_per_trade: float
    trades: list[TradeRecord]

    model_config = ConfigDict(arbitrary_types_allowed=True)


class BacktestRunner:
    """Execute and analyze backtests."""

    def __init__(self, cash: float = 100000.0, commission: float = 0.002) -> None:
        """Initialize backtest runner.

        Args:
            cash: Initial cash balance
            commission: Commission rate (0.002 = 0.2%)
        """
        self.cash = cash
        self.commission = commission
        logger.info(f"Initialized BacktestRunner (cash=${cash:,.2f}, commission={commission:.2%})")

    def run_backtest(
        self,
        symbol: str,
        start_date: str | datetime,
        end_date: str | datetime,
        strategy_class: type[MomentumBacktestStrategy] = MomentumBacktestStrategy,
    ) -> BacktestResult:
        """Run backtest for symbol.

        Args:
            symbol: Stock ticker
            start_date: Backtest start date (YYYY-MM-DD or datetime)
            end_date: Backtest end date (YYYY-MM-DD or datetime)
            strategy_class: Strategy class to use

        Returns:
            BacktestResult with metrics and trades
        """
        logger.info(f"Running backtest for {symbol} ({start_date} to {end_date})")

        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=UTC)
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=UTC)

        data = self._fetch_data(symbol, start_date, end_date)

        bt = Backtest(data, strategy_class, cash=self.cash, commission=self.commission)
        stats = bt.run()

        trades = self._convert_trades(stats, symbol)

        result = BacktestResult(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            total_return=stats["Return [%]"] / 100.0,
            sharpe_ratio=stats["Sharpe Ratio"] if not pd.isna(stats["Sharpe Ratio"]) else 0.0,
            max_drawdown=stats["Max. Drawdown [%]"] / 100.0,
            win_rate=stats["Win Rate [%]"] / 100.0,
            total_trades=stats["# Trades"],
            avg_return_per_trade=stats["Avg. Trade [%]"] / 100.0 if stats["# Trades"] > 0 else 0.0,
            trades=trades,
        )

        logger.info(
            f"Backtest complete: {result.total_trades} trades, "
            f"{result.total_return:.2%} return, "
            f"{result.sharpe_ratio:.2f} Sharpe"
        )

        return result

    def _fetch_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch historical data via yfinance.

        Args:
            symbol: Stock ticker
            start_date: Start date
            end_date: End date

        Returns:
            OHLCV dataframe
        """
        logger.info(f"Fetching data for {symbol} via yfinance")

        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date)

        if data.empty:
            msg = f"No data available for {symbol} between {start_date} and {end_date}"
            raise ValueError(msg)

        data = data[["Open", "High", "Low", "Close", "Volume"]]

        logger.info(f"Fetched {len(data)} bars for {symbol}")
        return data

    def _get_trades_from_stats(self, stats) -> list:  # noqa: ANN001
        """Extract trades from backtesting library stats object.

        The backtesting library stores trades in a private _trades attribute.
        This helper encapsulates that access in one place.

        Args:
            stats: Backtest stats object from backtesting library

        Returns:
            List of trade objects, empty list if none available
        """
        return getattr(stats, "_trades", []) or []

    def _convert_trades(self, stats, symbol: str) -> list[TradeRecord]:  # noqa: ANN001
        """Convert backtesting.py trades to TradeRecord format.

        Args:
            stats: Backtest stats object
            symbol: Stock ticker

        Returns:
            List of TradeRecord instances
        """
        records: list[TradeRecord] = []
        trades = self._get_trades_from_stats(stats)

        if not trades:
            return records

        for trade in trades:
            action = Signal.BUY if trade.Size > 0 else Signal.SELL
            shares = abs(trade.Size)

            entry_price = float(trade.EntryPrice)
            exit_price = float(trade.ExitPrice) if trade.ExitPrice else None

            if exit_price is not None:
                pnl = (
                    float(trade.PnL)
                    if hasattr(trade, "PnL")
                    else (
                        (exit_price - entry_price) * shares
                        if action == Signal.BUY
                        else (entry_price - exit_price) * shares
                    )
                )
                pnl_percent = (
                    ((exit_price - entry_price) / entry_price) * 100
                    if action == Signal.BUY
                    else ((entry_price - exit_price) / entry_price) * 100
                )
                status = "CLOSED"
            else:
                pnl = None
                pnl_percent = None
                status = "OPEN"

            record = TradeRecord(
                timestamp=trade.EntryTime,
                symbol=symbol,
                action=action,
                entry_price=entry_price,
                exit_price=exit_price,
                shares=shares,
                stop_loss_price=entry_price * 0.95,
                confidence=1.0,
                risk_level="MEDIUM",
                status=status,
                pnl=pnl,
                pnl_percent=pnl_percent,
            )

            records.append(record)

        return records

    def __repr__(self) -> str:
        """String representation."""
        return f"BacktestRunner(cash=${self.cash:,.2f}, commission={self.commission:.2%})"
