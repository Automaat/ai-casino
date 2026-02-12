"""Database-backed metrics tracker for trade persistence and performance analytics."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from src.metrics.tracker import BaseMetricsTracker, PerformanceMetrics, TradeRecord
from src.strategies.signal import Signal

if TYPE_CHECKING:
    from src.database.repositories.trade import TradeRepository
    from src.workflows.types import TradingWorkflowResult


class DatabaseMetricsTracker(BaseMetricsTracker):
    """Metrics tracker with database persistence."""

    def __init__(
        self,
        trade_repository: TradeRepository,
        risk_free_rate: float,
    ) -> None:
        """Initialize database metrics tracker.

        Args:
            trade_repository: Repository for trade persistence
            risk_free_rate: Annual risk-free rate for Sharpe ratio
        """
        super().__init__(risk_free_rate)
        self._repo = trade_repository
        self._trades_cache: list[TradeRecord] | None = None
        logger.info(f"Initialized DatabaseMetricsTracker (risk_free_rate={self.risk_free_rate:.4f})")

    @property
    def trades(self) -> list[TradeRecord]:
        """List of all trade records (cached)."""
        if self._trades_cache is None:
            return []
        return self._trades_cache

    @trades.setter
    def trades(self, value: list[TradeRecord]) -> None:
        """Set trade records (updates cache)."""
        self._trades_cache = value

    async def _get_trades(self) -> list[TradeRecord]:
        """Get all trades from database (cached)."""
        if self._trades_cache is None:
            trades = await self._repo.get_all()
            self._trades_cache = trades
        return self._trades_cache

    def _invalidate_cache(self) -> None:
        """Invalidate trades cache."""
        self._trades_cache = None

    def record_decision(
        self,
        result: TradingWorkflowResult,
        strategy_name: str | None = None,
        is_paper_trade: bool = True,
    ) -> TradeRecord:
        """Record a trading decision (sync wrapper).

        Note: For async usage, call record_decision_async directly.
        """
        return asyncio.run(self.record_decision_async(result, strategy_name, is_paper_trade))

    async def record_decision_async(
        self,
        result: TradingWorkflowResult,
        strategy_name: str | None = None,
        is_paper_trade: bool = True,
    ) -> TradeRecord:
        """Record a trading decision to database.

        Args:
            result: Trading workflow result
            strategy_name: Optional strategy name
            is_paper_trade: Whether trade is paper or live

        Returns:
            Created TradeRecord
        """
        logger.info(f"Recording decision for {result.symbol}: {result.decision.action.value}")

        status = "APPROVED" if result.risk.validation.approved else "REJECTED"
        shares = result.risk.position_sizing.recommended_shares if status == "APPROVED" else 0

        if status == "APPROVED" and result.decision.action != Signal.HOLD:
            status = "OPEN"

        trade = TradeRecord(
            timestamp=datetime.now(UTC),
            symbol=result.symbol,
            action=result.decision.action,
            entry_price=result.risk.current_price,
            exit_price=None,
            shares=shares,
            stop_loss_price=result.risk.stop_loss.stop_loss_price,
            confidence=result.decision.confidence,
            risk_level=result.decision.risk_level,
            status=status,
            pnl=None,
            pnl_percent=None,
            strategy_name=strategy_name,
            is_paper_trade=is_paper_trade,
        )

        await self._repo.create(trade)
        self._invalidate_cache()
        return trade

    def simulate_exits(self, current_prices: dict[str, float]) -> list[TradeRecord]:
        """Simulate trade exits (sync wrapper)."""
        return asyncio.run(self.simulate_exits_async(current_prices))

    async def simulate_exits_async(self, current_prices: dict[str, float]) -> list[TradeRecord]:
        """Simulate trade exits based on stop-loss prices.

        Args:
            current_prices: Dictionary mapping symbol to current price

        Returns:
            List of closed trades
        """
        closed_trades = []
        open_trades = await self._repo.get_open_trades()

        for trade in open_trades:
            current_price = current_prices.get(trade.symbol)
            if current_price is None:
                logger.warning(f"No price data for {trade.symbol}, skipping exit simulation")
                continue

            should_close = False

            if trade.action == Signal.BUY and current_price <= trade.stop_loss_price:
                should_close = True
                logger.info(
                    f"Stop-loss hit for BUY {trade.symbol}: "
                    f"price={current_price:.2f} <= stop={trade.stop_loss_price:.2f}"
                )
            elif trade.action == Signal.SELL and current_price >= trade.stop_loss_price:
                should_close = True
                logger.info(
                    f"Stop-loss hit for SELL {trade.symbol}: "
                    f"price={current_price:.2f} >= stop={trade.stop_loss_price:.2f}"
                )

            if should_close:
                trade.close_trade(current_price)
                closed_trades.append(trade)
                if trade.id:
                    await self._repo.update(
                        trade.id,
                        status=trade.status,
                        exit_price=trade.exit_price,
                        pnl=trade.pnl,
                        pnl_percent=trade.pnl_percent,
                        closed_at=trade.closed_at,
                    )

        self._invalidate_cache()
        return closed_trades

    def calculate_metrics(self, window: str = "all") -> PerformanceMetrics:
        """Calculate metrics (sync wrapper)."""
        return asyncio.run(self.calculate_metrics_async(window))

    async def calculate_metrics_async(self, window: str = "all") -> PerformanceMetrics:
        """Calculate performance metrics for specified time window.

        Args:
            window: Time window ("all", "30d", "7d")

        Returns:
            PerformanceMetrics with aggregated statistics
        """
        from src.metrics.performance import (
            calculate_max_drawdown,
            calculate_returns_from_trades,
            calculate_risk_adjusted_returns,
            calculate_sharpe_ratio,
            calculate_win_rate,
        )

        logger.info(f"Calculating metrics for window: {window}")

        filtered_trades = await self._repo.get_by_window(window)

        if not filtered_trades:
            logger.warning(f"No trades found for window: {window}")
            return self._empty_metrics(window)

        approved = [t for t in filtered_trades if not t.is_rejected()]
        closed = [t for t in filtered_trades if t.is_closed()]
        open_trades = [t for t in filtered_trades if t.is_open()]
        rejected = [t for t in filtered_trades if t.is_rejected()]

        winning = [t for t in closed if t.pnl and t.pnl > 0]
        losing = [t for t in closed if t.pnl and t.pnl < 0]

        total_pnl = sum(t.pnl for t in closed if t.pnl is not None)
        avg_win = sum(t.pnl for t in winning if t.pnl is not None) / len(winning) if winning else 0.0
        avg_loss = sum(t.pnl for t in losing if t.pnl is not None) / len(losing) if losing else 0.0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0

        win_rate = calculate_win_rate(closed)

        returns = calculate_returns_from_trades(closed)
        sharpe = calculate_sharpe_ratio(returns, self.risk_free_rate) if returns else 0.0
        max_dd, max_dd_pct = calculate_max_drawdown(closed)

        risk_adjusted = calculate_risk_adjusted_returns(returns) if returns else 0.0

        return PerformanceMetrics(
            window=window,
            total_decisions=len(filtered_trades),
            approved_trades=len(approved),
            rejected_trades=len(rejected),
            open_trades=len(open_trades),
            closed_trades=len(closed),
            winning_trades=len(winning),
            losing_trades=len(losing),
            win_rate=win_rate,
            total_pnl=total_pnl,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            max_drawdown=max_dd,
            max_drawdown_percent=max_dd_pct,
            sharpe_ratio=sharpe,
            risk_adjusted_return=risk_adjusted,
            start_date=min(t.timestamp for t in filtered_trades),
            end_date=max(t.timestamp for t in filtered_trades),
        )

    def _empty_metrics(self, window: str) -> PerformanceMetrics:
        """Create empty metrics object."""
        now = datetime.now(UTC)
        return PerformanceMetrics(
            window=window,
            total_decisions=0,
            approved_trades=0,
            rejected_trades=0,
            open_trades=0,
            closed_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            total_pnl=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            profit_factor=0.0,
            max_drawdown=0.0,
            max_drawdown_percent=0.0,
            sharpe_ratio=0.0,
            risk_adjusted_return=0.0,
            start_date=now,
            end_date=now,
        )

    def save_report(self, path: str = "logs/metrics_summary.json") -> None:
        """Save metrics report (sync wrapper)."""
        asyncio.run(self.save_report_async(path))

    async def save_report_async(self, path: str = "logs/metrics_summary.json") -> None:
        """Generate and save metrics report.

        Args:
            path: Output path for JSON report
        """
        logger.info(f"Generating metrics report to {path}")

        report = {
            "generated_at": datetime.now(UTC).isoformat(),
            "risk_free_rate": self.risk_free_rate,
            "all_time": (await self.calculate_metrics_async("all")).model_dump(),
            "last_30_days": (await self.calculate_metrics_async("30d")).model_dump(),
            "last_7_days": (await self.calculate_metrics_async("7d")).model_dump(),
        }

        def _write_report() -> None:
            report_path = Path(path)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with report_path.open("w") as f:
                json.dump(report, f, indent=2, default=str)

        try:
            await asyncio.to_thread(_write_report)
            logger.info(f"Saved metrics report to {path}")
        except Exception as e:
            logger.opt(exception=True).error(f"Failed to save report: {e}")
            raise

    def __repr__(self) -> str:
        """Return string representation."""
        return f"DatabaseMetricsTracker(risk_free_rate={self.risk_free_rate})"
