"""Position review task — enriches broker positions and enqueues for coordinator."""

import asyncio
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from loguru import logger
from result import Err

from src.daemon.events import EnrichedPosition, PositionReviewEvent, Sentiment, TriageResult, Urgency
from src.strategies.session import TradingSession
from src.v1.tasks.interface import Task
from src.v1.tasks.models import WEEKDAYS, DedupStrategy, TaskResult, TaskSchedule

if TYPE_CHECKING:
    from src.daemon.config.position_review import PositionReviewConfig
    from src.daemon.scheduler import MarketScheduler
    from src.database.engine import DatabaseEngine
    from src.v1.event_queue.service import MarketEventQueue
    from src.v1.trades.brokers.models import BrokerPosition
    from src.v1.trades.brokers.protocol import Broker

_FLAG_SIGNIFICANT_LOSS_PCT = -5.0
_FLAG_DETERIORATING_PCT = -2.0
_FLAG_EXTENDED_HOLD_DAYS = 20
_FLAG_AGING_DAYS = 10
_FLAG_LOW_CONFIDENCE = 0.6
_FLAG_PROFIT_TAKING_PCT = 10.0


class PositionReviewTask(Task):
    """Scheduled position review — gathers enriched data and fires event."""

    def __init__(
        self,
        broker: Broker,
        queue: MarketEventQueue,
        config: PositionReviewConfig,
        scheduler: MarketScheduler,
        database_engine: DatabaseEngine | None = None,
    ) -> None:
        """Initialize position review task.

        Args:
            broker: Broker for position data
            queue: Market event queue for enqueuing
            config: Position review configuration
            scheduler: Market scheduler for session checks
            database_engine: Optional DB engine for entry metadata
        """
        self._broker = broker
        self._queue = queue
        self._config = config
        self._scheduler = scheduler
        self._db = database_engine
        self._last_run: datetime | None = None

    @property
    def name(self) -> str:
        """Task identifier."""
        return "position_review"

    @property
    def schedule(self) -> TaskSchedule:
        """Schedule from config."""
        return TaskSchedule(
            days=WEEKDAYS,
            enabled=self._config.enabled,
            dedup=DedupStrategy.INTERVAL,
            dedup_interval_minutes=self._config.interval_minutes,
        )

    async def execute(self) -> TaskResult:
        """Fetch positions, enrich, and enqueue review event.

        Returns:
            TaskResult with outcome
        """
        start = time.monotonic()

        session = self._scheduler.get_trading_session()
        if self._config.run_during == "regular_market" and session != TradingSession.REGULAR:
            self._last_run = datetime.now(UTC)
            return TaskResult(
                task_name=self.name,
                success=True,
                duration_seconds=time.monotonic() - start,
                message=f"Skipped: session={session.value if session else 'closed'}",
            )

        _result = await asyncio.to_thread(self._broker.get_account_info)
        if isinstance(_result, Err):
            msg = f"Broker API unavailable: {_result.err_value}"
            logger.opt(exception=True).error(msg)
            return TaskResult(
                task_name=self.name, success=False, duration_seconds=time.monotonic() - start, message=msg
            )
        account = _result.ok()
        if not account.positions:
            self._last_run = datetime.now(UTC)
            return TaskResult(
                task_name=self.name,
                success=True,
                duration_seconds=time.monotonic() - start,
                message="No positions",
            )

        enriched = await self._enrich_positions(account.positions)
        event = PositionReviewEvent(
            positions=enriched,
            portfolio_value=account.portfolio_value,
            total_exposure=account.total_exposure,
        )
        triage = TriageResult(
            event_id=event.event_id,
            event_type="position_review",
            symbols=[p.symbol for p in enriched],
            urgency=Urgency.IMMEDIATE,
            sentiment=Sentiment.NEUTRAL,
            confidence=1.0,
            reasoning="Scheduled position review",
            relevance=1.0,
        )

        await self._queue.enqueue(event, triage, ttl_hours=2)
        self._last_run = datetime.now(UTC)

        duration = time.monotonic() - start
        flagged = sum(1 for p in enriched if p.flags)
        msg = f"{len(enriched)} positions reviewed, {flagged} flagged"
        logger.info(f"Position review enqueued: {msg}")

        return TaskResult(
            task_name=self.name,
            success=True,
            duration_seconds=duration,
            message=msg,
        )

    async def _enrich_positions(self, broker_positions: dict[str, BrokerPosition]) -> list[EnrichedPosition]:
        """Enrich broker positions with DB entry metadata and health flags.

        Args:
            broker_positions: Symbol -> BrokerPosition mapping

        Returns:
            List of enriched positions
        """
        enriched: list[EnrichedPosition] = []

        entry_trades = await self._fetch_entry_trades(list(broker_positions.keys()))

        for symbol, bp in broker_positions.items():
            trade = entry_trades.get(symbol)

            days_held = None
            entry_confidence = None
            entry_signal = None
            stop_loss_price = None

            if trade:
                days_held = (datetime.now(UTC) - trade.timestamp).days
                entry_confidence = trade.confidence
                entry_signal = trade.action.value
                stop_loss_price = trade.stop_loss_price

            pnl_pct_points = bp.unrealized_pnl_percent * 100.0
            flags = self.compute_health_flags(pnl_pct_points, days_held, entry_confidence)

            enriched.append(
                EnrichedPosition(
                    symbol=symbol,
                    qty=bp.qty,
                    avg_entry_price=bp.avg_entry_price,
                    current_price=bp.market_value / bp.qty if bp.qty else 0.0,
                    unrealized_pnl=bp.unrealized_pnl,
                    unrealized_pnl_percent=bp.unrealized_pnl_percent,
                    days_held=days_held,
                    entry_confidence=entry_confidence,
                    entry_signal=entry_signal,
                    stop_loss_price=stop_loss_price,
                    flags=flags,
                )
            )

        return enriched

    async def _fetch_entry_trades(self, symbols: list[str]) -> dict:
        """Fetch entry trades from DB for given symbols.

        Args:
            symbols: List of ticker symbols

        Returns:
            Dict mapping symbol to TradeRecord (or empty if no DB)
        """
        if not self._db:
            return {}

        try:
            from src.database.repositories.trade import TradeRepository

            async with self._db.session() as session:
                repo = TradeRepository(session)
                return await repo.get_entry_trades_bulk(symbols)
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to fetch entry trades: {e}")
            return {}

    @staticmethod
    def compute_health_flags(
        pnl_pct: float,
        days_held: int | None,
        entry_confidence: float | None,
    ) -> list[str]:
        """Compute health flags for a position.

        Args:
            pnl_pct: Unrealized P&L percentage
            days_held: Days position has been held
            entry_confidence: Original entry confidence

        Returns:
            List of flag strings
        """
        flags: list[str] = []

        if pnl_pct <= _FLAG_SIGNIFICANT_LOSS_PCT:
            flags.append("SIGNIFICANT_LOSS")
        elif pnl_pct <= _FLAG_DETERIORATING_PCT:
            flags.append("DETERIORATING")

        if days_held is not None:
            if days_held >= _FLAG_EXTENDED_HOLD_DAYS:
                flags.append("EXTENDED_HOLD")
            elif days_held >= _FLAG_AGING_DAYS:
                flags.append("AGING")

        if entry_confidence is not None and entry_confidence < _FLAG_LOW_CONFIDENCE:
            flags.append("LOW_ENTRY_CONFIDENCE")

        if pnl_pct >= _FLAG_PROFIT_TAKING_PCT:
            flags.append("CONSIDER_PROFIT_TAKING")

        return flags

    async def last_run_at(self) -> datetime | None:
        """Get last execution timestamp."""
        return self._last_run

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"PositionReviewTask(enabled={self._config.enabled}, interval={self._config.interval_minutes}m)"
        )
