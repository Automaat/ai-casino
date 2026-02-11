"""Trading state manager."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from loguru import logger
from pydantic import Field, PrivateAttr

from src.daemon.state.managers.base import StateManager, _log_task_exception
from src.daemon.state.models import AnalysisRecord
from src.strategies.session import TradingSession

if TYPE_CHECKING:
    from src.database.repositories.analysis import AnalysisRecordRepository


class TradingStateManager(StateManager):
    """Trading signal tracking and paper trading metrics."""

    last_run: datetime | None = None
    analyses: list[AnalysisRecord] = Field(default_factory=list)
    total_analyses: int = 0
    total_trades: int = 0
    paper_trading_start_date: datetime | None = None
    current_trading_mode: str = "paper"
    last_journal_date: str | None = None
    last_signal_tracking: datetime | None = None

    _analysis_repository: AnalysisRecordRepository | None = PrivateAttr(default=None)

    def set_repository(self, repository: AnalysisRecordRepository) -> None:
        """Inject analysis repository.

        Args:
            repository: Analysis record repository
        """
        self._analysis_repository = repository
        logger.debug("Analysis repository injected")

    def record_analysis(  # noqa: PLR0913
        self,
        symbol: str,
        signal: str,
        confidence: float,
        executed: bool = False,
        trading_session: TradingSession = TradingSession.REGULAR,
        is_paper_trade: bool = True,
        rsi: float | None = None,
        macd_hist: float | None = None,
        reasoning: list[str] | None = None,
    ) -> None:
        """Record an analysis result.

        Args:
            symbol: Stock ticker
            signal: Trading signal (BUY/SELL/HOLD)
            confidence: Signal confidence
            executed: Whether trade was executed
            trading_session: Trading session type (REGULAR/PRE_MARKET)
            is_paper_trade: Whether trade was paper or live
            rsi: RSI indicator value
            macd_hist: MACD histogram value
            reasoning: LLM decision reasoning
        """
        record = AnalysisRecord(
            symbol=symbol,
            timestamp=datetime.now(UTC),
            signal=signal,
            confidence=confidence,
            executed_trade=executed,
            trading_session=trading_session,
            is_paper_trade=is_paper_trade,
            rsi=rsi,
            macd_hist=macd_hist,
            reasoning=reasoning or [],
        )

        # Persist to database if repository available
        if self._analysis_repository:
            try:
                task = asyncio.create_task(self._analysis_repository.create(record))  # type: ignore[bad-argument-type]
                task.add_done_callback(_log_task_exception)
                logger.debug(f"Persisted analysis record to database: {symbol} {signal}")
            except Exception as e:
                logger.error(f"Failed to persist analysis record to database: {e}")
                raise

        # Keep in-memory list (capped for transition period)
        self.analyses.append(record)
        self.total_analyses += 1
        if executed:
            self.total_trades += 1
        self.last_run = datetime.now(UTC)

        self.analyses = self._cap_history(self.analyses, 1000, 500)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"TradingStateManager(analyses={self.total_analyses}, trades={self.total_trades})"
