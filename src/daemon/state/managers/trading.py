"""Trading state manager."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from loguru import logger
from pydantic import Field, PrivateAttr

from src.daemon.state.managers.base import StateManager, _make_task_cleanup_callback
from src.daemon.state.models import AnalysisRecord
from src.strategies.session import TradingSession

if TYPE_CHECKING:
    from src.database.repositories.analysis import AnalysisRecordRepository


@dataclass
class AnalysisRecordInput:
    """Input parameters for recording an analysis."""

    symbol: str
    signal: str
    confidence: float
    executed: bool = False
    trading_session: TradingSession = TradingSession.REGULAR
    is_paper_trade: bool = True
    rsi: float | None = None
    macd_hist: float | None = None
    reasoning: list[str] | None = None


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

    def record_analysis(self, input_data: AnalysisRecordInput) -> None:
        """Record an analysis result.

        Args:
            input_data: Analysis record input parameters
        """
        record = AnalysisRecord(
            symbol=input_data.symbol,
            timestamp=datetime.now(UTC),
            signal=input_data.signal,
            confidence=input_data.confidence,
            executed_trade=input_data.executed,
            trading_session=input_data.trading_session,
            is_paper_trade=input_data.is_paper_trade,
            rsi=input_data.rsi,
            macd_hist=input_data.macd_hist,
            reasoning=input_data.reasoning or [],
        )

        # Persist to database if repository available
        if self._analysis_repository:
            try:
                task = asyncio.create_task(self._analysis_repository.create(record))  # type: ignore[bad-argument-type]
                self._pending_tasks.add(task)
                task.add_done_callback(_make_task_cleanup_callback(self._pending_tasks))
                logger.debug(
                    f"Scheduled analysis record persistence to database: "
                    f"{input_data.symbol} {input_data.signal}"
                )
            except Exception as e:
                logger.error(f"Failed to schedule analysis record persistence: {e}")
                raise

        # Keep in-memory list (capped for transition period)
        self.analyses.append(record)
        self.total_analyses += 1
        if input_data.executed:
            self.total_trades += 1
        self.last_run = datetime.now(UTC)

        self.analyses = self._cap_history(self.analyses, 1000, 500)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"TradingStateManager(analyses={self.total_analyses}, trades={self.total_trades})"
