"""Trading state manager."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from loguru import logger
from pydantic import PrivateAttr

from src.daemon.state.managers.base import StateManager
from src.daemon.state.models import AnalysisRecord
from src.strategies.session import TradingSession

if TYPE_CHECKING:
    from src.database.repositories.analysis import AnalysisRecordRepository
    from src.database.repositories.metadata import MetadataRepository


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
    technical_analysis_reasoning: str | None = None
    sentiment_analysis_reasoning: str | None = None
    news_analysis_reasoning: str | None = None


class TradingStateManager(StateManager):
    """Trading signal tracking and paper trading metrics."""

    _analysis_repository: AnalysisRecordRepository | None = PrivateAttr(default=None)
    _metadata_repository: MetadataRepository | None = PrivateAttr(default=None)
    _analysis_cache: list[AnalysisRecord] | None = PrivateAttr(default=None)

    def set_repositories(
        self,
        analysis_repository: AnalysisRecordRepository,
        metadata_repository: MetadataRepository,
    ) -> None:
        """Inject repositories.

        Args:
            analysis_repository: Analysis record repository
            metadata_repository: Metadata repository
        """
        self._analysis_repository = analysis_repository
        self._metadata_repository = metadata_repository
        logger.debug("TradingStateManager repositories injected")

    async def get_last_run(self) -> datetime | None:
        """Get last run timestamp from DB."""
        if not self._metadata_repository:
            return None
        return await self._metadata_repository.get("trading.last_run")

    async def get_total_analyses(self) -> int:
        """Get total analyses count from DB."""
        if not self._metadata_repository:
            return 0
        value = await self._metadata_repository.get("trading.total_analyses")
        return value if value is not None else 0

    async def get_total_trades(self) -> int:
        """Get total trades count from DB."""
        if not self._metadata_repository:
            return 0
        value = await self._metadata_repository.get("trading.total_trades")
        return value if value is not None else 0

    async def get_paper_trading_start_date(self) -> datetime | None:
        """Get paper trading start date from DB."""
        if not self._metadata_repository:
            return None
        return await self._metadata_repository.get("trading.paper_trading_start_date")

    async def get_current_trading_mode(self) -> str:
        """Get current trading mode from DB."""
        if not self._metadata_repository:
            return "paper"
        value = await self._metadata_repository.get("trading.current_trading_mode")
        return value if value is not None else "paper"

    async def get_last_journal_date(self) -> str | None:
        """Get last journal date from DB."""
        if not self._metadata_repository:
            return None
        return await self._metadata_repository.get("trading.last_journal_date")

    async def get_last_signal_tracking(self) -> datetime | None:
        """Get last signal tracking timestamp from DB."""
        if not self._metadata_repository:
            return None
        return await self._metadata_repository.get("trading.last_signal_tracking")

    async def get_analyses(self, limit: int = 1000) -> list[AnalysisRecord]:
        """Get recent analyses with lazy loading.

        Args:
            limit: Maximum number of records to return

        Returns:
            List of recent AnalysisRecords
        """
        if not self._analysis_repository:
            return []
        if self._analysis_cache is None:
            self._analysis_cache = await self._analysis_repository.get_recent(limit)
        return self._analysis_cache

    async def record_analysis(self, input_data: AnalysisRecordInput) -> None:
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
            technical_analysis_reasoning=input_data.technical_analysis_reasoning,
            sentiment_analysis_reasoning=input_data.sentiment_analysis_reasoning,
            news_analysis_reasoning=input_data.news_analysis_reasoning,
        )

        # Immediate DB write
        if self._analysis_repository:
            await self._analysis_repository.create(record)
            logger.debug(f"Persisted analysis record: {input_data.symbol} {input_data.signal}")

        # Update metadata counters
        if self._metadata_repository:
            total = await self.get_total_analyses()
            await self._metadata_repository.set("trading.total_analyses", total + 1)
            await self._metadata_repository.set("trading.last_run", datetime.now(UTC))

            if input_data.executed:
                trades = await self.get_total_trades()
                await self._metadata_repository.set("trading.total_trades", trades + 1)

        # Invalidate cache
        self._analysis_cache = None

    def __repr__(self) -> str:
        """Return string representation."""
        return f"TradingStateManager(analyses={self.total_analyses}, trades={self.total_trades})"
