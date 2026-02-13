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

    _database_enabled: bool = PrivateAttr(default=False)
    _analysis_cache: list[AnalysisRecord] | None = PrivateAttr(default=None)

    def enable_database(self) -> None:
        """Enable database persistence."""
        self._database_enabled = True
        logger.debug("TradingStateManager database enabled")

    async def get_last_run(self) -> datetime | None:
        """Get last run timestamp from DB."""
        if not self._database_enabled:
            return None
        try:
            from src.database.connection import get_session
            from src.database.repositories.metadata import MetadataRepository

            async with get_session() as session:
                repo = MetadataRepository(session)
                return await repo.get_datetime("trading.last_run")
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to get last run: {e}")
            return None

    async def set_last_run(self, value: datetime | None) -> None:
        """Set last run timestamp in DB."""
        if not self._database_enabled or value is None:
            return
        try:
            from src.database.connection import get_session
            from src.database.repositories.metadata import MetadataRepository

            async with get_session() as session:
                repo = MetadataRepository(session)
                await repo.set("trading.last_run", value)
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to set last run: {e}")

    async def get_total_analyses(self) -> int:
        """Get total analyses count from DB."""
        if not self._database_enabled:
            return 0
        try:
            from src.database.connection import get_session
            from src.database.repositories.metadata import MetadataRepository

            async with get_session() as session:
                repo = MetadataRepository(session)
                value = await repo.get_int("trading.total_analyses")
                return value if value is not None else 0
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to get total analyses: {e}")
            return 0

    async def get_total_trades(self) -> int:
        """Get total trades count from DB."""
        if not self._database_enabled:
            return 0
        try:
            from src.database.connection import get_session
            from src.database.repositories.metadata import MetadataRepository

            async with get_session() as session:
                repo = MetadataRepository(session)
                value = await repo.get_int("trading.total_trades")
                return value if value is not None else 0
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to get total trades: {e}")
            return 0

    async def get_paper_trading_start_date(self) -> datetime | None:
        """Get paper trading start date from DB."""
        if not self._database_enabled:
            return None
        try:
            from src.database.connection import get_session
            from src.database.repositories.metadata import MetadataRepository

            async with get_session() as session:
                repo = MetadataRepository(session)
                return await repo.get_datetime("trading.paper_trading_start_date")
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to get paper trading start date: {e}")
            return None

    async def set_paper_trading_start_date(self, value: datetime | None) -> None:
        """Set paper trading start date in DB."""
        if not self._database_enabled or value is None:
            return
        try:
            from src.database.connection import get_session
            from src.database.repositories.metadata import MetadataRepository

            async with get_session() as session:
                repo = MetadataRepository(session)
                await repo.set("trading.paper_trading_start_date", value)
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to set paper trading start date: {e}")

    async def get_current_trading_mode(self) -> str:
        """Get current trading mode from DB."""
        if not self._database_enabled:
            return "paper"
        try:
            from src.database.connection import get_session
            from src.database.repositories.metadata import MetadataRepository

            async with get_session() as session:
                repo = MetadataRepository(session)
                value = await repo.get_str("trading.current_trading_mode")
                return value if value is not None else "paper"
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to get current trading mode: {e}")
            return "paper"

    async def set_current_trading_mode(self, value: str) -> None:
        """Set current trading mode in DB."""
        if not self._database_enabled:
            return
        try:
            from src.database.connection import get_session
            from src.database.repositories.metadata import MetadataRepository

            async with get_session() as session:
                repo = MetadataRepository(session)
                await repo.set("trading.current_trading_mode", value)
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to set current trading mode: {e}")

    async def get_last_journal_date(self) -> str | None:
        """Get last journal date from DB."""
        if not self._database_enabled:
            return None
        try:
            from src.database.connection import get_session
            from src.database.repositories.metadata import MetadataRepository

            async with get_session() as session:
                repo = MetadataRepository(session)
                return await repo.get_str("trading.last_journal_date")
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to get last journal date: {e}")
            return None

    async def set_last_journal_date(self, value: str | None) -> None:
        """Set last journal date in DB."""
        if not self._database_enabled or value is None:
            return
        try:
            from src.database.connection import get_session
            from src.database.repositories.metadata import MetadataRepository

            async with get_session() as session:
                repo = MetadataRepository(session)
                await repo.set("trading.last_journal_date", value)
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to set last journal date: {e}")

    async def get_last_signal_tracking(self) -> datetime | None:
        """Get last signal tracking timestamp from DB."""
        if not self._database_enabled:
            return None
        try:
            from src.database.connection import get_session
            from src.database.repositories.metadata import MetadataRepository

            async with get_session() as session:
                repo = MetadataRepository(session)
                return await repo.get_datetime("trading.last_signal_tracking")
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to get last signal tracking: {e}")
            return None

    async def set_last_signal_tracking(self, value: datetime | None) -> None:
        """Set last signal tracking timestamp in DB."""
        if not self._database_enabled or value is None:
            return
        try:
            from src.database.connection import get_session
            from src.database.repositories.metadata import MetadataRepository

            async with get_session() as session:
                repo = MetadataRepository(session)
                await repo.set("trading.last_signal_tracking", value)
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to set last signal tracking: {e}")

    async def get_analyses(self, limit: int = 1000) -> list[AnalysisRecord]:
        """Get recent analyses with lazy loading.

        Args:
            limit: Maximum number of records to return

        Returns:
            List of recent AnalysisRecords
        """
        if not self._database_enabled:
            return []
        if self._analysis_cache is None:
            try:
                from src.database.connection import get_session
                from src.database.repositories.analysis import AnalysisRecordRepository

                async with get_session() as session:
                    repo = AnalysisRecordRepository(session)
                    self._analysis_cache = await repo.get_recent(limit)
            except Exception as e:
                logger.opt(exception=True).warning(f"Failed to get analyses: {e}")
                return []
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
        if not self._database_enabled:
            self._analysis_cache = None
            return

        try:
            from src.database.connection import get_session
            from src.database.repositories.analysis import AnalysisRecordRepository
            from src.database.repositories.metadata import MetadataRepository

            async with get_session() as session:
                analysis_repo = AnalysisRecordRepository(session)
                metadata_repo = MetadataRepository(session)

                await analysis_repo.create(record)
                logger.debug(f"Persisted analysis record: {input_data.symbol} {input_data.signal}")

                # Update metadata counters in same session
                total = await self.get_total_analyses()
                await metadata_repo.set("trading.total_analyses", total + 1)
                await metadata_repo.set("trading.last_run", datetime.now(UTC))

                if input_data.executed:
                    trades = await self.get_total_trades()
                    await metadata_repo.set("trading.total_trades", trades + 1)

            # Invalidate cache
            self._analysis_cache = None
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to record analysis: {e}")

    def __repr__(self) -> str:
        """Return string representation."""
        return "TradingStateManager(db_backed)"
