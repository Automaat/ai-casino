"""v1 service for storing and retrieving analyses via DB."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from loguru import logger
from result import Err, Ok, Result

from src.daemon.state.models import AnalysisRecord

if TYPE_CHECKING:
    from src.database.engine import DatabaseEngine
    from src.workflows.types import TradingWorkflowResult


class AnalysisService:
    """v1 service for storing and retrieving analyses via DB."""

    def __init__(self, database_engine: DatabaseEngine | None) -> None:
        """Initialize with optional database engine.

        Args:
            database_engine: Database engine for persistence, or None to disable
        """
        self._database_engine = database_engine

    async def record(self, result: TradingWorkflowResult) -> Result[AnalysisRecord | None, Exception]:
        """Persist a TradingWorkflowResult as an AnalysisRecord.

        Args:
            result: Workflow result to persist

        Returns:
            Ok(AnalysisRecord) on success, Ok(None) if no engine, Err(Exception) on failure
        """
        if self._database_engine is None:
            return Ok(None)

        try:
            from src.di.providers.database import create_analysis_repository

            analysis_record = AnalysisRecord(
                symbol=result.symbol,
                timestamp=datetime.now(UTC),
                signal=result.decision.action.value,
                confidence=result.decision.confidence,
                executed_trade=False,
                trading_session=result.trading_session,
                is_paper_trade=True,
                rsi=result.technical.rsi,
                macd_hist=result.technical.macd_hist,
                reasoning=result.decision.reasoning,
            )
            repo = create_analysis_repository(self._database_engine)
            async with repo:
                await repo.create(analysis_record)
            return Ok(analysis_record)
        except Exception as e:
            logger.opt(exception=True).error(f"Failed to persist analysis for {result.symbol}: {e}")
            return Err(e)

    async def get_recent(self, limit: int = 1000) -> Result[list[AnalysisRecord], Exception]:
        """Get recent analysis records.

        Args:
            limit: Maximum number of records to return

        Returns:
            Ok(list[AnalysisRecord]) or Err(Exception)
        """
        if self._database_engine is None:
            return Ok([])

        try:
            from src.di.providers.database import create_analysis_repository

            repo = create_analysis_repository(self._database_engine)
            async with repo:
                records = await repo.get_recent(limit)
            return Ok(records)
        except Exception as e:
            logger.opt(exception=True).error(f"Failed to fetch recent analyses: {e}")
            return Err(e)

    async def get_by_symbol(self, symbol: str, limit: int = 100) -> Result[list[AnalysisRecord], Exception]:
        """Get analysis records for a specific symbol.

        Args:
            symbol: Stock ticker symbol
            limit: Maximum number of records to return

        Returns:
            Ok(list[AnalysisRecord]) or Err(Exception)
        """
        if self._database_engine is None:
            return Ok([])

        try:
            from src.di.providers.database import create_analysis_repository

            repo = create_analysis_repository(self._database_engine)
            async with repo:
                records = await repo.get_by_symbol(symbol, limit)
            return Ok(records)
        except Exception as e:
            logger.opt(exception=True).error(f"Failed to fetch analyses for {symbol}: {e}")
            return Err(e)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"AnalysisService(db={'enabled' if self._database_engine else 'disabled'})"
