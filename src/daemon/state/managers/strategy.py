"""Strategy state manager."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from loguru import logger
from pydantic import PrivateAttr

from src.daemon.state.managers.base import StateManager
from src.daemon.state.models import (
    DegradationRecord,
    GamePlanRecord,
    HealthReportRecord,
    PaperTradingReportRecord,
    TradeJournalRecord,
)

if TYPE_CHECKING:
    from src.daemon.degradation import DegradationContext


class StrategyStateManager(StateManager):
    """Daily planning, degradation tracking, error logging."""

    _database_enabled: bool = PrivateAttr(default=False)
    _game_plan_cache: list[GamePlanRecord] | None = PrivateAttr(default=None)
    _degradation_cache: list[DegradationRecord] | None = PrivateAttr(default=None)
    _health_report_cache: list[HealthReportRecord] | None = PrivateAttr(default=None)
    _trade_journal_cache: list[TradeJournalRecord] | None = PrivateAttr(default=None)
    _paper_trading_cache: list[PaperTradingReportRecord] | None = PrivateAttr(default=None)

    def enable_database(self) -> None:
        """Enable database persistence."""
        self._database_enabled = True
        logger.debug("StrategyStateManager database enabled")

    async def get_last_game_plan(self) -> datetime | None:
        """Get last game plan timestamp from DB."""
        if not self._database_enabled:
            return None
        try:
            from src.database.connection import get_session
            from src.database.repositories.metadata import MetadataRepository

            async with get_session() as session:
                repo = MetadataRepository(session)
                return await repo.get_datetime("strategy.last_game_plan")
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to get last game plan: {e}")
            return None

    async def get_last_degradation(self) -> datetime | None:
        """Get last degradation timestamp from DB."""
        if not self._database_enabled:
            return None
        try:
            from src.database.connection import get_session
            from src.database.repositories.metadata import MetadataRepository

            async with get_session() as session:
                repo = MetadataRepository(session)
                return await repo.get_datetime("strategy.last_degradation")
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to get last degradation: {e}")
            return None

    async def get_last_health_check(self) -> datetime | None:
        """Get last health check timestamp from DB."""
        if not self._database_enabled:
            return None
        try:
            from src.database.connection import get_session
            from src.database.repositories.metadata import MetadataRepository

            async with get_session() as session:
                repo = MetadataRepository(session)
                return await repo.get_datetime("strategy.last_health_check")
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to get last health check: {e}")
            return None

    async def set_last_health_check(self, value: datetime | None) -> None:
        """Set last health check timestamp in DB."""
        if not self._database_enabled or value is None:
            return
        try:
            from src.database.connection import get_session
            from src.database.repositories.metadata import MetadataRepository

            async with get_session() as session:
                repo = MetadataRepository(session)
                await repo.set("strategy.last_health_check", value)
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to set last health check: {e}")

    async def get_market_events(self, limit: int | None = None) -> list[dict]:
        """Get market events from DB metadata.

        Args:
            limit: Max number of events to return (optional)

        Returns:
            List of market events
        """
        if not self._database_enabled:
            return []
        try:
            from src.database.connection import get_session
            from src.database.repositories.metadata import MetadataRepository

            async with get_session() as session:
                repo = MetadataRepository(session)
                value = await repo.get("strategy.market_events")
                events = value if isinstance(value, list) else []
                if limit is not None and limit > 0:
                    return events[-limit:]
                return events
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to get market events: {e}")
            return []

    async def get_errors(self) -> list[str]:
        """Get errors from DB metadata."""
        if not self._database_enabled:
            return []
        try:
            from src.database.connection import get_session
            from src.database.repositories.metadata import MetadataRepository

            async with get_session() as session:
                repo = MetadataRepository(session)
                value = await repo.get("strategy.errors")
                return value if isinstance(value, list) else []
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to get errors: {e}")
            return []

    async def get_game_plan_history(self, limit: int = 30) -> list[GamePlanRecord]:
        """Get game plan history with lazy loading."""
        if not self._database_enabled:
            return []
        if self._game_plan_cache is None:
            try:
                from src.database.connection import get_session
                from src.database.repositories.game_plan import GamePlanRecordRepository

                async with get_session() as session:
                    repo = GamePlanRecordRepository(session)
                    self._game_plan_cache = await repo.get_recent(limit)
            except Exception as e:
                logger.opt(exception=True).warning(f"Failed to get game plan history: {e}")
                return []
        return self._game_plan_cache

    async def get_degradation_history(self, limit: int = 100) -> list[DegradationRecord]:
        """Get degradation history with lazy loading."""
        if not self._database_enabled:
            return []
        if self._degradation_cache is None:
            try:
                from src.database.connection import get_session
                from src.database.repositories.degradation import DegradationRecordRepository

                async with get_session() as session:
                    repo = DegradationRecordRepository(session)
                    self._degradation_cache = await repo.get_recent(limit)
            except Exception as e:
                logger.opt(exception=True).warning(f"Failed to get degradation history: {e}")
                return []
        return self._degradation_cache

    async def record_game_plan(
        self,
        priority_symbols: list[str],
        risk_stance: str,
        sector_focus: list[str],
    ) -> None:
        """Record game plan generation."""
        now = datetime.now(UTC)
        record = GamePlanRecord(
            timestamp=now,
            priority_symbols=priority_symbols,
            risk_stance=risk_stance,
            sector_focus=sector_focus,
        )

        if not self._database_enabled:
            self._game_plan_cache = None
            return

        try:
            from src.database.connection import get_session
            from src.database.repositories.game_plan import GamePlanRecordRepository
            from src.database.repositories.metadata import MetadataRepository

            async with get_session() as session:
                game_plan_repo = GamePlanRecordRepository(session)
                metadata_repo = MetadataRepository(session)
                await game_plan_repo.create(record)
                await metadata_repo.set("strategy.last_game_plan", now)
            self._game_plan_cache = None
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to record game plan: {e}")

    async def record_degradation(self, context: DegradationContext) -> None:
        """Record degradation event."""
        now = datetime.now(UTC)
        record = DegradationRecord(
            timestamp=now,
            tier=context.tier.value,
            unavailable_services=context.unavailable_services,
            confidence_adjustment=context.confidence_adjustment,
            halt_reason=context.halt_reason,
        )

        if not self._database_enabled:
            self._degradation_cache = None
            return

        try:
            from src.database.connection import get_session
            from src.database.repositories.degradation import DegradationRecordRepository
            from src.database.repositories.metadata import MetadataRepository

            async with get_session() as session:
                degradation_repo = DegradationRecordRepository(session)
                metadata_repo = MetadataRepository(session)
                await degradation_repo.create(record)
                await metadata_repo.set("strategy.last_degradation", now)
            self._degradation_cache = None
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to record degradation: {e}")

    async def record_error(self, error: str) -> None:
        """Record an error to metadata."""
        if not self._database_enabled:
            return

        timestamp = datetime.now(tz=UTC).isoformat()
        error_entry = f"{timestamp}: {error}"

        try:
            from src.database.connection import get_session
            from src.database.repositories.metadata import MetadataRepository

            async with get_session() as session:
                repo = MetadataRepository(session)
                # Get existing errors
                value = await repo.get("strategy.errors")
                errors = value if isinstance(value, list) else []
                errors.append(error_entry)

                # Cap at 100 errors
                if len(errors) > 100:
                    errors = errors[-50:]

                await repo.set("strategy.errors", errors)
        except Exception as e:
            # Don't log with exception=True to avoid infinite recursion
            logger.warning(f"Failed to record error to database: {e}")

    async def record_health_report(self, report: HealthReportRecord) -> None:
        """Record health report."""
        if not self._database_enabled:
            self._health_report_cache = None
            return
        try:
            from src.database.connection import get_session
            from src.database.repositories.health import HealthReportRepository
            from src.database.repositories.metadata import MetadataRepository

            async with get_session() as session:
                health_repo = HealthReportRepository(session)
                metadata_repo = MetadataRepository(session)
                await health_repo.create(report)
                await metadata_repo.set("strategy.last_health_check", report.timestamp)
            self._health_report_cache = None
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to record health report: {e}")

    async def get_recent_health_reports(self, limit: int = 100) -> list[HealthReportRecord]:
        """Get recent health reports."""
        if not self._database_enabled:
            return []
        if self._health_report_cache is None:
            try:
                from src.database.connection import get_session
                from src.database.repositories.health import HealthReportRepository

                async with get_session() as session:
                    repo = HealthReportRepository(session)
                    self._health_report_cache = await repo.get_recent(limit)
            except Exception as e:
                logger.opt(exception=True).warning(f"Failed to get health reports: {e}")
                return []
        return self._health_report_cache

    async def record_trade_journal(self, journal: TradeJournalRecord) -> None:
        """Record trade journal."""
        if not self._database_enabled:
            self._trade_journal_cache = None
            return
        try:
            from src.database.connection import get_session
            from src.database.repositories.journal import TradeJournalRepository

            async with get_session() as session:
                journal_repo = TradeJournalRepository(session)
                await journal_repo.create(journal)
            self._trade_journal_cache = None
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to record trade journal: {e}")

    async def get_recent_trade_journals(self, limit: int = 30) -> list[TradeJournalRecord]:
        """Get recent trade journals."""
        if not self._database_enabled:
            return []
        if self._trade_journal_cache is None:
            try:
                from src.database.connection import get_session
                from src.database.repositories.journal import TradeJournalRepository

                async with get_session() as session:
                    repo = TradeJournalRepository(session)
                    self._trade_journal_cache = await repo.get_recent(limit)
            except Exception as e:
                logger.opt(exception=True).warning(f"Failed to get trade journals: {e}")
                return []
        return self._trade_journal_cache

    async def record_paper_trading_report(self, report: PaperTradingReportRecord) -> None:
        """Record paper trading validation report."""
        if not self._database_enabled:
            self._paper_trading_cache = None
            return
        try:
            from src.database.connection import get_session
            from src.database.repositories.paper_trading import PaperTradingReportRepository

            async with get_session() as session:
                paper_trading_repo = PaperTradingReportRepository(session)
                await paper_trading_repo.create(report)
            self._paper_trading_cache = None
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to record paper trading report: {e}")

    async def get_latest_paper_trading_report(self) -> PaperTradingReportRecord | None:
        """Get latest paper trading report."""
        if not self._database_enabled:
            return None
        try:
            from src.database.connection import get_session
            from src.database.repositories.paper_trading import PaperTradingReportRepository

            async with get_session() as session:
                repo = PaperTradingReportRepository(session)
                return await repo.get_latest()
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to get paper trading report: {e}")
            return None

    async def get_recent_paper_trading_reports(self, limit: int = 10) -> list[PaperTradingReportRecord]:
        """Get recent paper trading reports."""
        if not self._database_enabled:
            return []
        if self._paper_trading_cache is None:
            try:
                from src.database.connection import get_session
                from src.database.repositories.paper_trading import PaperTradingReportRepository

                async with get_session() as session:
                    repo = PaperTradingReportRepository(session)
                    self._paper_trading_cache = await repo.get_recent(limit)
            except Exception as e:
                logger.opt(exception=True).warning(f"Failed to get paper trading reports: {e}")
                return []
        return self._paper_trading_cache

    def __repr__(self) -> str:
        """Return string representation."""
        return "StrategyStateManager()"
