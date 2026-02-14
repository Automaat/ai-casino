"""Pre-market screening task."""

from __future__ import annotations

from datetime import UTC, datetime

from loguru import logger

from src.daemon.tasks.base import TaskExecutor


class PreMarketScreeningTask(TaskExecutor):
    """Pre-market screening task (7:00 AM ET)."""

    @property
    def task_name(self) -> str:
        """Task display name."""
        return "Pre-Market Screening"

    async def get_last_run(self) -> datetime | None:
        """Get last run timestamp."""
        return await self.components.state.metadata.get("pre_market_screening.last_run")

    async def record_success(self) -> None:
        """Record successful execution."""
        await self.components.state.metadata.set("pre_market_screening.last_run", datetime.now(UTC))

    async def execute(self) -> None:
        """Execute pre-market screening logic."""
        config = self.components.config.pre_market
        screener = self.container.pre_market_screener()

        result = await screener.screen(
            universe=config.universe,
            top_n=config.top_n,
            gap_threshold=config.gap_threshold_percent,
            min_volume_ratio=config.min_volume_ratio,
            min_score=config.min_composite_score,
            timeout_seconds=config.timeout_seconds,
            earnings_lookahead_days=config.earnings_lookahead_days,
            overnight_news_hours=config.overnight_news_hours,
            gap_weight=config.gap_weight,
            volume_weight=config.volume_weight,
            catalyst_weight=config.catalyst_weight,
        )

        logger.info(
            f"Pre-market screening found {len(result.candidates)} candidates "
            f"(expires {result.expires_at.strftime('%H:%M %Z')})"
        )

        await self.components.state.discovery.record_pre_market_candidates(
            candidates=result.candidates,
            expires_at=result.expires_at,
        )
