"""Game plan task on v1 task framework."""

import time
from datetime import datetime
from typing import TYPE_CHECKING

from loguru import logger

from src.daemon.state.models import GamePlanRecord
from src.v1.tasks.interface import Task
from src.v1.tasks.models import WEEKDAYS, TaskResult, TaskSchedule

if TYPE_CHECKING:
    from src.agents.game_plan import GamePlanAgent
    from src.daemon.broker_manager import BrokerManager
    from src.daemon.config.portfolio import GamePlanConfig
    from src.daemon.scheduler import MarketScheduler
    from src.daemon.state import DaemonState


class GamePlanTask(Task):
    """Daily game plan generation via agentic GamePlanAgent."""

    def __init__(
        self,
        agent: GamePlanAgent,
        state: DaemonState,
        broker_manager: BrokerManager,
        config: GamePlanConfig,
        scheduler: MarketScheduler,
    ) -> None:
        """Initialize game plan task.

        Args:
            agent: Agentic game plan agent
            state: Daemon state for persistence
            broker_manager: Broker manager for watchlist
            config: Game plan configuration
            scheduler: Market scheduler for timezone
        """
        self._agent = agent
        self._state = state
        self._broker_manager = broker_manager
        self._config = config
        self._scheduler = scheduler

    @property
    def name(self) -> str:
        """Task identifier."""
        return "game_plan"

    @property
    def schedule(self) -> TaskSchedule:
        """Schedule from config."""
        return TaskSchedule(
            time=self._config.generation_time,
            days=WEEKDAYS,
            enabled=self._config.enabled,
        )

    async def execute(self) -> TaskResult:
        """Generate game plan and persist.

        Attempts generation up to max_retries times if confidence is below min_confidence.

        Returns:
            TaskResult with outcome
        """
        start = time.monotonic()

        watchlist = await self._broker_manager.get_merged_watchlist()
        logger.info(f"Generating game plan for {len(watchlist)} symbols")

        plan = None
        for attempt in range(1, self._config.max_retries + 1):
            candidate = await self._agent.generate(
                watchlist,
                timezone=self._scheduler.timezone,
            )
            if candidate.confidence >= self._config.min_confidence:
                plan = candidate
                break
            if attempt < self._config.max_retries:
                logger.warning(
                    f"Game plan attempt {attempt}/{self._config.max_retries} low confidence "
                    f"({candidate.confidence:.2f} < {self._config.min_confidence:.2f}), retrying"
                )
            else:
                logger.warning(
                    f"Game plan attempt {attempt}/{self._config.max_retries} low confidence "
                    f"({candidate.confidence:.2f} < {self._config.min_confidence:.2f}), "
                    "accepting low-confidence plan on final attempt"
                )
            plan = candidate

        if plan is None:
            msg = "Game plan generation produced no result"
            raise RuntimeError(msg)

        await self._state.record_game_plan(
            GamePlanRecord(
                timestamp=plan.generated_at,
                priority_symbols=plan.priority_symbols,
                risk_stance=plan.risk_stance,
                sector_focus=plan.sector_focus,
                reasoning=plan.reasoning,
                confidence=plan.confidence,
                overnight_summary=plan.overnight_summary,
                key_levels=plan.key_levels,
                generated_at=plan.generated_at,
            )
        )

        duration = time.monotonic() - start
        msg = (
            f"{len(plan.priority_symbols)} priority symbols, "
            f"stance={plan.risk_stance}, confidence={plan.confidence:.2f}"
        )
        if plan.confidence < self._config.min_confidence:
            logger.warning(f"Game plan complete (low confidence after all retries): {msg}")
        else:
            logger.info(f"Game plan complete: {msg}")

        return TaskResult(
            task_name=self.name,
            success=True,
            duration_seconds=duration,
            message=msg,
        )

    async def last_run_at(self) -> datetime | None:
        """Get last game plan timestamp from state."""
        return await self._state.get_last_game_plan()

    def __repr__(self) -> str:
        """String representation."""
        return f"GamePlanTask(enabled={self._config.enabled}, time={self._config.generation_time})"
