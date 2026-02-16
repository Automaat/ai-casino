"""Event batch evaluator for discovery candidate evaluation."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from src.agents.supervisor.agent import TradingSupervisor
    from src.agents.supervisor.models import CandidateEvaluationContext, CandidateRanking
    from src.daemon.broker_manager import BrokerManager
    from src.daemon.config import DaemonConfig
    from src.daemon.state import DaemonState
    from src.data.broker import AlpacaBroker
    from src.discovery.models import DiscoveryCandidate


class EventBatchEvaluator:
    """Evaluates accumulated event-based discovery candidates in batches."""

    def __init__(
        self,
        supervisor: TradingSupervisor | None,
        state: DaemonState,
        config: DaemonConfig,
        broker_manager: BrokerManager,
        broker: AlpacaBroker | None = None,
    ) -> None:
        """Initialize batch evaluator.

        Args:
            supervisor: Trading supervisor for candidate evaluation (optional)
            state: Daemon state manager
            config: Daemon configuration
            broker_manager: Broker manager for watchlist access
            broker: Optional broker for portfolio access
        """
        self.supervisor = supervisor
        self.state = state
        self.config = config
        self.broker_manager = broker_manager
        self.broker = broker
        self._last_batch_evaluation: datetime | None = None

    async def should_evaluate_batch(self) -> bool:
        """Check if enough time passed since last evaluation.

        Returns:
            True if batch should be evaluated
        """
        if not self.config.event_integration.enable_discovery_integration:
            return False

        if self._last_batch_evaluation is None:
            return True

        interval_minutes = self.config.event_integration.batch_evaluation_interval_minutes
        elapsed = datetime.now(UTC) - self._last_batch_evaluation
        return elapsed > timedelta(minutes=interval_minutes)

    async def evaluate_batch(self) -> list[str]:
        """Evaluate accumulated candidates, return approved symbols.

        Separates urgent (score >= threshold) vs normal
        Urgent bypass batching if config enabled
        Apply portfolio filters before approval

        Returns:
            List of approved symbols for watchlist
        """
        candidates = await self.state.discovery.get_active_discovery_candidates()

        if not candidates:
            logger.debug("No event candidates to evaluate")
            return []

        urgent_threshold = self.config.event_integration.urgent_evaluation_threshold
        urgent_bypass = self.config.event_integration.urgent_bypass_batch

        urgent_candidates = [c for c in candidates if c.composite_score >= urgent_threshold]
        normal_candidates = [c for c in candidates if c.composite_score < urgent_threshold]

        approved_symbols: list[str] = []

        if urgent_candidates and urgent_bypass:
            logger.info(f"Bypassing batch for {len(urgent_candidates)} urgent candidates")
            ranking = await self._evaluate_candidate_group(urgent_candidates, bypass_batch=True)
            approved_symbols.extend(ranking.add_watchlist)

        if normal_candidates:
            max_batch = self.config.event_integration.max_candidates_per_batch
            batch_candidates = normal_candidates[:max_batch]
            logger.info(f"Evaluating batch of {len(batch_candidates)} normal candidates")
            ranking = await self._evaluate_candidate_group(batch_candidates, bypass_batch=False)
            approved_symbols.extend(ranking.add_watchlist)

        self._last_batch_evaluation = datetime.now(UTC)
        logger.info(f"Batch evaluation complete: {len(approved_symbols)} candidates approved")

        return approved_symbols

    async def _evaluate_candidate_group(
        self, candidates: list[DiscoveryCandidate], bypass_batch: bool
    ) -> CandidateRanking:
        """Evaluate a group of candidates using supervisor.

        Args:
            candidates: Candidates to evaluate
            bypass_batch: If True, skip capacity checks (urgent)

        Returns:
            CandidateRanking with recommendations
        """
        from src.agents.supervisor.models import CandidateEvaluationContext, CandidateRanking
        from src.strategies.session import TradingSession

        if not self.supervisor:
            logger.warning("Supervisor not available, returning empty ranking")
            return CandidateRanking(
                evaluations=[],
                add_watchlist=[],
                defer=[],
                skip=[],
                priority_order=[],
                overall_reasoning="Supervisor not configured",
                warnings=["Supervisor not available for candidate evaluation"],
            )

        portfolio_symbols: list[str] = []
        watchlist_symbols = await self.state.get_active_discovery_symbols()
        max_size = self.config.discovery.max_watchlist_size
        capacity = max_size - len(watchlist_symbols) if not bypass_batch else 999

        context: CandidateEvaluationContext = CandidateEvaluationContext(
            candidates=candidates,
            market_regime=None,
            portfolio_symbols=portfolio_symbols,
            watchlist_symbols=watchlist_symbols,
            watchlist_capacity=capacity,
            sector_exposure={},
            recent_discovery_outcomes=None,
            time_budget_ms=30000,
            session=TradingSession.REGULAR,
        )

        ranking = await self.supervisor.evaluate_candidates(context)

        logger.debug(
            f"Candidate evaluation: {len(ranking.add_watchlist)} add, "
            f"{len(ranking.defer)} defer, {len(ranking.skip)} skip"
        )

        return ranking

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"EventBatchEvaluator(enabled={self.config.event_integration.enable_discovery_integration}, "
            f"interval={self.config.event_integration.batch_evaluation_interval_minutes}m)"
        )
