"""Event watcher base class for real-time market signal monitoring.

Base pattern generalized from TrumpWatcher:
- Poll-based daemon with configurable intervals
- Lazy initialization of heavy components (LLM, workflow)
- Deduplication via state tracking
- In-memory cooldown to prevent analysis storms
- Graceful shutdown (signal handlers managed by CLI orchestrator)
- Async throughout with concurrent analysis
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from loguru import logger
from rich.console import Console

from src.agents.event_triage import EventTriageAgent
from src.cache.historical import HistoricalCache
from src.daemon.events import BaseEvent, EventSignal, TriageResult, Urgency
from src.workflows import TradingWorkflow
from src.workflows.types import TradingWorkflowResult

if TYPE_CHECKING:
    from src.daemon.state.facade import DaemonState
    from src.di.container import AppContainer

console = Console()


@dataclass
class EventWatcherConfig:
    """Base configuration for EventWatcher."""

    poll_interval: int
    relevance_threshold: float
    cooldown_minutes: int
    max_concurrent_analyses: int
    period_days: int = 60


class EventWatcher(ABC):
    """Base class for event-driven stock analysis watchers.

    Subclasses implement _fetch_events() to poll specific data sources.
    Base class handles triage, cooldown, analysis orchestration, and signaling.
    """

    def __init__(
        self,
        config: EventWatcherConfig,
        historical_cache: HistoricalCache,
        container: AppContainer | None = None,
        signal_callback: Callable[[EventSignal], None] | None = None,
        discovery_mode: bool = False,
        discovery_callback: Callable[[list], None] | None = None,
        state: DaemonState | None = None,
    ) -> None:
        """Initialize event watcher.

        Args:
            config: Base configuration (poll interval, thresholds, etc.)
            historical_cache: Shared cache for market/news data
            container: Optional DI container (auto-created if not provided)
            signal_callback: Optional callback to persist signals (e.g., to state)
            discovery_mode: If True, route events to discovery engine instead of direct analysis
            discovery_callback: Callback to receive discovery candidates
            state: Optional DaemonState for WATCHLIST event persistence
        """
        from src.di.container import create_container

        self.poll_interval = config.poll_interval
        self.relevance_threshold = config.relevance_threshold
        self.cooldown_minutes = config.cooldown_minutes
        self.max_concurrent_analyses = config.max_concurrent_analyses
        self.period_days = config.period_days
        self.running = False
        self._signal_callback = signal_callback
        self._container = container or create_container()

        # Discovery mode support
        self._discovery_mode = discovery_mode
        self._discovery_callback = discovery_callback
        self._event_adapter = None

        # State manager for WATCHLIST events
        self._state = state

        # State tracking (in-memory)
        self._last_check: datetime | None = None
        self._symbol_cooldowns: dict[str, datetime] = {}

        # Lazy init (TrumpWatcher pattern)
        self._historical_cache = historical_cache
        self._triage_agent: EventTriageAgent | None = None
        self._workflow: TradingWorkflow | None = None

    @abstractmethod
    async def _fetch_events(self) -> list[BaseEvent]:
        """Fetch new events from source (source-specific implementation).

        Returns:
            List of new events since last check
        """
        ...

    def _init_components(self) -> None:
        """Lazy initialization of shared components."""
        llm_client = self._container.llm_client()

        if self._triage_agent is None:
            self._triage_agent = EventTriageAgent(llm_client)

        if self._workflow is None:
            self._workflow = self._container.workflow_meta(
                historical_cache=self._historical_cache,
                container=self._container,
            )
            logger.info(f"{self.__class__.__name__} workflow initialized")

    def _check_cooldown(self, symbol: str) -> bool:
        """Check if symbol is in cooldown period.

        Args:
            symbol: Stock ticker to check

        Returns:
            True if symbol can be analyzed (not in cooldown)
        """
        if symbol not in self._symbol_cooldowns:
            return True

        elapsed = datetime.now(UTC) - self._symbol_cooldowns[symbol]
        return elapsed.total_seconds() > (self.cooldown_minutes * 60)

    def _set_cooldown(self, symbol: str) -> None:
        """Set cooldown timestamp for symbol.

        Args:
            symbol: Stock ticker to put in cooldown
        """
        self._symbol_cooldowns[symbol] = datetime.now(UTC)
        logger.debug(f"{symbol} in cooldown for {self.cooldown_minutes}m")

    async def _analyze_stocks(self, symbols: list[str]) -> dict[str, TradingWorkflowResult]:
        """Run trading analysis for symbols with concurrency control.

        Args:
            symbols: Stock symbols to analyze

        Returns:
            Dict mapping symbol to analysis result
        """
        self._init_components()
        if self._workflow is None:
            msg = "Failed to initialize TradingWorkflow"
            raise RuntimeError(msg)
        logger.info(f"Analyzing {len(symbols)} symbols: {symbols}")

        semaphore = asyncio.Semaphore(self.max_concurrent_analyses)
        workflow = self._workflow

        async def analyze_one(symbol: str) -> tuple[str, TradingWorkflowResult | None]:
            async with semaphore:
                try:
                    result = await workflow.analyze(symbol, period_days=self.period_days)
                    return symbol, result
                except Exception as e:
                    logger.opt(exception=True).error(f"Failed to analyze {symbol}: {e}")
                    return symbol, None

        # Wrap tasks to handle exceptions
        async def safe_analyze(symbol: str) -> tuple[str, TradingWorkflowResult | None] | BaseException:
            try:
                return await analyze_one(symbol)
            except BaseException as e:
                # Re-raise control-flow exceptions so TaskGroup can cancel siblings promptly
                if isinstance(e, (asyncio.CancelledError, KeyboardInterrupt)):
                    raise
                return e

        # Run analyses in parallel using TaskGroup
        async with asyncio.TaskGroup() as tg:
            task_results = [tg.create_task(safe_analyze(s)) for s in symbols]

        raw_results = [task.result() for task in task_results]
        return self._process_analysis_results(raw_results, len(symbols))

    def _process_analysis_results(
        self,
        raw_results: list[tuple[str, TradingWorkflowResult | None] | BaseException],
        total_symbols: int,
    ) -> dict[str, TradingWorkflowResult]:
        """Process raw analysis results and extract successful ones.

        Args:
            raw_results: Raw results from parallel analyses
            total_symbols: Total number of symbols analyzed

        Returns:
            Dict mapping symbol to successful analysis results
        """
        results: dict[str, TradingWorkflowResult] = {}
        for entry in raw_results:
            if isinstance(entry, BaseException):
                if isinstance(entry, (asyncio.CancelledError, KeyboardInterrupt)):
                    raise entry
                logger.error(f"Analysis task failed: {entry}")
                continue
            symbol, result = entry
            if result:
                results[symbol] = result

        logger.info(f"Analysis complete: {len(results)}/{total_symbols} successful")
        return results

    def _emit_signal(self, signal: EventSignal) -> None:
        """Emit trading signal to console and persist via callback.

        Args:
            signal: Event signal with triage and analysis results
        """
        console.print()
        console.print("[bold cyan]═══ EVENT SIGNAL DETECTED ═══[/bold cyan]")
        console.print(f"[dim]{signal.signal_timestamp:%Y-%m-%d %H:%M:%S}[/dim]")
        console.print()

        # Event details
        console.print(f"[bold]Event Type:[/bold] {signal.event.event_type}")
        console.print(f"[bold]Source:[/bold] {signal.event.source}")
        console.print()

        # Triage result
        console.print(f"[bold yellow]Relevance:[/bold yellow] {signal.triage.relevance:.2f}")
        console.print(f"[bold]Urgency:[/bold] {signal.triage.urgency.value}")
        console.print(f"[bold]Sentiment:[/bold] {signal.triage.sentiment.value}")
        console.print(f"[bold]Reasoning:[/bold] {signal.triage.reasoning[:200]}...")
        console.print()

        # Analysis results
        console.print(f"[bold green]Analyzed Stocks ({len(signal.analyses)}):[/bold green]")
        for symbol, result in signal.analyses.items():
            action = result.decision.action.value
            color = {"BUY": "green", "SELL": "red"}.get(action, "yellow")
            console.print(
                f"  [bold]{symbol}[/bold]: [{color}]{action}[/{color}] "
                f"(confidence: {result.decision.confidence:.2f})"
            )

        console.print("[bold cyan]═══════════════════════════════[/bold cyan]")
        console.print()

        # Persist signal via callback if provided
        if self._signal_callback:
            try:
                self._signal_callback(signal)
            except Exception as e:
                logger.opt(exception=True).error(f"Signal callback failed: {e}")

    async def _triage_events(self, events: list[BaseEvent]) -> list[TriageResult | BaseException]:
        """Triage events with LLM in parallel."""

        async def safe_triage(event: object) -> TriageResult | BaseException:
            try:
                return await self._triage_agent.analyze(event)  # type: ignore[attr-defined, union-attr]
            except BaseException as e:
                return e

        async with asyncio.TaskGroup() as tg:
            triage_task_results = [tg.create_task(safe_triage(e)) for e in events]

        return [task.result() for task in triage_task_results]

    def _filter_relevant_events(
        self,
        events: list[BaseEvent],
        triage_results: list[TriageResult | BaseException],
    ) -> list[tuple[BaseEvent, TriageResult]]:
        """Filter triage results by relevance/urgency, log failures."""
        relevant = []
        for event, triage in zip(events, triage_results, strict=True):
            if isinstance(triage, BaseException):
                logger.error(f"Event triage failed: {triage}")
            elif triage.relevance >= self.relevance_threshold and triage.urgency == Urgency.IMMEDIATE:
                relevant.append((event, triage))
        return relevant

    async def _add_watchlist_candidates(
        self, events: list[BaseEvent], triage_results: list[TriageResult | BaseException]
    ) -> None:
        """Add WATCHLIST events to discovery candidates for later analysis.

        Args:
            events: List of events
            triage_results: Corresponding triage results
        """
        from datetime import timedelta

        from src.discovery.models import DiscoveryCandidate, DiscoverySource

        if not self._state:
            return

        watchlist_events = []
        for event, triage in zip(events, triage_results, strict=True):
            if isinstance(triage, BaseException):
                continue
            if triage.urgency == Urgency.WATCHLIST and triage.relevance >= self.relevance_threshold:
                watchlist_events.append((event, triage))

        if not watchlist_events:
            return

        # Convert to discovery candidates
        candidates = []
        now = datetime.now(UTC)
        ttl_hours = 24

        for event, triage in watchlist_events:
            for symbol in triage.symbols:
                candidate = DiscoveryCandidate(
                    symbol=symbol,
                    name="Unknown",
                    sector="Unknown",
                    sources=[DiscoverySource.EVENT_WATCHLIST],
                    composite_score=triage.relevance,
                    source_scores={"event_watchlist": triage.relevance},
                    discovery_timestamp=now,
                    ttl_expires_at=now + timedelta(hours=ttl_hours),
                    metadata={
                        "event_id": event.event_id,
                        "event_type": event.event_type,
                        "sentiment": triage.sentiment.value,
                        "confidence": triage.confidence,
                        "reasoning": triage.reasoning,
                    },
                )
                candidates.append(candidate)

        # Add to state
        try:
            await self._state.discovery.add_event_candidates(candidates)
            logger.info(f"Added {len(candidates)} WATCHLIST events to discovery candidates")
        except Exception as e:
            logger.opt(exception=True).error(f"Failed to add WATCHLIST candidates: {e}")

    def _extract_symbols_with_cooldown(self, relevant: list[tuple[BaseEvent, TriageResult]]) -> set[str]:
        """Extract symbols from relevant events and check cooldowns."""
        symbols_to_analyze = set()
        for _, triage in relevant:
            for symbol in triage.symbols:
                if self._check_cooldown(symbol):
                    symbols_to_analyze.add(symbol)
                else:
                    logger.debug(f"{symbol} skipped (in cooldown)")
        return symbols_to_analyze

    async def _run_cycle(self) -> None:
        """Main poll cycle (template method)."""
        self._init_components()
        if self._triage_agent is None:
            msg = "Failed to initialize EventTriageAgent"
            raise RuntimeError(msg)

        events = await self._fetch_events()
        if not events:
            logger.debug("No new events")
            return

        logger.info(f"Found {len(events)} new event(s)")
        triage_results = await self._triage_events(events)
        await self._add_watchlist_candidates(events, triage_results)
        relevant = self._filter_relevant_events(events, triage_results)

        if not relevant:
            logger.debug(
                f"No events above threshold (relevance>={self.relevance_threshold}, urgency=IMMEDIATE)"
            )
            return

        logger.info(f"Found {len(relevant)} high-relevance event(s)")

        # Route based on discovery mode
        if self._discovery_mode:
            await self._route_to_discovery(relevant)
        else:
            await self._route_to_direct_analysis(relevant)

    async def _route_to_discovery(self, relevant: list[tuple[BaseEvent, TriageResult]]) -> None:
        """Convert events to discovery candidates and feed to discovery engine.

        Args:
            relevant: List of (event, triage) tuples
        """
        if self._event_adapter is None:
            from src.daemon.adapters.event_discovery_adapter import EventDiscoveryAdapter

            self._event_adapter = EventDiscoveryAdapter(self._container.market_fetcher())

        all_candidates = []
        for event, triage in relevant:
            try:
                candidates = await self._event_adapter.convert_event_to_candidate(event, triage)
                all_candidates.extend(candidates)
            except Exception as e:
                logger.opt(exception=True).error(f"Failed to convert event to candidate: {e}")

        if all_candidates and self._discovery_callback:
            try:
                self._discovery_callback(all_candidates)
                logger.info(f"Routed {len(all_candidates)} event candidates to discovery engine")
            except Exception as e:
                logger.opt(exception=True).error(f"Discovery callback failed: {e}")

    async def _route_to_direct_analysis(self, relevant: list[tuple[BaseEvent, TriageResult]]) -> None:
        """Legacy direct analysis path (pre-discovery mode).

        Args:
            relevant: List of (event, triage) tuples
        """
        symbols_to_analyze = self._extract_symbols_with_cooldown(relevant)

        if not symbols_to_analyze:
            logger.info("All symbols in cooldown, skipping analysis")
            return

        symbols_list = sorted(symbols_to_analyze)[: self.max_concurrent_analyses]
        analyses = await self._analyze_stocks(symbols_list)

        for symbol in analyses:
            self._set_cooldown(symbol)

        signal = EventSignal(
            event=relevant[0][0],
            triage=relevant[0][1],
            analyses=analyses,
            signal_timestamp=datetime.now(UTC),
        )
        self._emit_signal(signal)

    async def run(self) -> None:
        """Run watcher daemon (signal handlers managed by CLI orchestrator)."""
        self.running = True

        console.print()
        console.print(f"[bold green]{self.__class__.__name__} Started[/bold green]")
        console.print(f"Poll interval: {self.poll_interval}s")
        console.print(f"Relevance threshold: {self.relevance_threshold}")
        console.print(f"Cooldown: {self.cooldown_minutes}m")
        console.print("[dim]Monitoring for events...[/dim]")
        console.print()

        while self.running:
            try:
                await self._run_cycle()
                await asyncio.sleep(self.poll_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in watcher loop: {e}")
                await asyncio.sleep(60)

        console.print()
        console.print(f"[bold yellow]{self.__class__.__name__} Stopped[/bold yellow]")
        logger.info(f"{self.__class__.__name__} shutdown complete")

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"{self.__class__.__name__}(poll_interval={self.poll_interval}s, "
            f"threshold={self.relevance_threshold})"
        )
