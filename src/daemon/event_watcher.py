"""Event watcher base class for real-time market signal monitoring.

Base pattern generalized from TrumpWatcher:
- Poll-based daemon with configurable intervals
- Lazy initialization of heavy components (LLM, workflow)
- Deduplication via state tracking
- In-memory cooldown to prevent analysis storms
- Graceful shutdown (signal handlers managed by CLI orchestrator)
- Async throughout with concurrent analysis
"""

import asyncio
from abc import ABC, abstractmethod
from datetime import UTC, datetime

from loguru import logger
from rich.console import Console

from src.agents.event_triage import EventTriageAgent
from src.cache.historical import HistoricalCache
from src.daemon.events import BaseEvent, EventSignal, Urgency
from src.data.fundamental import FundamentalDataFetcher
from src.data.market import MarketDataFetcher
from src.data.news import NewsFetcher
from src.models.llm import LLMClient
from src.models.sentiment import get_finbert_sentiment
from src.workflows.trading import TradingWorkflow
from src.workflows.types import TradingWorkflowResult

console = Console()


class EventWatcher(ABC):
    """Base class for event-driven stock analysis watchers.

    Subclasses implement _fetch_events() to poll specific data sources.
    Base class handles triage, cooldown, analysis orchestration, and signaling.
    """

    def __init__(  # noqa: PLR0913
        self,
        poll_interval: int,
        relevance_threshold: float,
        cooldown_minutes: int,
        max_concurrent_analyses: int,
        historical_cache: HistoricalCache,
        signal_callback: callable[[EventSignal], None] | None = None,
    ) -> None:
        """Initialize event watcher.

        Args:
            poll_interval: Seconds between poll cycles
            relevance_threshold: Minimum relevance score to trigger analysis (0.0-1.0)
            cooldown_minutes: Minutes to wait before re-analyzing same symbol
            max_concurrent_analyses: Maximum symbols to analyze per cycle
            historical_cache: Shared cache for market/news data
            signal_callback: Optional callback to persist signals (e.g., to state)
        """
        self.poll_interval = poll_interval
        self.relevance_threshold = relevance_threshold
        self.cooldown_minutes = cooldown_minutes
        self.max_concurrent_analyses = max_concurrent_analyses
        self.running = False
        self._signal_callback = signal_callback

        # State tracking (in-memory)
        self._last_check: datetime | None = None
        self._symbol_cooldowns: dict[str, datetime] = {}

        # Lazy init (TrumpWatcher pattern)
        self._historical_cache = historical_cache
        self._triage_agent: EventTriageAgent | None = None
        self._workflow: TradingWorkflow | None = None
        self._llm: LLMClient | None = None

    @abstractmethod
    async def _fetch_events(self) -> list[BaseEvent]:
        """Fetch new events from source (source-specific implementation).

        Returns:
            List of new events since last check
        """
        ...

    def _init_components(self) -> None:
        """Lazy initialization of shared components."""
        if self._llm is None:
            self._llm = LLMClient()

        if self._triage_agent is None:
            self._triage_agent = EventTriageAgent(self._llm)

        if self._workflow is None:
            # Initialize TradingWorkflow (same as TrumpWatcher lines 89-108)
            market_fetcher = MarketDataFetcher(
                use_alpha_vantage=False, historical_cache=self._historical_cache
            )
            news_fetcher = NewsFetcher(historical_cache=self._historical_cache)
            finbert = get_finbert_sentiment()
            fundamental_fetcher = FundamentalDataFetcher(historical_cache=self._historical_cache)

            self._workflow = TradingWorkflow(
                self._llm,
                market_fetcher,
                news_fetcher,
                finbert,
                fundamental_fetcher,
                broker=None,
                metrics_tracker=None,
                use_meta_agent=True,
                trump_mode=False,
                historical_cache=self._historical_cache,
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
        logger.info(f"Analyzing {len(symbols)} symbols: {symbols}")

        results: dict[str, TradingWorkflowResult] = {}
        semaphore = asyncio.Semaphore(self.max_concurrent_analyses)

        async def analyze_one(symbol: str) -> tuple[str, TradingWorkflowResult | None]:
            async with semaphore:
                try:
                    result = await self._workflow.analyze(symbol, period_days=30)
                    return symbol, result
                except Exception as e:
                    logger.error(f"Failed to analyze {symbol}: {e}")
                    return symbol, None

        tasks = [analyze_one(s) for s in symbols]
        raw_results = await asyncio.gather(*tasks)

        for symbol, result in raw_results:
            if result:
                results[symbol] = result

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
                logger.error(f"Signal callback failed: {e}")

    async def _run_cycle(self) -> None:
        """Main poll cycle (template method)."""
        self._init_components()

        # 1. Fetch new events
        events = await self._fetch_events()
        if not events:
            logger.debug("No new events")
            return

        logger.info(f"Found {len(events)} new event(s)")

        # 2. Triage events with LLM
        triage_tasks = [self._triage_agent.analyze(e) for e in events]
        triage_results = await asyncio.gather(*triage_tasks)

        # 3. Filter by relevance threshold and urgency
        relevant = [
            (event, triage)
            for event, triage in zip(events, triage_results, strict=True)
            if triage.relevance >= self.relevance_threshold and triage.urgency == Urgency.IMMEDIATE
        ]

        if not relevant:
            logger.debug(
                f"No events above threshold (relevance>={self.relevance_threshold}, urgency=IMMEDIATE)"
            )
            return

        logger.info(f"Found {len(relevant)} high-relevance event(s)")

        # 4. Extract symbols and check cooldowns
        symbols_to_analyze = set()
        for _, triage in relevant:
            for symbol in triage.symbols:
                if self._check_cooldown(symbol):
                    symbols_to_analyze.add(symbol)
                else:
                    logger.debug(f"{symbol} skipped (in cooldown)")

        if not symbols_to_analyze:
            logger.info("All symbols in cooldown, skipping analysis")
            return

        # 5. Run trading analysis (limit to max_concurrent_analyses)
        symbols_list = sorted(symbols_to_analyze)[: self.max_concurrent_analyses]
        analyses = await self._analyze_stocks(symbols_list)

        # 6. Set cooldowns only for successfully analyzed symbols
        for symbol in analyses:
            self._set_cooldown(symbol)

        # 7. Emit signal (use first relevant event as primary)
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
