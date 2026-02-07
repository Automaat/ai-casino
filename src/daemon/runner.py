"""Main daemon runner for autonomous trading."""

import asyncio
import os
import signal
from datetime import datetime
from pathlib import Path

from loguru import logger
from rich.console import Console

from src.daemon.config import DaemonConfig
from src.daemon.scheduler import MarketScheduler
from src.daemon.state import DaemonState
from src.data.broker import AlpacaBroker
from src.data.fundamental import FundamentalDataFetcher
from src.data.market import MarketDataFetcher
from src.data.news import NewsFetcher
from src.models.llm import LLMClient
from src.models.sentiment import get_finbert_sentiment
from src.optimization.param_store import OptimizedParamStore
from src.workflows.trading import TradingWorkflow
from src.workflows.types import TradingWorkflowResult

console = Console()


class DaemonRunner:
    """Main daemon runner for autonomous trading."""

    def __init__(self, config: DaemonConfig) -> None:
        """Initialize daemon runner.

        Args:
            config: Daemon configuration
        """
        self.config = config
        self.scheduler = MarketScheduler(
            start_time=config.schedule.start_time,
            end_time=config.schedule.end_time,
            timezone=config.schedule.timezone,
            enable_pre_market=config.schedule.enable_pre_market,
            enable_after_hours=config.schedule.enable_after_hours,
            enable_screening=config.screening.enabled,
            screen_time=config.screening.screen_time,
            screen_days=config.screening.screen_days,
            optimization_time=config.optimization.optimization_time,
            optimization_days=config.optimization.optimization_days,
        )
        self.state = DaemonState.load(config.state.state_file)
        self.running = False
        self._workflow: TradingWorkflow | None = None
        self.param_store: OptimizedParamStore | None = None
        self._daemon_optimizer = None
        if config.optimization.enabled:
            self.param_store = OptimizedParamStore(config.optimization.params_file)
            from src.daemon.optimization import DaemonOptimizer

            self._daemon_optimizer = DaemonOptimizer(
                param_store=self.param_store,
                n_trials=config.optimization.n_trials,
                min_trades=config.optimization.min_trades,
            )
        self.broker: AlpacaBroker | None = None
        if config.auto_trade:
            api_key = os.getenv("ALPACA_API_KEY")
            secret_key = os.getenv("ALPACA_SECRET_KEY")
            if not api_key or not secret_key:
                msg = "auto_trade=true requires ALPACA_API_KEY and ALPACA_SECRET_KEY env vars"
                raise ValueError(msg)
            self.broker = AlpacaBroker(paper=True)
            logger.info("Alpaca broker initialized for auto-trading")
        elif os.getenv("ALPACA_API_KEY") and os.getenv("ALPACA_SECRET_KEY"):
            try:
                self.broker = AlpacaBroker(paper=True)
                logger.info("Alpaca broker initialized for watchlist merging")
            except Exception as e:
                logger.exception(f"Failed to initialize broker: {e}")
                self.broker = None
        logger.info(f"DaemonRunner initialized with {config}")

    def _init_workflow(self) -> TradingWorkflow:
        """Initialize trading workflow (lazy initialization)."""
        if self._workflow is None:
            llm_client = LLMClient()
            market_fetcher = MarketDataFetcher(use_alpha_vantage=False)
            news_fetcher = NewsFetcher()
            finbert = get_finbert_sentiment()
            fundamental_fetcher = FundamentalDataFetcher()

            self._workflow = TradingWorkflow(
                llm_client,
                market_fetcher,
                news_fetcher,
                finbert,
                fundamental_fetcher,
                broker=self.broker,
                metrics_tracker=None,
                use_meta_agent=True,
                param_store=self.param_store,
            )
            logger.info("Trading workflow initialized")
        return self._workflow

    def _get_merged_watchlist(self) -> list[str]:
        """Get watchlist merged with broker positions and screening candidates.

        Returns:
            Deduplicated list combining config watchlist, broker positions,
            and latest screening candidates. Config order preserved, new
            symbols appended alphabetically per source.
        """
        # Source 1: config watchlist (preserve order)
        merged_watchlist: list[str] = []
        seen: set[str] = set()

        for symbol in self.config.watchlist:
            if symbol not in seen:
                merged_watchlist.append(symbol)
                seen.add(symbol)

        # Source 2: broker positions
        if self.broker:
            try:
                account_info = self.broker.get_account_info()
                position_symbols = set(account_info.positions.keys())

                if position_symbols:
                    added = position_symbols - seen
                    if added:
                        logger.info(f"Merged {len(added)} positions into watchlist: {sorted(added)}")
                        merged_watchlist.extend(sorted(added))
                        seen.update(added)
                else:
                    logger.debug("No positions to merge")
            except Exception as e:
                logger.warning(f"Failed to fetch positions for watchlist merge: {e}")
        else:
            logger.debug("No broker configured, using config watchlist only")

        # Source 3: latest screening candidates (ordered by score)
        if self.config.screening.enabled and self.state.screening_history:
            latest = self.state.screening_history[-1]
            new_symbols = [s for s in latest.top_symbols if s not in seen]
            if new_symbols:
                logger.info(f"Merged {len(new_symbols)} screening candidates: {new_symbols}")
                merged_watchlist.extend(new_symbols)
                seen.update(new_symbols)

        return merged_watchlist

    async def _analyze_symbol(self, symbol: str) -> TradingWorkflowResult | None:
        """Analyze a single symbol.

        Args:
            symbol: Stock ticker symbol

        Returns:
            TradingWorkflowResult or None on error
        """
        from src.strategies.session import TradingSession

        try:
            workflow = self._init_workflow()

            # Determine current session (default to REGULAR if called outside market hours)
            session = self.scheduler.get_trading_session() or TradingSession.REGULAR

            result = await workflow.analyze(symbol, period_days=90, trading_session=session)

            self.state.record_analysis(
                symbol=symbol,
                signal=result.decision.action.value,
                confidence=result.decision.confidence,
                executed=result.order is not None,
                trading_session=result.trading_session.value,
            )

            return result
        except Exception as e:
            error_msg = f"Failed to analyze {symbol}: {e}"
            logger.error(error_msg)
            self.state.record_error(error_msg)
            return None

    async def _analyze_watchlist(self, watchlist: list[str]) -> list[TradingWorkflowResult]:
        """Analyze all symbols in watchlist.

        Args:
            watchlist: List of symbols to analyze

        Returns:
            List of analysis results
        """
        results: list[TradingWorkflowResult] = []
        semaphore = asyncio.Semaphore(self.config.max_concurrent_analyses)

        async def analyze_with_limit(symbol: str) -> TradingWorkflowResult | None:
            async with semaphore:
                return await self._analyze_symbol(symbol)

        tasks = [analyze_with_limit(s) for s in watchlist]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(raw_results):
            if isinstance(result, Exception):
                symbol = watchlist[i]
                logger.error(f"Analysis failed for {symbol}: {result}")
                self.state.record_error(f"{symbol}: {result}")
            elif result is not None:
                results.append(result)

        return results

    def _log_results(self, results: list[TradingWorkflowResult]) -> None:
        """Log analysis results to console.

        Args:
            results: List of analysis results
        """
        from src.strategies.session import TradingSession

        console.print(f"\n[bold cyan]Analysis Results ({datetime.now():%Y-%m-%d %H:%M})[/bold cyan]")  # noqa: DTZ005
        console.print("-" * 50)

        for result in results:
            signal = result.decision.action.value
            color = {"BUY": "green", "SELL": "red"}.get(signal, "yellow")

            # Add pre-market badge if applicable
            session_badge = ""
            if result.trading_session == TradingSession.PRE_MARKET:
                session_badge = " [dim](PRE-MARKET)[/dim]"

            console.print(
                f"[bold]{result.symbol}[/bold]: "
                f"[{color}]{signal}[/{color}] "
                f"(confidence: {result.decision.confidence:.2f}){session_badge}"
            )

        console.print("-" * 50)
        console.print(f"Total: {len(results)} symbols analyzed\n")

    async def _maybe_run_journal(self) -> None:
        """Run after-hours journal if conditions are met."""
        if not self.config.journal.enabled:
            return

        if not self.scheduler.is_journal_window(self.config.journal.run_offset_minutes):
            return

        today = datetime.now(self.scheduler.timezone).date()
        if self.state.last_journal_date == today.isoformat():
            return

        # Filter today's analysis records
        today_records = [r for r in self.state.analyses if r.timestamp.date() == today]
        if not today_records:
            logger.info("No analyses today, skipping journal")
            return

        logger.info(f"Generating trade journal for {today} ({len(today_records)} records)")
        console.print(f"\n[bold magenta]Generating trade journal for {today}...[/bold magenta]")

        try:
            from src.agents.journal import TradeJournalAgent

            workflow = self._init_workflow()
            market_fetcher = MarketDataFetcher(use_alpha_vantage=False)
            journal_agent = TradeJournalAgent(workflow.llm_client, market_fetcher)

            journal = await journal_agent.generate(today, today_records)
            file_path = journal_agent.persist(journal, self.config.journal.journal_dir)

            self.state.last_journal_date = today.isoformat()
            self.state.save(self.config.state.state_file)

            correct = sum(1 for o in journal.outcomes if o.signal_correct)
            total = len(journal.outcomes)
            console.print(f"[bold magenta]Journal saved:[/bold magenta] {file_path}")
            if total > 0:
                console.print(f"[bold magenta]Signal accuracy:[/bold magenta] {correct}/{total}")
        except Exception as e:
            logger.error(f"Journal generation failed: {e}")
            self.state.record_error(f"Journal failed: {e}")
            self.state.save(self.config.state.state_file)

    def _run_optimization(self) -> None:
        """Run after-hours strategy parameter optimization."""
        if not self._daemon_optimizer:
            return

        # Check if already optimized today
        now = datetime.now(self.scheduler.timezone)
        if self.state.last_optimization:
            last_date = self.state.last_optimization.astimezone(self.scheduler.timezone).date()
            if last_date == now.date():
                logger.debug("Optimization already completed today")
                return

        logger.info("Starting after-hours parameter optimization")
        console.print(f"\n[bold cyan]Parameter Optimization ({now:%H:%M})[/bold cyan]")
        console.print("-" * 50)

        try:
            import time as time_mod

            start_time = time_mod.time()
            watchlist = self._get_merged_watchlist()

            optimized, skipped, failed = self._daemon_optimizer.optimize_watchlist(
                watchlist=watchlist,
                strategies=self.config.optimization.strategies,
                refresh_days=self.config.optimization.refresh_days,
            )

            total_time = time_mod.time() - start_time

            self.state.record_optimization(
                symbols_optimized=optimized,
                symbols_skipped=skipped,
                total_time_seconds=total_time,
            )
            self.state.save(self.config.state.state_file)

            if failed:
                for symbol, strategies_str in failed:
                    logger.warning(f"Failed to optimize {symbol}: {strategies_str}")

            console.print(
                f"\n[dim]Optimization complete: {len(optimized)} symbols optimized, "
                f"{len(skipped)} skipped ({total_time:.0f}s)[/dim]\n"
            )
            logger.info(f"Parameter optimization completed in {total_time:.0f}s")

        except Exception as e:
            error_msg = f"Parameter optimization failed: {e}"
            logger.error(error_msg)
            self.state.record_error(error_msg)

    def _run_after_hours_screening(self) -> None:
        """Run after-hours screening for watchlist candidates."""
        from src.data.universe import StockUniverseFetcher
        from src.screening.exporter import ScreeningExporter
        from src.screening.screener import ScreeningCriteria, StockScreener

        # Check if already screened today
        now = datetime.now(self.scheduler.timezone)
        if self.state.last_after_hours_screening:
            last_date = self.state.last_after_hours_screening.astimezone(self.scheduler.timezone).date()
            if last_date == now.date():
                logger.debug("After-hours screening already completed today")
                return

        logger.info("Starting after-hours watchlist screening")
        console.print(f"\n[bold cyan]After-Hours Screening ({now:%H:%M})[/bold cyan]")
        console.print("-" * 50)

        try:
            # Initialize screener
            universe_fetcher = StockUniverseFetcher()
            screener = StockScreener(universe_fetcher)

            # Parse criteria
            criteria_map = {
                "momentum": ScreeningCriteria.MOMENTUM,
                "value": ScreeningCriteria.VALUE,
                "breakout": ScreeningCriteria.BREAKOUT,
            }
            criteria = criteria_map.get(self.config.screening.criteria.lower(), ScreeningCriteria.MOMENTUM)

            # Run screening
            console.print(
                f"[dim]{criteria.value.title()} Screening[/dim]\n"
                f"[dim]Universe: {self.config.screening.universe}[/dim]"
            )
            output = screener.screen(
                criteria=criteria,
                universe=self.config.screening.universe,
                top_n=self.config.screening.top_n,
            )

            # Log top 5 to console
            self._log_screening_results(output.results[:5])

            # Save to watchlist file
            exporter = ScreeningExporter()
            exporter.save_to_watchlist(
                results=output.results[: self.config.screening.top_n],
                criteria=criteria,
                watchlist_name=self.config.screening.watchlist_name,
            )

            # Record in state
            self.state.record_after_hours_screening(
                criteria=criteria.value,
                universe=self.config.screening.universe,
                candidates=output.results,
                top_n=self.config.screening.top_n,
                screened_at=output.screened_at,
            )
            self.state.save(self.config.state.state_file)

            console.print(
                f"\n[dim]Top {self.config.screening.top_n} candidates saved to daemon state "
                f"({len(output.results)} total screened)[/dim]\n"
            )
            logger.info(f"After-hours screening completed: {len(output.results)} candidates")

        except Exception as e:
            error_msg = f"After-hours screening failed: {e}"
            logger.error(error_msg)
            self.state.record_error(error_msg)

    def _log_screening_results(self, results: list) -> None:
        """Log screening results to console.

        Args:
            results: List of ScreeningResult objects (top 5)
        """
        for i, result in enumerate(results, 1):
            console.print(
                f"[bold]{i}. {result.symbol}[/bold] ({result.name}) - Score: {result.score:.2f}\n"
                f"   {result.reason}"
            )

    async def _run_cycle(self) -> int:
        """Run a single analysis cycle.

        Returns:
            Seconds to sleep before next cycle
        """
        # Check if it's time for screening (before regular analysis)
        if self.scheduler.is_screening_time():
            self._run_after_hours_screening()

        # Check if it's time for parameter optimization
        if self.config.optimization.enabled and self.scheduler.is_optimization_time():
            self._run_optimization()

        if self.config.market_hours_only and not self.scheduler.is_market_open():
            await self._maybe_run_journal()
            wait_time = self.scheduler.time_until_open()
            if wait_time > 0:
                logger.info(f"Market closed, waiting {wait_time // 60} minutes until open")
                return min(wait_time, 60)

        watchlist = self._get_merged_watchlist()
        logger.info(f"Starting analysis cycle for {len(watchlist)} symbols")
        console.print(f"\n[bold]Running analysis cycle...[/bold] ({datetime.now():%H:%M:%S})")  # noqa: DTZ005

        results = await self._analyze_watchlist(watchlist)
        self._log_results(results)

        # Check journal regardless of market_hours_only setting
        await self._maybe_run_journal()

        self.state.save(self.config.state.state_file)
        return self.config.interval_minutes * 60

    async def run(self) -> None:
        """Run the daemon main loop."""
        self.running = True

        def shutdown_handler(sig: int, _frame: object) -> None:
            logger.info(f"Received signal {sig}, shutting down...")
            self.running = False

        signal.signal(signal.SIGINT, shutdown_handler)
        signal.signal(signal.SIGTERM, shutdown_handler)

        console.print("\n[bold green]Daemon started[/bold green]")
        console.print(f"Watchlist: {', '.join(self.config.watchlist)}")
        console.print(f"Interval: {self.config.interval_minutes} minutes")
        console.print(f"Market hours only: {self.config.market_hours_only}")
        console.print(f"Auto trade: {self.config.auto_trade}")
        console.print()

        while self.running:
            try:
                sleep_seconds = await self._run_cycle()
                logger.info(f"Sleeping for {sleep_seconds // 60} minutes")
                await asyncio.sleep(sleep_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in daemon loop: {e}")
                self.state.record_error(str(e))
                await asyncio.sleep(60)

        self.state.save(self.config.state.state_file)
        console.print("\n[bold yellow]Daemon stopped[/bold yellow]")
        logger.info("Daemon shutdown complete")

    @classmethod
    def from_config_file(cls, path: Path) -> "DaemonRunner":
        """Create runner from config file.

        Args:
            path: Path to TOML config file

        Returns:
            DaemonRunner instance
        """
        config = DaemonConfig.from_toml(path)
        return cls(config)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"DaemonRunner(config={self.config})"
