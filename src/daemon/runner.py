"""Main daemon runner for autonomous trading."""

import asyncio
import os
import signal
from datetime import datetime
from pathlib import Path

from loguru import logger
from rich.console import Console

from src.cache.historical import HistoricalCache
from src.daemon.config import DaemonConfig
from src.daemon.prefetch import DataPrefetcher
from src.daemon.scheduler import MarketScheduler
from src.daemon.state import DaemonState, SectorRotationRecord
from src.data.broker import AlpacaBroker
from src.data.fundamental import FundamentalDataFetcher
from src.data.market import MarketDataFetcher
from src.data.news import NewsFetcher
from src.metrics.sector_rotation import SectorRotationAnalysis
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
        self._historical_cache = HistoricalCache()
        self.scheduler = MarketScheduler(
            start_time=config.schedule.start_time,
            end_time=config.schedule.end_time,
            timezone=config.schedule.timezone,
            enable_pre_market=config.schedule.enable_pre_market,
            enable_after_hours=config.screening.enabled,
            after_hours_screen_time=config.screening.screen_time,
            after_hours_screen_days=config.screening.screen_days,
            optimization_time=config.optimization.optimization_time,
            optimization_days=config.optimization.optimization_days,
            health_check_time=config.health.run_time,
            prefetch_time=config.prefetch.prefetch_time,
            pre_market_refresh_time=config.prefetch.pre_market_refresh_time,
            sector_rotation_time=config.sector_rotation.run_time,
            sector_rotation_days=config.sector_rotation.run_days,
            enable_sector_rotation=config.sector_rotation.enabled,
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
            self.broker = AlpacaBroker(paper=True, historical_cache=self._historical_cache)
            logger.info("Alpaca broker initialized for auto-trading")
        elif os.getenv("ALPACA_API_KEY") and os.getenv("ALPACA_SECRET_KEY"):
            try:
                self.broker = AlpacaBroker(paper=True, historical_cache=self._historical_cache)
                logger.info("Alpaca broker initialized for watchlist merging")
            except Exception as e:
                logger.exception(f"Failed to initialize broker: {e}")
                self.broker = None
        self._prefetcher: DataPrefetcher | None = None
        logger.info(f"DaemonRunner initialized with {config}")

    def _init_prefetcher(self) -> DataPrefetcher | None:
        """Initialize data prefetcher (lazy initialization).

        Returns:
            DataPrefetcher instance or None if API key missing
        """
        if self._prefetcher is None:
            try:
                market_fetcher = MarketDataFetcher(
                    use_alpha_vantage=False, historical_cache=self._historical_cache
                )
                news_fetcher = NewsFetcher(historical_cache=self._historical_cache)
                fundamental_fetcher = FundamentalDataFetcher(historical_cache=self._historical_cache)

                self._prefetcher = DataPrefetcher(
                    market_fetcher=market_fetcher,
                    news_fetcher=news_fetcher,
                    fundamental_fetcher=fundamental_fetcher,
                    cache_dir=self.config.prefetch.cache_dir,
                )
                logger.info("DataPrefetcher initialized")
            except ValueError as e:
                logger.warning(f"Failed to initialize prefetcher: {e}")
                return None
        return self._prefetcher

    def _init_workflow(self) -> TradingWorkflow:
        """Initialize trading workflow (lazy initialization)."""
        if self._workflow is None:
            llm_client = LLMClient()
            market_fetcher = MarketDataFetcher(
                use_alpha_vantage=False, historical_cache=self._historical_cache
            )
            news_fetcher = NewsFetcher(historical_cache=self._historical_cache)
            finbert = get_finbert_sentiment()
            fundamental_fetcher = FundamentalDataFetcher(historical_cache=self._historical_cache)

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
                historical_cache=self._historical_cache,
            )
            logger.info("Trading workflow initialized")
        return self._workflow

    def _get_merged_watchlist(self) -> list[str]:
        """Get watchlist merged with broker positions and screening candidates.

        Returns:
            Deduplicated list combining config watchlist, broker positions,
            and latest screening candidates. Config order is preserved,
            broker positions are appended in alphabetical order, and screening
            candidates are appended in the order of ``latest.top_symbols``
            (typically ordered by screening score/rank).
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
            logger.debug("No broker configured, skipping position merge")

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

            # Build sector rotation context if available
            sector_context: str | None = None
            if self.config.sector_rotation.enabled and self.state.sector_rotation_history:
                try:
                    # Reuse latest sector rotation from daily run (stored in state)
                    latest_record = self.state.sector_rotation_history[-1]
                    sector_context = self._format_sector_context(latest_record)
                except Exception as e:
                    logger.warning(f"Failed to build sector context: {e}")

            result = await workflow.analyze(
                symbol, period_days=90, trading_session=session, sector_context=sector_context
            )

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

    def _format_sector_context(self, record: SectorRotationRecord) -> str:
        """Format sector rotation record as text for trader prompt.

        Args:
            record: Sector rotation state record

        Returns:
            Formatted context string
        """
        lines = [
            f"Leading Sectors: {', '.join(record.leading_sectors)}",
            f"Lagging Sectors: {', '.join(record.lagging_sectors)}",
            "",
        ]

        # Sort by strength descending
        sorted_sectors = sorted(record.sector_strengths.items(), key=lambda x: x[1], reverse=True)

        for rank, (sector, strength) in enumerate(sorted_sectors, 1):
            momentum = record.sector_momenta.get(sector, "NEUTRAL")
            lines.append(f"  {rank}. {sector}: strength={strength:+.2f} [{momentum}]")

        return "\n".join(lines)

    def _reconstruct_rotation_analysis(self, record: SectorRotationRecord) -> SectorRotationAnalysis:
        """Reconstruct SectorRotationAnalysis from state record.

        Args:
            record: Sector rotation state record

        Returns:
            Full SectorRotationAnalysis pydantic model
        """
        from src.data.comparative import Sector
        from src.metrics.sector_rotation import Momentum, SectorStrength

        # Reconstruct SectorStrength list from record data
        sectors = []
        sorted_sectors = sorted(record.sector_strengths.items(), key=lambda x: x[1], reverse=True)

        for rank, (sector_name, strength) in enumerate(sorted_sectors, 1):
            momentum_str = record.sector_momenta.get(sector_name, "NEUTRAL")

            # Find ETF for sector (map back from Sector enum)
            try:
                sector_enum = Sector[sector_name]
                etf = sector_enum.value
            except KeyError:
                logger.warning(f"Unknown sector {sector_name}, skipping")
                continue

            sectors.append(
                SectorStrength(
                    sector=sector_name,
                    etf=etf,
                    return_1w=0.0,  # Not stored in record
                    return_1m=0.0,
                    return_3m=0.0,
                    relative_strength=strength,
                    momentum=Momentum(momentum_str),
                    rank=rank,
                )
            )

        return SectorRotationAnalysis(
            sectors=sectors,
            leading_sectors=record.leading_sectors,
            lagging_sectors=record.lagging_sectors,
            spy_return_1w=0.0,  # Not stored, not needed for weighting
            spy_return_1m=0.0,
            spy_return_3m=0.0,
            timestamp=record.timestamp,
        )

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

    def _run_prefetch(self) -> None:
        """Run after-hours data prefetching for watchlist symbols."""
        if not self.config.prefetch.enabled:
            return

        # Dedup check
        now = datetime.now(self.scheduler.timezone)
        if self.state.last_prefetch:
            last_date = self.state.last_prefetch.astimezone(self.scheduler.timezone).date()
            if last_date == now.date():
                logger.debug("Prefetch already completed today")
                return

        logger.info("Starting after-hours data prefetching")
        console.print(f"\n[bold cyan]Data Prefetch ({now:%H:%M})[/bold cyan]")
        console.print("-" * 50)

        try:
            prefetcher = self._init_prefetcher()
            if prefetcher is None:
                logger.warning("Prefetcher unavailable (missing ALPHA_VANTAGE_API_KEY), skipping")
                return

            watchlist = self._get_merged_watchlist()

            console.print(f"[dim]Prefetching {len(watchlist)} symbols...[/dim]")
            report = prefetcher.prefetch_watchlist(watchlist)

            # Warm FinBERT if configured
            finbert_ready = False
            if self.config.prefetch.warm_finbert:
                console.print("[dim]Warming FinBERT model...[/dim]")
                finbert_ready = prefetcher.warm_finbert()
            report.finbert_ready = finbert_ready

            # Check API connectivity if configured
            if self.config.prefetch.check_connectivity:
                report.api_connectivity = prefetcher.check_api_key_presence()

            # Count successes/failures
            succeeded = sum(1 for r in report.results if r.market_data or r.news or r.fundamentals)
            failed = len(report.results) - succeeded

            self.state.record_prefetch(
                symbols_prefetched=succeeded,
                symbols_failed=failed,
                finbert_ready=finbert_ready,
                total_duration_seconds=report.total_duration_seconds,
            )
            self.state.save(self.config.state.state_file)

            console.print(
                f"\n[dim]Prefetch complete: {succeeded} symbols cached, "
                f"{failed} failed ({report.total_duration_seconds:.0f}s)[/dim]\n"
            )
            logger.info(
                f"Data prefetch completed: {succeeded} cached, {failed} failed "
                f"in {report.total_duration_seconds:.0f}s"
            )

        except Exception as e:
            error_msg = f"Data prefetch failed: {e}"
            logger.error(error_msg)
            self.state.record_error(error_msg)

    def _run_pre_market_refresh(self) -> None:
        """Run pre-market data refresh to update stale cache."""
        if not self.config.prefetch.enabled or not self.config.prefetch.enable_pre_market_refresh:
            return

        # Dedup check
        now = datetime.now(self.scheduler.timezone)
        if self.state.last_pre_market_refresh:
            last_date = self.state.last_pre_market_refresh.astimezone(self.scheduler.timezone).date()
            if last_date == now.date():
                logger.debug("Pre-market refresh already completed today")
                return

        logger.info("Starting pre-market data refresh")
        console.print(f"\n[bold cyan]Pre-Market Data Refresh ({now:%H:%M})[/bold cyan]")
        console.print("-" * 50)

        try:
            prefetcher = self._init_prefetcher()
            if prefetcher is None:
                logger.warning("Prefetcher unavailable (missing ALPHA_VANTAGE_API_KEY), skipping")
                return

            watchlist = self._get_merged_watchlist()

            console.print(f"[dim]Refreshing {len(watchlist)} symbols...[/dim]")
            report = prefetcher.prefetch_watchlist(watchlist)

            succeeded = sum(1 for r in report.results if r.market_data or r.news or r.fundamentals)

            self.state.last_pre_market_refresh = datetime.now(self.scheduler.timezone)
            self.state.save(self.config.state.state_file)

            console.print(
                f"\n[dim]Pre-market refresh complete: {succeeded} symbols updated "
                f"({report.total_duration_seconds:.0f}s)[/dim]\n"
            )
            logger.info(
                f"Pre-market refresh completed: {succeeded} symbols in {report.total_duration_seconds:.0f}s"
            )

        except Exception as e:
            error_msg = f"Pre-market refresh failed: {e}"
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

            # Apply sector rotation weighting if available
            results_to_save = output.results
            if self.config.sector_rotation.enabled and self.state.sector_rotation_history:
                try:
                    from src.daemon.sector_rotation import DaemonSectorRotation

                    # Reconstruct analysis from latest state record
                    latest_record = self.state.sector_rotation_history[-1]
                    rotation_analysis = self._reconstruct_rotation_analysis(latest_record)

                    daemon_rotation = DaemonSectorRotation()
                    results_to_save = daemon_rotation.weight_candidates(
                        output.results,
                        rotation_analysis,
                        self.config.sector_rotation.boost_factor,
                    )
                    logger.info("Applied sector rotation weighting to screening candidates")
                except Exception as e:
                    logger.warning(f"Failed to apply sector weighting: {e}")

            # Log top 5 to console
            self._log_screening_results(results_to_save[:5])

            # Save to watchlist file
            exporter = ScreeningExporter()
            exporter.save_to_watchlist(
                results=results_to_save[: self.config.screening.top_n],
                criteria=criteria,
                watchlist_name=self.config.screening.watchlist_name,
            )

            # Record in state
            self.state.record_after_hours_screening(
                criteria=criteria.value,
                universe=self.config.screening.universe,
                candidates=results_to_save,
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

    def _run_sector_rotation(self) -> None:
        """Run sector rotation analysis."""
        from src.daemon.sector_rotation import DaemonSectorRotation

        # Check if already ran today
        now = datetime.now(self.scheduler.timezone)
        if self.state.last_sector_rotation:
            last_date = self.state.last_sector_rotation.astimezone(self.scheduler.timezone).date()
            if last_date == now.date():
                logger.debug("Sector rotation already completed today")
                return

        logger.info("Starting sector rotation analysis")
        console.print(f"\n[bold cyan]Sector Rotation Analysis ({now:%H:%M})[/bold cyan]")
        console.print("-" * 50)

        try:
            daemon_rotation = DaemonSectorRotation()
            analysis = daemon_rotation.run()

            # Flag positions in weak sectors
            flagged: list[str] = []
            if self.broker:
                try:
                    account_info = self.broker.get_account_info()
                    position_symbols = list(account_info.positions.keys())
                    flagged = daemon_rotation.flag_weak_positions(position_symbols, analysis)
                except Exception as e:
                    logger.warning(f"Failed to flag positions: {e}")

            # Record in state
            sector_strengths = {s.sector: s.relative_strength for s in analysis.sectors}
            sector_momenta = {s.sector: s.momentum.value for s in analysis.sectors}

            self.state.record_sector_rotation(
                leading_sectors=analysis.leading_sectors,
                lagging_sectors=analysis.lagging_sectors,
                sector_strengths=sector_strengths,
                sector_momenta=sector_momenta,
                flagged_positions=flagged,
            )
            self.state.save(self.config.state.state_file)

            # Console output
            console.print(f"[dim]Leading: {', '.join(analysis.leading_sectors)}[/dim]")
            console.print(f"[dim]Lagging: {', '.join(analysis.lagging_sectors)}[/dim]")
            if flagged:
                console.print(f"[bold yellow]Flagged positions: {', '.join(flagged)}[/bold yellow]")
            console.print(
                f"\n[dim]Sector rotation complete: {len(analysis.sectors)} sectors analyzed[/dim]\n"
            )
            logger.info("Sector rotation analysis completed")

        except Exception as e:
            error_msg = f"Sector rotation failed: {e}"
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

    async def _maybe_run_health_check(self) -> None:
        """Run health check if conditions are met."""
        if not self.config.health.enabled:
            return

        if not self.scheduler.is_health_check_time(self.config.health.run_time):
            return

        today = datetime.now(self.scheduler.timezone).date()
        if self.state.last_health_check and self.state.last_health_check.date() == today:
            return

        logger.info("Starting API health checks")
        console.print(f"\n[bold cyan]Running Health Checks ({datetime.now():%H:%M})[/bold cyan]")  # noqa: DTZ005

        try:
            from src.daemon.health import HealthChecker

            checker = HealthChecker(self.config, self.state)
            report = await checker.run()

            self.state.last_health_check = datetime.now(tz=self.scheduler.timezone)
            self.state.save(self.config.state.state_file)

            console.print(
                f"[bold cyan]Health:[/bold cyan] {report.overall_status} "
                f"({len(report.service_checks)} services, {report.total_duration_ms:.0f}ms)"
            )
            logger.info(f"Health check complete: {report.overall_status}")
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            self.state.record_error(f"Health check failed: {e}")
            self.state.save(self.config.state.state_file)

    async def _run_cycle(self) -> int:
        """Run a single analysis cycle.

        Returns:
            Seconds to sleep before next cycle
        """
        # Check if it's time for data prefetching (before screening)
        if self.config.prefetch.enabled and self.scheduler.is_prefetch_time():
            self._run_prefetch()

        # Check if it's time for pre-market data refresh
        if self.config.prefetch.enabled and self.scheduler.is_pre_market_refresh_time():
            self._run_pre_market_refresh()

        # Check if it's time for sector rotation (before screening)
        if self.scheduler.is_sector_rotation_time():
            self._run_sector_rotation()

        # Check if it's time for screening (before regular analysis)
        if self.scheduler.is_after_hours_screening_time():
            self._run_after_hours_screening()

        await self._maybe_run_health_check()

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
