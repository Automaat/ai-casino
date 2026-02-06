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
        )
        self.state = DaemonState.load(config.state.state_file)
        self.running = False
        self._workflow: TradingWorkflow | None = None
        self.broker: AlpacaBroker | None = None
        if config.auto_trade or (os.getenv("ALPACA_API_KEY") and os.getenv("ALPACA_SECRET_KEY")):
            try:
                self.broker = AlpacaBroker(paper=True)
                logger.info("Alpaca broker initialized for daemon")
            except Exception as e:
                logger.warning(f"Failed to initialize broker: {e}")
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
            )
            logger.info("Trading workflow initialized")
        return self._workflow

    def _get_merged_watchlist(self) -> list[str]:
        """Get watchlist merged with broker positions.

        Returns:
            Deduplicated list combining config watchlist and broker positions
        """
        base_watchlist = set(self.config.watchlist)

        if not self.broker:
            logger.debug("No broker configured, using config watchlist only")
            return list(base_watchlist)

        try:
            account_info = self.broker.get_account_info()
            position_symbols = set(account_info.positions.keys())

            if position_symbols:
                merged = base_watchlist | position_symbols
                added = position_symbols - base_watchlist
                if added:
                    logger.info(f"Merged {len(added)} positions into watchlist: {sorted(added)}")
                return list(merged)

            logger.debug("No positions to merge")
            return list(base_watchlist)
        except Exception as e:
            logger.warning(f"Failed to fetch positions for watchlist merge: {e}")
            return list(base_watchlist)

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
                executed=False,
                trading_session=result.trading_session.value,
            )

            return result
        except Exception as e:
            error_msg = f"Failed to analyze {symbol}: {e}"
            logger.error(error_msg)
            self.state.record_error(error_msg)
            return None

    async def _analyze_watchlist(self) -> list[TradingWorkflowResult]:
        """Analyze all symbols in watchlist (merged with positions).

        Returns:
            List of analysis results
        """
        results: list[TradingWorkflowResult] = []
        watchlist = self._get_merged_watchlist()
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

    async def _run_cycle(self) -> int:
        """Run a single analysis cycle.

        Returns:
            Seconds to sleep before next cycle
        """
        if self.config.market_hours_only and not self.scheduler.is_market_open():
            wait_time = self.scheduler.time_until_open()
            if wait_time > 0:
                logger.info(f"Market closed, waiting {wait_time // 60} minutes until open")
                return min(wait_time, 60)

        watchlist = self._get_merged_watchlist()
        logger.info(f"Starting analysis cycle for {len(watchlist)} symbols")
        console.print(f"\n[bold]Running analysis cycle...[/bold] ({datetime.now():%H:%M:%S})")  # noqa: DTZ005

        results = await self._analyze_watchlist()
        self._log_results(results)

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
