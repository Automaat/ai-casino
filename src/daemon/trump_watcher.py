"""Trump social media watcher daemon."""

import asyncio
import re
import signal
from datetime import UTC, datetime
from typing import ClassVar

from loguru import logger
from pydantic import BaseModel
from rich.console import Console

from src.agents.trump import COMPANY_TICKERS, TrumpAnalysis, TrumpAnalyst
from src.data.truth_social import TruthPost, TruthSocialFetcher
from src.di.container import AppContainer
from src.models.llm import LLMClient
from src.workflows import TradingWorkflow
from src.workflows.types import TradingWorkflowResult

console = Console()


class TrumpSignal(BaseModel):
    """Signal emitted when Trump posts affect stocks."""

    post: TruthPost
    affected_symbols: list[str]
    trump_analysis: TrumpAnalysis
    analyses: dict[str, TradingWorkflowResult]
    timestamp: datetime


class TrumpWatcher:
    """Daemon that monitors Trump's posts and triggers stock analysis."""

    # Sector-based stock mapping for policy posts
    SECTOR_STOCKS: ClassVar[dict[str, list[str]]] = {
        "tariff": ["CAT", "DE", "BA", "X", "CLF"],
        "china": ["BABA", "JD", "NIO", "XPEV", "PDD"],
        "crypto": ["COIN", "MSTR", "RIOT", "MARA"],
        "oil": ["XOM", "CVX", "OXY", "COP"],
        "bank": ["JPM", "BAC", "WFC", "GS", "MS"],
        "tech": ["AAPL", "MSFT", "GOOGL", "META", "NVDA"],
        "defense": ["LMT", "RTX", "NOC", "BA", "GD"],
        "pharma": ["PFE", "JNJ", "MRK", "ABBV", "LLY"],
    }

    def __init__(
        self,
        poll_interval: int = 60,
        max_analyses: int = 5,
        container: AppContainer | None = None,
    ) -> None:
        """Initialize Trump watcher.

        Args:
            poll_interval: Seconds between poll cycles
            max_analyses: Maximum stocks to analyze per cycle
            container: Optional DI container (auto-created if not provided)
        """
        from src.di.container import create_container

        self.poll_interval = poll_interval
        self.max_analyses = max_analyses
        self.running = False
        self._last_post_id: str | None = None
        self._last_check: datetime | None = None
        self._container = container or create_container()

        # Lazy init
        self._historical_cache = self._container.historical_cache()
        self._fetcher: TruthSocialFetcher | None = None
        self._analyst: TrumpAnalyst | None = None
        self._workflow: TradingWorkflow | None = None
        self._llm: LLMClient | None = None  # Lazy init via container

        logger.info(f"TrumpWatcher initialized (poll_interval={poll_interval}s)")

    def _init_components(self) -> None:
        """Lazy initialization of components."""
        if self._fetcher is None:
            self._fetcher = TruthSocialFetcher(historical_cache=self._historical_cache)

        if self._llm is None:
            self._llm = self._container.llm_client()

        if self._analyst is None:
            self._analyst = TrumpAnalyst(self._llm)

        if self._workflow is None:
            self._workflow = self._container.workflow_trump(historical_cache=self._historical_cache)
            logger.info("TrumpWatcher workflow initialized")

    async def _check_new_posts(self) -> list[TruthPost]:
        """Check for new posts since last check."""
        self._init_components()
        if self._fetcher is None:
            msg = "Failed to initialize TruthSocialFetcher"
            raise RuntimeError(msg)

        if self._last_check is None:
            # First run: get posts from last hour
            data = self._fetcher.fetch_recent(hours=1)
        else:
            data = self._fetcher.fetch_since(self._last_check)

        self._last_check = datetime.now(UTC)

        if not data.posts:
            return []

        # Filter to only new posts
        new_posts = []
        for post in data.posts:
            if self._last_post_id is None or post.id != self._last_post_id:
                new_posts.append(post)
            else:
                break  # Posts are sorted newest first

        if new_posts:
            self._last_post_id = new_posts[0].id

        return new_posts

    async def _identify_affected_stocks(self, posts: list[TruthPost]) -> list[str]:
        """Identify stocks affected by posts.

        Uses:
        1. Direct ticker mentions ($TSLA)
        2. Company name mentions (Tesla, Apple)
        3. Sector inference via LLM
        """
        self._init_components()

        tickers = set()
        combined_text = " ".join(p.content for p in posts).lower()

        # Direct company mentions with word boundaries
        for company, ticker in COMPANY_TICKERS.items():
            pattern = r"\b" + company + r"\b"
            if re.search(pattern, combined_text):
                tickers.add(ticker)

        # Sector keywords
        for keyword, stocks in self.SECTOR_STOCKS.items():
            if keyword in combined_text:
                tickers.update(stocks[:2])  # Top 2 from each sector

        # If no direct matches, use LLM to identify affected stocks
        if not tickers and posts:
            tickers.update(await self._llm_identify_stocks(posts))

        return sorted(tickers)[: self.max_analyses]

    def _sanitize_post_content(self, content: str, max_length: int = 200) -> str:
        """Sanitize post content for LLM prompt.

        Args:
            content: Raw post content
            max_length: Maximum length after truncation

        Returns:
            Sanitized content
        """
        sanitized = content[:max_length]
        sanitized = sanitized.replace("\n", " ").replace("\r", " ")
        sanitized = sanitized.replace("{", "{{").replace("}", "}}")
        return sanitized.replace("```", "'''")

    async def _llm_identify_stocks(self, posts: list[TruthPost]) -> list[str]:
        """Use LLM to identify affected stocks."""
        if self._llm is None:
            msg = "LLM client not initialized"
            raise RuntimeError(msg)

        posts_text = "\n".join(f"- {self._sanitize_post_content(p.content)}" for p in posts[:5])

        prompt = f"""Based on these Truth Social posts from Donald Trump, identify up to 5 stock \
tickers that could be significantly affected:

{posts_text}

Return ONLY a comma-separated list of tickers (e.g., AAPL, TSLA, BA). \
If no specific stocks are affected, return "NONE".
"""

        response = await self._llm.acomplete(
            prompt,
            system="You are a financial analyst identifying stocks affected by political statements.",
            temperature=0.3,
        )

        if "NONE" in response.upper():
            return []

        # Parse tickers
        tickers = []
        for part in response.replace(",", " ").split():
            cleaned = part.strip().upper()
            if 1 <= len(cleaned) <= 5 and cleaned.isalpha():
                tickers.append(cleaned)

        return tickers[:5]

    async def _analyze_stocks(
        self, symbols: list[str], trump_analysis: TrumpAnalysis
    ) -> dict[str, TradingWorkflowResult]:
        """Run trading analysis for affected stocks.

        Args:
            symbols: Stock symbols to analyze
            trump_analysis: Trump analysis with market context
        """
        self._init_components()
        if self._workflow is None:
            msg = "Trading workflow not initialized"
            raise RuntimeError(msg)

        logger.debug(
            f"Analyzing stocks with trump context: signal={trump_analysis.signal}, "
            f"confidence={trump_analysis.confidence:.2f}"
        )

        results: dict[str, TradingWorkflowResult] = {}
        semaphore = asyncio.Semaphore(2)  # Limit concurrent analyses
        workflow = self._workflow

        async def analyze_one(symbol: str) -> tuple[str, TradingWorkflowResult | None]:
            async with semaphore:
                try:
                    result = await workflow.analyze(symbol, period_days=30)
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

    def _emit_signal(self, signal: TrumpSignal) -> None:
        """Emit trading signal to console."""
        console.print()
        console.print("[bold magenta]═══ TRUMP SIGNAL DETECTED ═══[/bold magenta]")
        console.print(f"[dim]{signal.timestamp:%Y-%m-%d %H:%M:%S}[/dim]")
        console.print()
        console.print(f"[bold]Post:[/bold] {signal.post.content[:200]}...")
        console.print(f"[dim]Likes: {signal.post.likes} | Reposts: {signal.post.reposts}[/dim]")
        console.print()

        if signal.trump_analysis.market_relevant:
            console.print("[bold yellow]Market Relevant:[/bold yellow] Yes")
            console.print(f"[bold]Signal:[/bold] {signal.trump_analysis.signal.value}")
            console.print(f"[bold]Confidence:[/bold] {signal.trump_analysis.confidence:.2f}")
            console.print(f"[bold]Sentiment:[/bold] {signal.trump_analysis.sentiment}")
            if signal.trump_analysis.mentioned_tickers:
                tickers = ", ".join(signal.trump_analysis.mentioned_tickers)
                console.print(f"[bold]Mentioned Tickers:[/bold] {tickers}")
        else:
            console.print("[dim]Not market relevant[/dim]")

        console.print()
        console.print(f"[bold cyan]Affected Stocks ({len(signal.affected_symbols)}):[/bold cyan]")

        for symbol, result in signal.analyses.items():
            action = result.decision.action.value
            color = {"BUY": "green", "SELL": "red"}.get(action, "yellow")
            console.print(
                f"  [bold]{symbol}[/bold]: [{color}]{action}[/{color}] "
                f"(confidence: {result.decision.confidence:.2f})"
            )

        console.print("[bold magenta]═══════════════════════════════[/bold magenta]")
        console.print()

    async def _run_cycle(self) -> None:
        """Run a single check cycle."""
        try:
            new_posts = await self._check_new_posts()

            if not new_posts:
                logger.debug("No new posts")
                return

            logger.info(f"Found {len(new_posts)} new post(s)")

            # Identify affected stocks (_identify_affected_stocks initializes components including _analyst)
            affected = await self._identify_affected_stocks(new_posts)

            # Analyze trump posts (analyst guaranteed initialized by _identify_affected_stocks)
            analyst = self._analyst
            if analyst is None:
                msg = "TrumpAnalyst not initialized after _identify_affected_stocks"
                raise RuntimeError(msg)
            trump_analysis = await analyst.analyze(new_posts)

            if not affected:
                console.print("[dim]New post detected but no affected stocks identified[/dim]")
                return

            logger.info(f"Analyzing {len(affected)} affected stocks: {affected}")

            # Run full analysis for each
            analyses = await self._analyze_stocks(affected, trump_analysis)

            # Emit signal for most recent post
            signal = TrumpSignal(
                post=new_posts[0],
                affected_symbols=affected,
                trump_analysis=trump_analysis,
                analyses=analyses,
                timestamp=datetime.now(UTC),
            )

            self._emit_signal(signal)

        except Exception as e:
            logger.error(f"Error in cycle: {e}")

    async def run(self) -> None:
        """Run the watcher daemon."""
        self.running = True

        def shutdown_handler(sig: int, _frame: object) -> None:
            logger.info(f"Received signal {sig}, shutting down...")
            self.running = False

        signal.signal(signal.SIGINT, shutdown_handler)
        signal.signal(signal.SIGTERM, shutdown_handler)

        console.print()
        console.print("[bold green]Trump Watcher Started[/bold green]")
        console.print(f"Poll interval: {self.poll_interval}s")
        console.print(f"Max analyses per signal: {self.max_analyses}")
        console.print("[dim]Waiting for new posts...[/dim]")
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
        console.print("[bold yellow]Trump Watcher Stopped[/bold yellow]")
        logger.info("TrumpWatcher shutdown complete")

    def __repr__(self) -> str:
        """String representation."""
        return f"TrumpWatcher(poll_interval={self.poll_interval}s)"
