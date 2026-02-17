"""Data operation tasks for prefetching, screening, and earnings calendar."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from loguru import logger
from rich.console import Console

from src.daemon.tasks.base import TaskExecutor

if TYPE_CHECKING:
    from src.daemon.factory import DaemonComponents
    from src.di.container import AppContainer

console = Console()


class PrefetchTask(TaskExecutor):
    """After-hours data prefetch task."""

    def __init__(self, components: DaemonComponents, container: AppContainer) -> None:
        """Initialize prefetch task."""
        super().__init__(components, container)
        self._succeeded = 0
        self._failed = 0
        self._finbert_ready = False
        self._duration = 0.0
        self._skipped = False

    @property
    def task_name(self) -> str:
        """Task display name."""
        return "Data Prefetch"

    async def execute(self) -> None:
        """Execute prefetch logic."""
        prefetcher = self.components.prefetcher
        if prefetcher is None:
            logger.warning("Prefetcher unavailable (missing ALPHA_VANTAGE_API_KEY), skipping")
            self._skipped = True
            return

        watchlist = await self.components.broker_manager.get_merged_watchlist()

        console.print(f"[dim]Prefetching {len(watchlist)} symbols...[/dim]")
        report = await prefetcher.prefetch_watchlist(watchlist)

        # Warm FinBERT if configured
        finbert_ready = False
        if self.components.config.prefetch.warm_finbert:
            console.print("[dim]Warming FinBERT model...[/dim]")
            finbert_ready = await asyncio.to_thread(prefetcher.warm_finbert)
        report.finbert_ready = finbert_ready

        # Check API connectivity if configured
        if self.components.config.prefetch.check_connectivity:
            report.api_connectivity = prefetcher.check_api_key_presence()

        # Count successes/failures
        succeeded = sum(1 for r in report.results if r.market_data or r.news or r.fundamentals)
        failed = len(report.results) - succeeded

        # Store for record_success
        self._succeeded = succeeded
        self._failed = failed
        self._finbert_ready = finbert_ready
        self._duration = report.total_duration_seconds

        console.print(
            f"\n[dim]Prefetch complete: {succeeded} symbols cached, "
            f"{failed} failed ({report.total_duration_seconds:.0f}s)[/dim]"
        )
        logger.info(
            f"Data prefetch completed: {succeeded} cached, {failed} failed "
            f"in {report.total_duration_seconds:.0f}s"
        )

    async def get_last_run(self) -> datetime | None:
        """Get last prefetch timestamp."""
        return await self.components.state.get_last_prefetch()

    async def record_success(self, duration: float) -> None:
        """Record prefetch completion."""
        if self._skipped:
            return
        await self.components.state.record_prefetch(
            symbols_prefetched=self._succeeded,
            symbols_failed=self._failed,
            finbert_ready=self._finbert_ready,
            total_duration_seconds=self._duration,
        )


class PreMarketRefreshTask(TaskExecutor):
    """Pre-market data refresh task to update stale cache."""

    def __init__(self, components: DaemonComponents, container: AppContainer) -> None:
        """Initialize pre-market refresh task."""
        super().__init__(components, container)
        self._skipped = False

    @property
    def task_name(self) -> str:
        """Task display name."""
        return "Pre-Market Data Refresh"

    async def execute(self) -> None:
        """Execute pre-market refresh logic."""
        prefetcher = self.components.prefetcher
        if prefetcher is None:
            logger.warning("Prefetcher unavailable (missing ALPHA_VANTAGE_API_KEY), skipping")
            self._skipped = True
            return

        watchlist = await self.components.broker_manager.get_merged_watchlist()

        console.print(f"[dim]Refreshing {len(watchlist)} symbols...[/dim]")
        report = await prefetcher.prefetch_watchlist(watchlist)

        succeeded = sum(1 for r in report.results if r.market_data or r.news or r.fundamentals)

        console.print(
            f"\n[dim]Pre-market refresh complete: {succeeded} symbols updated "
            f"({report.total_duration_seconds:.0f}s)[/dim]"
        )
        logger.info(
            f"Pre-market refresh completed: {succeeded} symbols in {report.total_duration_seconds:.0f}s"
        )

    async def get_last_run(self) -> datetime | None:
        """Get last pre-market refresh timestamp."""
        return await self.components.state.get_last_pre_market_refresh()

    async def record_success(self, duration: float) -> None:
        """Record pre-market refresh completion."""
        if self._skipped:
            return
        await self.components.state.set_last_pre_market_refresh(
            datetime.now(self.components.scheduler.timezone)
        )


class EarningsFetchTask(TaskExecutor):
    """Weekly earnings calendar fetch task."""

    @property
    def task_name(self) -> str:
        """Task display name."""
        return "Earnings Calendar Fetch"

    async def execute(self) -> None:
        """Execute earnings fetch logic."""
        from src.daemon.earnings import DaemonEarningsCalendar
        from src.daemon.state import EarningsEventRecord

        daemon_earnings = DaemonEarningsCalendar()
        watchlist = await self.components.broker_manager.get_merged_watchlist()

        console.print(f"[dim]Fetching earnings for {len(watchlist)} symbols...[/dim]")
        calendar = await asyncio.to_thread(daemon_earnings.fetch, watchlist)

        # Build event records
        event_records = [
            EarningsEventRecord(
                symbol=e.symbol,
                earnings_date=e.earnings_date.isoformat(),
                estimate_eps=e.estimate_eps,
            )
            for e in calendar.events
        ]

        symbols_with_earnings = len(calendar.events)
        symbols_without_earnings = max(0, len(watchlist) - symbols_with_earnings)
        if symbols_without_earnings:
            logger.info(
                "Earnings calendar: %d symbols with earnings data, %d symbols with no earnings data",
                symbols_with_earnings,
                symbols_without_earnings,
            )

        # NOTE: Missing earnings data is normal, not a failure
        await self.components.state.record_earnings_fetch(
            events=event_records,
            symbols_fetched=symbols_with_earnings,
            symbols_failed=0,  # Only track known fetch failures
        )

        # Show upcoming earnings
        now = datetime.now(self.components.scheduler.timezone)
        upcoming = daemon_earnings.get_upcoming(
            calendar.events, days_ahead=self.components.config.earnings_calendar.lookahead_days
        )
        if upcoming:
            console.print("[bold yellow]Upcoming earnings:[/bold yellow]")
            for event in upcoming:
                days_until = (event.earnings_date - now.date()).days
                console.print(f"  {event.symbol}: {event.earnings_date} ({days_until}d away)")
        else:
            console.print("[dim]No upcoming earnings within lookahead window[/dim]")

        console.print(
            f"\n[dim]Earnings fetch complete: {len(calendar.events)} symbols with earnings data[/dim]"
        )
        logger.info(f"Earnings calendar fetch completed: {len(calendar.events)} events")

    async def get_last_run(self) -> datetime | None:
        """Get last earnings fetch timestamp."""
        return await self.components.state.get_last_earnings_fetch()

    async def record_success(self, duration: float) -> None:
        """Record earnings fetch completion."""
        # State already recorded in execute()

    async def should_skip_today(self) -> bool:
        """Custom dedup: check weekly schedule.

        Returns:
            True if already fetched today or not fetch time
        """
        # Check if already fetched today
        last_run = await self.get_last_run()
        if last_run:
            now = datetime.now(self.components.scheduler.timezone)
            last_date = last_run.astimezone(self.components.scheduler.timezone).date()
            if last_date == now.date():
                return True

        # Check calendar-aware weekly schedule
        if not self.components.scheduler.is_earnings_fetch_time():
            logger.debug("Not earnings fetch time, skipping")
            return True

        return False


class DiscoveryOutcomeTask(TaskExecutor):
    """Discovery outcome tracking task (T+7d/30d price updates)."""

    def __init__(self, components: DaemonComponents, container: AppContainer) -> None:
        """Initialize discovery outcome task."""
        super().__init__(components, container)
        self._skipped = False

    @property
    def task_name(self) -> str:
        """Task display name."""
        return "Discovery Outcome Tracking"

    async def execute(self) -> None:
        """Execute discovery outcome tracking logic."""
        from src.daemon.discovery_tracker import DiscoveryOutcomeTracker
        from src.database.connection import get_session
        from src.database.repositories.discovery import DiscoveryHistoryRepository
        from src.database.repositories.discovery_source_metrics import (
            DiscoverySourceMetricsRepository,
        )

        if self.components.market_fetcher is None:
            logger.warning("Market fetcher not available, skipping discovery outcome tracking")
            self._skipped = True
            return

        async with get_session() as session:
            discovery_repo = DiscoveryHistoryRepository(session)
            metrics_repo = DiscoverySourceMetricsRepository(session)

            tracker = DiscoveryOutcomeTracker(
                market_fetcher=self.components.market_fetcher,
                discovery_repo=discovery_repo,
                metrics_repo=metrics_repo,
            )

            stats = await tracker.update_all_outcomes()

            today = datetime.now(UTC).date()
            metrics = await tracker.calculate_daily_source_metrics(today)

            console.print(
                f"[bold green]✓[/bold green] Discovery tracking: "
                f"{stats['updated_7d']} 7d outcomes, {stats['updated_30d']} 30d outcomes"
            )
            logger.info(
                f"Discovery outcome tracking: {stats['updated_7d']} 7d, "
                f"{stats['updated_30d']} 30d, {stats['failed']} failed, "
                f"{len(metrics)} source metrics calculated"
            )

    async def get_last_run(self) -> datetime | None:
        """Get last discovery outcome tracking timestamp."""
        return await self.components.state.get_last_discovery_outcome_tracking()

    async def record_success(self, duration: float) -> None:
        """Record discovery outcome tracking completion."""
        if self._skipped:
            return
        await self.components.state.set_last_discovery_outcome_tracking(datetime.now(UTC))


class PeriodicRedditScrapingTask(TaskExecutor):
    """Periodic Reddit scraping task using Playwright."""

    def __init__(self, components: DaemonComponents, container: AppContainer) -> None:
        """Initialize periodic Reddit scraping task."""
        super().__init__(components, container)
        self._posts_scraped = 0
        self._comments_scraped = 0
        self._mentions_extracted = 0
        self._skipped = False

    @property
    def task_name(self) -> str:
        """Task display name."""
        return "Reddit Scraping"

    async def execute(self) -> None:
        """Execute Reddit scraping logic."""
        from src.data.reddit_scraper import RedditPlaywrightScraper
        from src.data.reddit_ticker_extractor import RedditTickerExtractor
        from src.database.connection import get_session
        from src.database.models.reddit import ExtractionMethod
        from src.database.repositories.reddit import (
            RedditCommentRepository,
            RedditPostRepository,
            RedditTickerMentionRepository,
            RedditTickerSentimentRepository,
        )

        if not self.components.config.reddit_scraper.enabled:
            logger.debug("Reddit scraping disabled in config")
            self._skipped = True
            return

        cfg = self.components.config.reddit_scraper
        logger.info(
            f"Starting Reddit scraping: {len(cfg.high_priority_subreddits)} subreddits, "
            f"{cfg.posts_per_subreddit} posts each"
        )

        # Initialize scraper and extractor
        scraper = RedditPlaywrightScraper(config=cfg)
        extractor = RedditTickerExtractor(
            llm_client=self.components.container.llm_client(),
            config=cfg,
        )
        logger.debug(f"Initialized scraper and extractor (LLM={cfg.use_llm_extraction})")

        try:
            await scraper.start()

            all_posts = []
            all_comments = []
            all_mentions = []

            # Scrape each subreddit
            for subreddit in self.components.config.reddit_scraper.high_priority_subreddits:
                try:
                    logger.info(f"Scraping r/{subreddit}")
                    # Scrape posts
                    posts = await scraper.scrape_subreddit_posts(
                        subreddit=subreddit,
                        limit=self.components.config.reddit_scraper.posts_per_subreddit,
                    )
                    all_posts.extend(posts)
                    logger.info(f"Scraped {len(posts)} posts from r/{subreddit}")

                    # Scrape comments from top posts
                    top_posts = sorted(posts, key=lambda p: p.score, reverse=True)[:10]
                    logger.debug(f"Scraping comments from top {len(top_posts)} posts")
                    for post in top_posts:
                        comments = await scraper.scrape_post_comments(
                            post=post,
                            limit=self.components.config.reddit_scraper.comments_per_post,
                        )
                        all_comments.extend(comments)

                        # Extract tickers
                        if self.components.config.reddit_scraper.use_llm_extraction:
                            logger.debug(f"Extracting tickers from post {post.id}")
                            mentions = await extractor.extract_tickers(post=post, comments=comments)
                            if mentions:
                                logger.info(f"Extracted {len(mentions)} ticker mentions from post {post.id}")
                            all_mentions.append((post, mentions))

                except Exception:
                    logger.opt(exception=True).warning(f"Failed to scrape r/{subreddit}")
                    continue

            # Store to database
            async with get_session() as session:
                # Bulk insert posts
                post_repo = RedditPostRepository(session)
                posts_inserted = await post_repo.bulk_insert(all_posts)

                # Bulk insert comments
                comment_repo = RedditCommentRepository(session)
                comments_inserted = await comment_repo.bulk_insert(all_comments)

                # Insert ticker mentions
                mention_repo = RedditTickerMentionRepository(session)
                mentions_inserted = 0
                for post, mentions in all_mentions:
                    if mentions:
                        count = await mention_repo.bulk_insert_from_post(
                            post=post,
                            mentions=mentions,
                            extraction_method=ExtractionMethod.LLM,
                        )
                        mentions_inserted += count

                # Compute sentiment aggregates
                sentiment_repo = RedditTickerSentimentRepository(session)
                aggregates = await sentiment_repo.compute_hourly_aggregates(lookback_hours=1)

            # Store stats
            self._posts_scraped = posts_inserted
            self._comments_scraped = comments_inserted
            self._mentions_extracted = mentions_inserted

            console.print(
                f"[bold green]✓[/bold green] Reddit scraping: "
                f"{posts_inserted} posts, {comments_inserted} comments, "
                f"{mentions_inserted} ticker mentions, {aggregates} aggregates"
            )
            logger.info(
                f"Reddit scraping completed: {posts_inserted} posts, {comments_inserted} comments, "
                f"{mentions_inserted} mentions, {aggregates} aggregates"
            )

        finally:
            await scraper.close()

    async def get_last_run(self) -> datetime | None:
        """Get last Reddit scraping timestamp."""
        # TODO: Add state tracking in future iteration
        return None

    async def record_success(self, duration: float) -> None:
        """Record Reddit scraping completion."""
        if self._skipped:
            return
        # TODO: Add state tracking in future iteration
        logger.debug(
            f"Reddit scraping stats: {self._posts_scraped} posts, "
            f"{self._comments_scraped} comments, {self._mentions_extracted} mentions"
        )
