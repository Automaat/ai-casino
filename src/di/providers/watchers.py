"""DI providers for event watchers."""

from typing import TYPE_CHECKING

from loguru import logger

from src.cache.historical import HistoricalCache
from src.daemon.config import DaemonConfig
from src.daemon.watchers.news_trending_watcher import NewsTrendingWatcher
from src.daemon.watchers.news_watcher import NewsWatcher
from src.daemon.watchers.social_watcher import SocialWatcher
from src.daemon.watchers.trump_watcher import TrumpWatcher
from src.data.base_news_fetcher import BaseNewsFetcher

if TYPE_CHECKING:
    from src.daemon.state.facade import DaemonState
    from src.di.container import AppContainer


def create_news_watcher(
    historical_cache: HistoricalCache,
    daemon_config: DaemonConfig,
    container: AppContainer | None = None,
) -> NewsWatcher | None:
    """Create news watcher with enabled sources.

    Args:
        historical_cache: Historical cache for data persistence
        daemon_config: Daemon configuration
        container: Optional DI container (auto-created if not provided)

    Returns:
        NewsWatcher instance if enabled, None otherwise
    """
    config = daemon_config.news_watcher
    if not config.enabled:
        logger.debug("NewsWatcher disabled in config")
        return None

    # Build fetcher list based on enabled sources
    fetchers: list[BaseNewsFetcher] = []

    if config.sources.enable_marketaux:
        from src.di.container import create_container
        from src.di.providers.data import create_news_fetcher

        # Get circuit breaker registry from container (reuse existing or create once)
        if container is None:
            container = create_container()
        circuit_breaker_registry = container.circuit_breaker_registry()

        fetchers.append(create_news_fetcher(daemon_config, historical_cache, circuit_breaker_registry))

    if config.sources.enable_finnhub:
        from src.di.providers.data import create_finnhub_news_fetcher

        fetchers.append(create_finnhub_news_fetcher(daemon_config, historical_cache))

    if config.sources.enable_newsdata:
        from src.di.providers.data import create_newsdata_fetcher

        fetchers.append(create_newsdata_fetcher(daemon_config, historical_cache))

    if config.sources.enable_duckduckgo:
        from src.di.providers.data import create_duckduckgo_news_fetcher

        fetchers.append(create_duckduckgo_news_fetcher(historical_cache))

    if not fetchers:
        logger.warning("NewsWatcher enabled but no sources configured")
        return None

    logger.info(f"NewsWatcher configured with sources: {[f.get_source_name() for f in fetchers]}")

    return NewsWatcher(
        historical_cache=historical_cache,
        fetchers=fetchers,
        container=container,
        poll_interval=config.poll_interval_minutes * 60,
        relevance_threshold=config.relevance_threshold,
        cooldown_minutes=config.cooldown_minutes,
        breaking_threshold_minutes=config.breaking_threshold_minutes,
        max_concurrent_analyses=config.max_concurrent_analyses,
    )


def create_social_watcher(
    historical_cache: HistoricalCache,
    daemon_config: DaemonConfig,
    container: AppContainer | None = None,
    state: DaemonState | None = None,
) -> SocialWatcher | None:
    """Create social media watcher if enabled.

    Args:
        historical_cache: Historical cache for data persistence
        daemon_config: Daemon configuration
        container: Optional DI container (auto-created if not provided)
        state: Optional daemon state for WATCHLIST event persistence

    Returns:
        SocialWatcher instance if enabled, None otherwise
    """
    config = daemon_config.social_watcher
    if not config.enabled:
        logger.debug("SocialWatcher disabled in config")
        return None

    return SocialWatcher(
        historical_cache=historical_cache,
        container=container,
        state=state,
        poll_interval=config.poll_interval_minutes * 60,
        relevance_threshold=config.relevance_threshold,
        cooldown_minutes=config.cooldown_minutes,
        volume_spike_threshold=config.volume_spike_threshold,
        viral_score_threshold=config.viral_score_threshold,
        viral_upvote_ratio=config.viral_upvote_ratio,
        subreddits=config.subreddits,
        max_concurrent_analyses=config.max_concurrent_analyses,
    )


def create_trump_watcher(
    historical_cache: HistoricalCache,
    daemon_config: DaemonConfig,
    container: AppContainer | None = None,
    state: DaemonState | None = None,
) -> TrumpWatcher | None:
    """Create Trump Truth Social watcher if enabled.

    Args:
        historical_cache: Historical cache for data persistence
        daemon_config: Daemon configuration
        container: Optional DI container (auto-created if not provided)
        state: Optional daemon state for WATCHLIST event persistence

    Returns:
        TrumpWatcher instance if enabled, None otherwise
    """
    config = daemon_config.trump_watcher
    if not config.enabled:
        logger.debug("TrumpWatcher disabled in config")
        return None

    return TrumpWatcher(
        historical_cache=historical_cache,
        container=container,
        state=state,
        poll_interval=config.poll_interval_minutes * 60,
        relevance_threshold=config.relevance_threshold,
        cooldown_minutes=config.cooldown_minutes,
        max_concurrent_analyses=config.max_concurrent_analyses,
    )


def create_news_trending_watcher(
    historical_cache: HistoricalCache,
    daemon_config: DaemonConfig,
    container: AppContainer | None = None,
) -> NewsTrendingWatcher | None:
    """Create news trending watcher for continuous discovery.

    Args:
        historical_cache: Historical cache for data persistence
        daemon_config: Daemon configuration
        container: Optional DI container (auto-created if not provided)

    Returns:
        NewsTrendingWatcher instance if enabled, None otherwise
    """
    from src.daemon.watchers.news_trending_watcher import NewsTrendingWatcherConfig

    config = daemon_config.news_trending_watcher
    if not config.enabled:
        logger.debug("NewsTrendingWatcher disabled in config")
        return None

    if container is None:
        from src.di.container import create_container

        container = create_container()

    websearch_fetcher = container.websearch_fetcher()

    watcher_config = NewsTrendingWatcherConfig(
        poll_interval=config.poll_interval_minutes * 60,
        trending_window_minutes=config.trending_window_minutes,
        min_mention_threshold=config.min_mention_threshold,
        relevance_threshold=config.relevance_threshold,
        max_candidates_per_cycle=config.max_candidates_per_cycle,
        search_queries=config.search_queries,
        max_results_per_query=config.max_results_per_query,
    )

    watcher = NewsTrendingWatcher(
        websearch_fetcher=websearch_fetcher,
        historical_cache=historical_cache,
        config=watcher_config,
        container=container,
    )

    logger.info("News trending watcher created (discovery mode)")
    return watcher
