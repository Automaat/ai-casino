"""DI providers for event watchers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from src.cache.historical import HistoricalCache
from src.daemon.config import DaemonConfig
from src.data.base_news_fetcher import BaseNewsFetcher
from src.v1.watchers.anomaly_watcher import AnomalyWatcher, AnomalyWatcherConfig
from src.v1.watchers.news_trending_watcher import NewsTrendingWatcher, NewsTrendingWatcherConfig
from src.v1.watchers.news_watcher import NewsWatcher, NewsWatcherConfig
from src.v1.watchers.pipeline import EventTriagePipeline
from src.v1.watchers.social_watcher import SocialWatcher, SocialWatcherConfig
from src.v1.watchers.trump_watcher import TrumpWatcher, TrumpWatcherConfig

if TYPE_CHECKING:
    from src.daemon.state.facade import DaemonState
    from src.database.engine import DatabaseEngine
    from src.di.container import AppContainer
    from src.v1.watchers.economic_calendar_watcher import EconomicCalendarWatcher
    from src.v1.watchers.options_flow_watcher import OptionsFlowWatcher
    from src.v1.watchers.social_sentiment_watcher import SocialSentimentWatcher


def _build_pipeline(
    daemon_config: DaemonConfig,
    container: AppContainer,
    state: DaemonState | None,
) -> EventTriagePipeline:
    """Build EventTriagePipeline from container + config.

    Args:
        daemon_config: Daemon configuration
        container: DI container
        state: Optional daemon state for WATCHLIST candidates

    Returns:
        EventTriagePipeline instance
    """
    triage_agent = container.event_triage_agent()
    ttl = daemon_config.event_integration.urgency_ttl_hours
    immediate_ttl = ttl.get("IMMEDIATE", 4)
    watchlist_ttl = ttl.get("WATCHLIST", 24)

    queue = None
    try:
        queue = container.market_event_queue()
    except Exception as e:
        logger.debug(f"market_event_queue unavailable, pipeline will log-and-drop IMMEDIATE events: {e}")

    return EventTriagePipeline(
        triage_agent=triage_agent,
        queue=queue,
        state=state,
        immediate_ttl_hours=immediate_ttl,
        watchlist_ttl_hours=watchlist_ttl,
    )


def create_news_watcher(
    historical_cache: HistoricalCache,
    daemon_config: DaemonConfig,
    container: AppContainer | None = None,
    state: DaemonState | None = None,
) -> NewsWatcher | None:
    """Create news watcher with enabled sources.

    Args:
        historical_cache: Historical cache for data persistence
        daemon_config: Daemon configuration
        container: DI container (auto-created if not provided)
        state: Daemon state for WATCHLIST event persistence

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

    if container is None:
        from src.di.container import create_container

        container = create_container()

    logger.info(f"NewsWatcher configured with sources: {[f.get_source_name() for f in fetchers]}")

    pipeline = _build_pipeline(daemon_config, container, state)
    watcher_config = NewsWatcherConfig(
        poll_interval=config.poll_interval_minutes * 60,
        breaking_threshold_minutes=config.breaking_threshold_minutes,
    )
    return NewsWatcher(
        pipeline=pipeline,
        historical_cache=historical_cache,
        fetchers=fetchers,
        config=watcher_config,
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
        container: DI container (auto-created if not provided)
        state: Daemon state for WATCHLIST event persistence

    Returns:
        SocialWatcher instance if enabled, None otherwise
    """
    config = daemon_config.social_watcher
    if not config.enabled:
        logger.debug("SocialWatcher disabled in config")
        return None

    if container is None:
        from src.di.container import create_container

        container = create_container()

    pipeline = _build_pipeline(daemon_config, container, state)
    watcher_config = SocialWatcherConfig(
        poll_interval=config.poll_interval_minutes * 60,
        volume_spike_threshold=config.volume_spike_threshold,
        viral_score_threshold=config.viral_score_threshold,
        viral_upvote_ratio=config.viral_upvote_ratio,
        subreddits=config.subreddits,
    )
    return SocialWatcher(
        pipeline=pipeline,
        historical_cache=historical_cache,
        config=watcher_config,
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
        container: DI container (auto-created if not provided)
        state: Daemon state for WATCHLIST event persistence

    Returns:
        TrumpWatcher instance if enabled, None otherwise
    """
    config = daemon_config.trump_watcher
    if not config.enabled:
        logger.debug("TrumpWatcher disabled in config")
        return None

    if container is None:
        from src.di.container import create_container

        container = create_container()

    pipeline = _build_pipeline(daemon_config, container, state)
    watcher_config = TrumpWatcherConfig(poll_interval=config.poll_interval_minutes * 60)
    return TrumpWatcher(
        pipeline=pipeline,
        historical_cache=historical_cache,
        config=watcher_config,
    )


def create_anomaly_watcher(
    daemon_config: DaemonConfig,
    container: AppContainer | None = None,
    state: DaemonState | None = None,
) -> AnomalyWatcher | None:
    """Create anomaly watcher if enabled.

    Args:
        daemon_config: Daemon configuration
        container: DI container (auto-created if not provided)
        state: Daemon state for WATCHLIST event persistence

    Returns:
        AnomalyWatcher instance if enabled, None otherwise
    """
    from src.data.market import MarketDataFetcher
    from src.di.config import resolve_config_or_env

    config = daemon_config.anomaly_watcher if hasattr(daemon_config, "anomaly_watcher") else None
    if config is not None and not config.enabled:
        logger.debug("AnomalyWatcher disabled in config")
        return None

    if container is None:
        from src.di.container import create_container

        container = create_container()

    alpha_vantage_key = resolve_config_or_env(
        daemon_config.api_keys.alpha_vantage_api_key, "ALPHA_VANTAGE_API_KEY"
    )
    market_fetcher = MarketDataFetcher(api_key=alpha_vantage_key)

    pipeline = _build_pipeline(daemon_config, container, state)
    watcher_config = AnomalyWatcherConfig(
        watchlist=list(daemon_config.watchlist),
    )
    watcher = AnomalyWatcher(
        pipeline=pipeline,
        market_fetcher=market_fetcher,
        config=watcher_config,
    )
    logger.info("AnomalyWatcher created")
    return watcher


def create_economic_calendar_watcher(
    daemon_config: DaemonConfig,
    database_engine: DatabaseEngine | None = None,
) -> EconomicCalendarWatcher | None:
    """Create economic calendar watcher if enabled.

    Args:
        daemon_config: Daemon configuration
        database_engine: Optional database engine for signal persistence

    Returns:
        EconomicCalendarWatcher instance if enabled, None otherwise
    """
    from src.data.economic_calendar import EconomicCalendarFetcher
    from src.di.config import resolve_config_or_env
    from src.v1.watchers.economic_calendar_watcher import (
        EconomicCalendarWatcher,
        EconomicCalendarWatcherConfig,
    )

    config = daemon_config.economic_calendar_watcher
    if not config.enabled:
        logger.debug("EconomicCalendarWatcher disabled in config")
        return None

    api_key = resolve_config_or_env(daemon_config.api_keys.fred_api_key, "FRED_API_KEY")
    fetcher = EconomicCalendarFetcher(api_key=api_key, cache_ttl=config.cache_ttl_minutes * 60)
    watcher_config = EconomicCalendarWatcherConfig(
        poll_interval_minutes=config.poll_interval_minutes,
        lookahead_hours=config.lookahead_hours,
        high_impact_avoid_hours=config.high_impact_avoid_hours,
    )
    watcher = EconomicCalendarWatcher(fetcher=fetcher, config=watcher_config, database_engine=database_engine)
    logger.info("EconomicCalendarWatcher created")
    return watcher


def create_news_trending_watcher(
    daemon_config: DaemonConfig,
    container: AppContainer | None = None,
    state: DaemonState | None = None,
) -> NewsTrendingWatcher | None:
    """Create news trending watcher for continuous discovery.

    Args:
        daemon_config: Daemon configuration
        container: DI container (auto-created if not provided)
        state: DaemonState for discovery routing

    Returns:
        NewsTrendingWatcher instance if enabled, None otherwise
    """
    config = daemon_config.news_trending_watcher
    if not config.enabled:
        logger.debug("NewsTrendingWatcher disabled in config")
        return None

    if container is None:
        from src.di.container import create_container

        container = create_container()

    websearch_fetcher = container.websearch_fetcher()
    pipeline = _build_pipeline(daemon_config, container, state)

    watcher_config = NewsTrendingWatcherConfig(
        poll_interval=config.poll_interval_minutes * 60,
        trending_window_minutes=config.trending_window_minutes,
        min_mention_threshold=config.min_mention_threshold,
        max_candidates_per_cycle=config.max_candidates_per_cycle,
        search_queries=config.search_queries,
        max_results_per_query=config.max_results_per_query,
    )

    watcher = NewsTrendingWatcher(
        pipeline=pipeline,
        websearch_fetcher=websearch_fetcher,
        config=watcher_config,
    )

    logger.info("News trending watcher created")
    return watcher


def create_options_flow_watcher(
    daemon_config: DaemonConfig,
) -> OptionsFlowWatcher | None:
    """Create options flow watcher if enabled.

    Args:
        daemon_config: Daemon configuration

    Returns:
        OptionsFlowWatcher instance if enabled, None otherwise
    """
    from src.data.options_flow import OptionsFlowFetcher
    from src.v1.watchers.options_flow_watcher import (
        OptionsFlowWatcher,
        OptionsFlowWatcherConfig,
    )

    config = daemon_config.options_flow_watcher
    if not config.enabled:
        logger.debug("OptionsFlowWatcher disabled in config")
        return None

    fetcher = OptionsFlowFetcher()
    watcher_config = OptionsFlowWatcherConfig(
        poll_interval_minutes=config.poll_interval_minutes,
        volume_spike_threshold=config.volume_spike_threshold,
        block_trade_threshold=config.block_trade_threshold,
        symbols=list(daemon_config.watchlist),
    )
    watcher = OptionsFlowWatcher(fetcher=fetcher, config=watcher_config)
    logger.info("OptionsFlowWatcher created")
    return watcher


def create_social_sentiment_watcher(
    daemon_config: DaemonConfig,
) -> SocialSentimentWatcher | None:
    """Create social sentiment watcher if enabled.

    Args:
        daemon_config: Daemon configuration

    Returns:
        SocialSentimentWatcher instance if enabled, None otherwise
    """
    from src.data.apewisdom import ApeWisdomFetcher
    from src.v1.watchers.social_sentiment_watcher import (
        SocialSentimentWatcher,
        SocialSentimentWatcherConfig,
    )

    config = daemon_config.social_sentiment_watcher
    if not config.enabled:
        logger.debug("SocialSentimentWatcher disabled in config")
        return None

    fetcher = ApeWisdomFetcher()
    watcher_config = SocialSentimentWatcherConfig(
        poll_interval_minutes=config.poll_interval_minutes,
        trending_rank_threshold=config.trending_rank_threshold,
        buzz_spike_threshold=config.buzz_spike_threshold,
        symbols=list(daemon_config.watchlist),
    )
    watcher = SocialSentimentWatcher(apewisdom_fetcher=fetcher, config=watcher_config)
    logger.info("SocialSentimentWatcher created")
    return watcher
