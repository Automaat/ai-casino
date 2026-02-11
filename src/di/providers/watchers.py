"""DI providers for event watchers."""

from loguru import logger

from src.cache.historical import HistoricalCache
from src.daemon.config import DaemonConfig
from src.daemon.watchers.news_watcher import NewsWatcher
from src.daemon.watchers.social_watcher import SocialWatcher


def create_news_watcher(
    historical_cache: HistoricalCache,
    daemon_config: DaemonConfig,
) -> NewsWatcher | None:
    """Create news watcher if enabled.

    Args:
        historical_cache: Historical cache for data persistence
        daemon_config: Daemon configuration

    Returns:
        NewsWatcher instance if enabled, None otherwise
    """
    config = daemon_config.news_watcher
    if not config.enabled:
        logger.debug("NewsWatcher disabled in config")
        return None

    return NewsWatcher(
        historical_cache=historical_cache,
        poll_interval=config.poll_interval_minutes * 60,
        relevance_threshold=config.relevance_threshold,
        cooldown_minutes=config.cooldown_minutes,
        breaking_threshold_minutes=config.breaking_threshold_minutes,
        max_concurrent_analyses=config.max_concurrent_analyses,
    )


def create_social_watcher(
    historical_cache: HistoricalCache,
    daemon_config: DaemonConfig,
) -> SocialWatcher | None:
    """Create social media watcher if enabled.

    Args:
        historical_cache: Historical cache for data persistence
        daemon_config: Daemon configuration

    Returns:
        SocialWatcher instance if enabled, None otherwise
    """
    config = daemon_config.social_watcher
    if not config.enabled:
        logger.debug("SocialWatcher disabled in config")
        return None

    return SocialWatcher(
        historical_cache=historical_cache,
        poll_interval=config.poll_interval_minutes * 60,
        relevance_threshold=config.relevance_threshold,
        cooldown_minutes=config.cooldown_minutes,
        volume_spike_threshold=config.volume_spike_threshold,
        viral_score_threshold=config.viral_score_threshold,
        viral_upvote_ratio=config.viral_upvote_ratio,
        subreddits=config.subreddits,
        max_concurrent_analyses=config.max_concurrent_analyses,
    )
