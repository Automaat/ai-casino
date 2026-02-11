"""Tests for watcher DI providers."""

from src.daemon.config import DaemonConfig, NewsWatcherConfig, SocialWatcherConfig
from src.daemon.watchers.news_watcher import NewsWatcher
from src.daemon.watchers.social_watcher import SocialWatcher
from src.di.container import create_container


def test_news_watcher_provider_disabled():
    """Test news watcher provider returns None when disabled."""
    config = DaemonConfig(news_watcher=NewsWatcherConfig(enabled=False))
    container = create_container()
    container.daemon_config.override(config)

    watcher = container.news_watcher()

    assert watcher is None


def test_news_watcher_provider_enabled():
    """Test news watcher provider creates instance when enabled."""
    config = DaemonConfig(
        news_watcher=NewsWatcherConfig(
            enabled=True,
            poll_interval_minutes=5,
            relevance_threshold=0.7,
            cooldown_minutes=15,
        )
    )
    container = create_container()
    container.daemon_config.override(config)

    watcher = container.news_watcher()

    assert isinstance(watcher, NewsWatcher)
    assert watcher.poll_interval == 300  # 5 minutes * 60
    assert watcher.relevance_threshold == 0.7
    assert watcher.cooldown_minutes == 15
    assert watcher.breaking_threshold_minutes == 15  # default


def test_social_watcher_provider_disabled():
    """Test social watcher provider returns None when disabled."""
    config = DaemonConfig(social_watcher=SocialWatcherConfig(enabled=False))
    container = create_container()
    container.daemon_config.override(config)

    watcher = container.social_watcher()

    assert watcher is None


def test_social_watcher_provider_enabled():
    """Test social watcher provider creates instance when enabled."""
    config = DaemonConfig(
        social_watcher=SocialWatcherConfig(
            enabled=True,
            poll_interval_minutes=15,
            relevance_threshold=0.7,
            cooldown_minutes=15,
            volume_spike_threshold=0.5,
            viral_score_threshold=1000,
            subreddits=["wallstreetbets", "stocks"],
        )
    )
    container = create_container()
    container.daemon_config.override(config)

    watcher = container.social_watcher()

    assert isinstance(watcher, SocialWatcher)
    assert watcher.poll_interval == 900  # 15 minutes * 60
    assert watcher.relevance_threshold == 0.7
    assert watcher.cooldown_minutes == 15
    assert watcher.volume_spike_threshold == 0.5
    assert watcher.viral_score_threshold == 1000
    assert watcher.subreddits == ["wallstreetbets", "stocks"]


def test_news_watcher_singleton():
    """Test news watcher is singleton."""
    config = DaemonConfig(news_watcher=NewsWatcherConfig(enabled=True))
    container = create_container()
    container.daemon_config.override(config)

    watcher1 = container.news_watcher()
    watcher2 = container.news_watcher()

    assert watcher1 is watcher2


def test_social_watcher_singleton():
    """Test social watcher is singleton."""
    config = DaemonConfig(social_watcher=SocialWatcherConfig(enabled=True))
    container = create_container()
    container.daemon_config.override(config)

    watcher1 = container.social_watcher()
    watcher2 = container.social_watcher()

    assert watcher1 is watcher2
