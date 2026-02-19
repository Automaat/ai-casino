"""Tests for watcher DI providers."""

from unittest.mock import Mock, patch

import pytest

from src.daemon.config import DaemonConfig, NewsWatcherConfig, SocialWatcherConfig
from src.di.providers.watchers import create_news_watcher, create_social_watcher
from src.watchers.news_watcher import NewsWatcher
from src.watchers.pipeline import EventTriagePipeline
from src.watchers.social_watcher import SocialWatcher


@pytest.fixture
def mock_pipeline():
    """Mock EventTriagePipeline."""
    return Mock(spec=EventTriagePipeline)


def test_news_watcher_provider_disabled():
    """Test news watcher provider returns None when disabled."""
    config = DaemonConfig(news_watcher=NewsWatcherConfig(enabled=False))
    result = create_news_watcher(Mock(), config)
    assert result is None


def test_news_watcher_provider_enabled(mock_pipeline):
    """Test news watcher provider creates instance when enabled."""
    from src.daemon.config.analysis import NewsSourcesConfig

    config = DaemonConfig(
        news_watcher=NewsWatcherConfig(
            enabled=True,
            poll_interval_minutes=5,
            breaking_threshold_minutes=15,
            sources=NewsSourcesConfig(
                enable_marketaux=False,
                enable_finnhub=False,
                enable_newsdata=False,
                enable_duckduckgo=True,
            ),
        )
    )
    mock_container = Mock()
    mock_fetcher = Mock()
    mock_fetcher.get_source_name.return_value = "duckduckgo"

    with patch("src.di.providers.watchers._build_pipeline", return_value=mock_pipeline):
        with patch("src.di.providers.data.create_duckduckgo_news_fetcher", return_value=mock_fetcher):
            result = create_news_watcher(Mock(), config, mock_container)

    assert isinstance(result, NewsWatcher)


def test_news_watcher_provider_enabled_no_sources(mock_pipeline):
    """Test news watcher provider with no sources returns None."""
    from src.daemon.config.analysis import NewsSourcesConfig

    config = DaemonConfig(
        news_watcher=NewsWatcherConfig(
            enabled=True,
            poll_interval_minutes=5,
            sources=NewsSourcesConfig(
                enable_marketaux=False,
                enable_finnhub=False,
                enable_newsdata=False,
                enable_duckduckgo=False,
            ),
        )
    )
    result = create_news_watcher(Mock(), config)
    assert result is None


def test_social_watcher_provider_disabled():
    """Test social watcher provider returns None when disabled."""
    config = DaemonConfig(social_watcher=SocialWatcherConfig(enabled=False))
    result = create_social_watcher(Mock(), config)
    assert result is None


def test_social_watcher_provider_enabled(mock_pipeline):
    """Test social watcher provider creates instance when enabled."""
    config = DaemonConfig(
        social_watcher=SocialWatcherConfig(
            enabled=True,
            poll_interval_minutes=15,
            volume_spike_threshold=0.5,
            viral_score_threshold=1000,
            subreddits=["wallstreetbets", "stocks"],
        )
    )
    mock_container = Mock()

    with patch("src.di.providers.watchers._build_pipeline", return_value=mock_pipeline):
        result = create_social_watcher(Mock(), config, mock_container)

    assert isinstance(result, SocialWatcher)
    assert result.poll_interval == 900  # 15 minutes * 60
    assert result.volume_spike_threshold == 0.5
    assert result.viral_score_threshold == 1000
    assert result.subreddits == ["wallstreetbets", "stocks"]
