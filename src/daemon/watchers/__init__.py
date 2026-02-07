"""Event watcher implementations."""

from src.daemon.watchers.anomaly_watcher import AnomalyWatcher
from src.daemon.watchers.news_watcher import NewsWatcher
from src.daemon.watchers.social_watcher import SocialWatcher

__all__ = ["AnomalyWatcher", "NewsWatcher", "SocialWatcher"]
