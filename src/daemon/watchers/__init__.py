"""Event watcher implementations."""

from src.daemon.watchers.anomaly_watcher import AnomalyWatcher
from src.daemon.watchers.news_watcher import NewsWatcher
from src.daemon.watchers.social_watcher import SocialWatcher
from src.daemon.watchers.trump_watcher import TrumpWatcher

__all__ = ["AnomalyWatcher", "NewsWatcher", "SocialWatcher", "TrumpWatcher"]
