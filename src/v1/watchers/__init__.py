"""Watchers package — market signal monitoring and event discovery."""

from src.v1.watchers.anomaly_watcher import AnomalyWatcher, AnomalyWatcherConfig
from src.v1.watchers.base import PeriodicWatcher, Watcher
from src.v1.watchers.economic_calendar_watcher import EconomicCalendarWatcher, EconomicCalendarWatcherConfig
from src.v1.watchers.news_trending_watcher import NewsTrendingWatcher, NewsTrendingWatcherConfig
from src.v1.watchers.news_watcher import NewsWatcher, NewsWatcherConfig
from src.v1.watchers.options_flow_watcher import OptionsFlowWatcher, OptionsFlowWatcherConfig
from src.v1.watchers.pipeline import EventTriagePipeline
from src.v1.watchers.social_sentiment_watcher import SocialSentimentWatcher, SocialSentimentWatcherConfig
from src.v1.watchers.social_watcher import SocialWatcher, SocialWatcherConfig
from src.v1.watchers.trump_watcher import TrumpWatcher, TrumpWatcherConfig

__all__ = [
    "AnomalyWatcher",
    "AnomalyWatcherConfig",
    "EconomicCalendarWatcher",
    "EconomicCalendarWatcherConfig",
    "EventTriagePipeline",
    "NewsTrendingWatcher",
    "NewsTrendingWatcherConfig",
    "NewsWatcher",
    "NewsWatcherConfig",
    "OptionsFlowWatcher",
    "OptionsFlowWatcherConfig",
    "PeriodicWatcher",
    "SocialSentimentWatcher",
    "SocialSentimentWatcherConfig",
    "SocialWatcher",
    "SocialWatcherConfig",
    "TrumpWatcher",
    "TrumpWatcherConfig",
    "Watcher",
]
