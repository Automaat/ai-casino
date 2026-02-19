"""Watchers package — market signal monitoring and event discovery."""

from src.watchers.anomaly_watcher import AnomalyWatcher, AnomalyWatcherConfig
from src.watchers.base import PeriodicWatcher, Watcher
from src.watchers.economic_calendar_watcher import EconomicCalendarWatcher, EconomicCalendarWatcherConfig
from src.watchers.news_trending_watcher import NewsTrendingWatcher, NewsTrendingWatcherConfig
from src.watchers.news_watcher import NewsWatcher, NewsWatcherConfig
from src.watchers.options_flow_watcher import OptionsFlowWatcher, OptionsFlowWatcherConfig
from src.watchers.pipeline import EventTriagePipeline
from src.watchers.social_sentiment_watcher import SocialSentimentWatcher, SocialSentimentWatcherConfig
from src.watchers.social_watcher import SocialWatcher, SocialWatcherConfig
from src.watchers.trump_watcher import TrumpWatcher, TrumpWatcherConfig

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
