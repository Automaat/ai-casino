"""Configuration for analysis watchers and orchestration."""

from pydantic import BaseModel, Field


class AnalysisOrchestratorConfig(BaseModel):
    """Configuration for analysis orchestration."""

    max_concurrent_analyses: int = Field(default=3, ge=1, le=10)
    target_allocation_ttl_days: int = Field(default=7, ge=1, le=30)
    enable_position_sync: bool = True


class NewsSourcesConfig(BaseModel):
    """News source configuration."""

    enable_marketaux: bool = True
    enable_finnhub: bool = False
    enable_newsdata: bool = False
    enable_duckduckgo: bool = False


class NewsWatcherConfig(BaseModel):
    """Configuration for news watcher."""

    enabled: bool = False
    poll_interval_minutes: int = Field(default=5, ge=1, le=60)
    breaking_threshold_minutes: int = Field(default=15, ge=5, le=120)
    relevance_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    cooldown_minutes: int = Field(default=15, ge=1, le=120)
    max_concurrent_analyses: int = Field(default=2, ge=1, le=10)
    sources: NewsSourcesConfig = Field(default_factory=NewsSourcesConfig)


class SocialWatcherConfig(BaseModel):
    """Configuration for social media watcher."""

    enabled: bool = False
    poll_interval_minutes: int = Field(default=15, ge=5, le=60)
    volume_spike_threshold: float = Field(default=0.5, ge=0.1, le=2.0)
    viral_score_threshold: int = Field(default=1000, ge=100, le=10000)
    viral_upvote_ratio: float = Field(default=0.8, ge=0.5, le=1.0)
    subreddits: list[str] = Field(default_factory=lambda: ["wallstreetbets", "stocks"])
    relevance_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    cooldown_minutes: int = Field(default=15, ge=1, le=120)
    max_concurrent_analyses: int = Field(default=2, ge=1, le=10)


class FilingsWatcherConfig(BaseModel):
    """Configuration for SEC filings watcher."""

    enabled: bool = False
    poll_interval_minutes: int = Field(default=10, ge=5, le=60)
    filing_types: list[str] = Field(default_factory=lambda: ["8-K", "4", "13D"])
    cik_ticker_mapping_file: str = "~/.ai-casino/cik_ticker_map.json"
    relevance_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    cooldown_minutes: int = Field(default=15, ge=1, le=120)
    max_concurrent_analyses: int = Field(default=2, ge=1, le=10)


class AnomalyWatcherConfig(BaseModel):
    """Configuration for market anomaly watcher."""

    enabled: bool = False
    poll_interval_minutes: int = Field(default=15, ge=5, le=60)
    volume_spike_multiplier: float = Field(default=2.0, ge=1.5, le=5.0)
    price_move_threshold_pct: float = Field(default=5.0, ge=2.0, le=20.0)
    gap_threshold_pct: float = Field(default=3.0, ge=1.0, le=10.0)
    max_symbols_per_cycle: int = Field(default=5, ge=1, le=50)
    relevance_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    cooldown_minutes: int = Field(default=15, ge=1, le=120)
    max_concurrent_analyses: int = Field(default=2, ge=1, le=10)


class TrumpWatcherConfig(BaseModel):
    """Configuration for Trump social media watcher."""

    enabled: bool = False
    poll_interval_minutes: int = Field(default=5, ge=1, le=60)
    max_analyses: int = Field(default=5, ge=1, le=20)
    relevance_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    cooldown_minutes: int = Field(default=15, ge=1, le=120)
    max_concurrent_analyses: int = Field(default=2, ge=1, le=10)
