"""Configuration for analysis watchers and orchestration."""

from pydantic import BaseModel, ConfigDict, Field


class AnalysisOrchestratorConfig(BaseModel):
    """Configuration for analysis orchestration."""

    max_concurrent_analyses: int = Field(default=3, ge=1, le=10)
    target_allocation_ttl_days: int = Field(default=7, ge=1, le=30)
    enable_position_sync: bool = True

    enable_supervisor_routing: bool = Field(
        default=False, description="Enable supervisor-driven conditional worker execution"
    )
    supervisor_planning_timeout_ms: int = Field(
        default=5000, ge=1000, le=10000, description="Timeout for supervisor planning phase"
    )
    worker_execution_timeout_ms: int = Field(
        default=30000, ge=10000, le=60000, description="Timeout for worker execution phase"
    )


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
    period_days: int = Field(default=60, ge=30, le=180)
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
    period_days: int = Field(default=60, ge=30, le=180)


class FilingsWatcherConfig(BaseModel):
    """Configuration for SEC filings watcher."""

    enabled: bool = False
    poll_interval_minutes: int = Field(default=10, ge=5, le=60)
    filing_types: list[str] = Field(default_factory=lambda: ["8-K", "4", "13D"])
    cik_ticker_mapping_file: str = "~/.ai-casino/cik_ticker_map.json"
    relevance_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    cooldown_minutes: int = Field(default=15, ge=1, le=120)
    max_concurrent_analyses: int = Field(default=2, ge=1, le=10)
    period_days: int = Field(default=60, ge=30, le=180)


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
    period_days: int = Field(default=60, ge=30, le=180)


class TrumpWatcherConfig(BaseModel):
    """Configuration for Trump social media watcher."""

    enabled: bool = False
    poll_interval_minutes: int = Field(default=5, ge=1, le=60)
    relevance_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    cooldown_minutes: int = Field(default=15, ge=1, le=120)
    max_concurrent_analyses: int = Field(default=2, ge=1, le=10)
    period_days: int = Field(default=60, ge=30, le=180)


class NewsTrendingWatcherConfig(BaseModel):
    """Configuration for news trending watcher (continuous discovery)."""

    enabled: bool = False  # Opt-in to avoid unexpected web searches and LLM costs
    poll_interval_minutes: int = Field(default=10, ge=5, le=60)
    trending_window_minutes: int = Field(default=60, ge=30, le=180)
    min_mention_threshold: int = Field(default=3, ge=2, le=10)
    relevance_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    max_candidates_per_cycle: int = Field(default=5, ge=1, le=20)
    search_queries: list[str] = Field(
        default_factory=lambda: [
            "trending stocks today",
            "hot stocks right now",
            "stock market movers",
        ]
    )
    max_results_per_query: int = Field(default=10, ge=5, le=20)

    model_config = ConfigDict(arbitrary_types_allowed=True)
