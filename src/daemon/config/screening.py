"""Stock screening and discovery configuration."""

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from src.daemon.config._validators import validate_time_range


class ScreeningConfig(BaseModel):
    """Configuration for after-hours watchlist screening."""

    enabled: bool = False
    screen_time: str = "16:30"
    screen_days: list[str] = Field(default_factory=lambda: ["mon", "tue", "wed", "thu", "fri"])
    criteria: Literal["momentum", "value", "breakout"] = "momentum"
    universe: Literal["SP500", "NASDAQ100", "COMBINED", "RUSSELL3000", "US_LIQUID"] = "COMBINED"
    top_n: int = 10
    watchlist_name: str = "daemon-screening"

    @model_validator(mode="after")
    def validate_screen_time(self) -> ScreeningConfig:
        """Validate screen_time is within 16:00-20:00."""
        if self.enabled:
            validate_time_range(self.screen_time, "screen_time", "after_hours")
        return self


class DiscoveryConfig(BaseModel):
    """Configuration for automated stock discovery (legacy, use event watchers for continuous discovery).

    Controls discovery engine creation, active candidate merging into watchlist, and outcome tracking.
    Kept for backward compatibility and discovery outcome tracking.
    """

    # Core enablement flag controlling discovery engine activation, candidate merging, and outcome tracking
    enabled: bool = False

    # Source enablement (used by StockDiscoveryEngine if discovery task is manually triggered)
    enable_technical_screening: bool = True
    enable_reddit_trending: bool = False
    enable_earnings_calendar: bool = True
    enable_sector_rotation: bool = True
    enable_volume_spikes: bool = False
    enable_price_gaps: bool = False
    enable_news_trending: bool = False

    # Technical screening
    screening_criteria: list[str] = Field(default_factory=lambda: ["momentum"])
    screening_universe: Literal["SP500", "NASDAQ100", "COMBINED", "RUSSELL3000", "US_LIQUID"] = "COMBINED"
    screening_top_n: int = 20

    # Social/Reddit
    reddit_min_mentions: int = 5
    reddit_min_upvote_ratio: float = 0.75

    # Earnings
    earnings_lookahead_days: int = 7

    # Trigger thresholds for intraday detection
    volume_spike_threshold: float = 2.0
    price_gap_threshold: float = 5.0

    # Scoring weights
    scoring_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "technical_weight": 0.35,
            "liquidity_weight": 0.25,
            "timing_weight": 0.20,
            "social_weight": 0.15,
            "volatility_weight": 0.05,
        }
    )

    # Limits
    max_discovered_per_cycle: int = 5
    min_composite_score: float = 0.60
    max_watchlist_size: int = 50

    # Portfolio filters
    portfolio_filters: Any = Field(
        default_factory=lambda: {
            "max_sector_concentration": 0.30,
            "min_market_cap": 1e9,
            "min_avg_volume": 1_000_000,
            "price_range": [10.0, 500.0],
            "exclude_sectors": [],
        }
    )

    # Lifecycle management
    candidate_ttl_days: int = 7
    auto_remove_on_signal: bool = False

    # State tracking
    track_outcomes: bool = True
    outcome_lookback_days: int = 90


class LiquidityFilterConfig(BaseModel):
    """Configuration for universe liquidity filtering."""

    min_market_cap: float = Field(
        default=1e9,
        gt=0,
        description="Minimum market capitalization in USD (must be > 0)",
    )
    min_avg_volume: int = Field(
        default=1_000_000,
        gt=0,
        description="Minimum average daily volume in shares (must be > 0)",
    )
    price_range: tuple[float, float] = (10.0, 500.0)

    @model_validator(mode="after")
    def validate_price_range(self) -> LiquidityFilterConfig:
        """Validate that price_range is (min_price, max_price) with 0 < min < max."""
        if self.price_range is None:
            return self

        if not isinstance(self.price_range, tuple) or len(self.price_range) != 2:
            msg = (
                f"price_range must be a tuple of two values (min_price, max_price), got {self.price_range!r}"
            )
            raise ValueError(msg)

        min_price, max_price = self.price_range

        if min_price <= 0 or max_price <= 0:
            msg = f"price_range values must be positive, got ({min_price}, {max_price})"
            raise ValueError(msg)

        if min_price >= max_price:
            msg = f"price_range must satisfy min_price < max_price, got ({min_price}, {max_price})"
            raise ValueError(msg)

        return self


class SectorRotationConfig(BaseModel):
    """Configuration for sector rotation analysis."""

    enabled: bool = False
    run_time: str = "16:15"
    run_days: list[str] = Field(default_factory=lambda: ["mon", "tue", "wed", "thu", "fri"])
    boost_factor: float = Field(
        default=0.15, ge=0.0, le=1.0, description="Sector weight boost factor (0.0-1.0)"
    )

    @model_validator(mode="after")
    def validate_run_time(self) -> SectorRotationConfig:
        """Validate run_time is in HH:MM format within 16:00-20:00."""
        if self.enabled:
            validate_time_range(self.run_time, "run_time", "after_hours")
        return self


class EarningsCalendarConfig(BaseModel):
    """Configuration for earnings calendar preparation."""

    enabled: bool = False
    fetch_time: str = "16:45"
    fetch_days: list[str] = Field(default_factory=lambda: ["mon"])
    lookahead_days: int = Field(default=3, ge=1, le=14)
    reduce_position_t1: bool = False
    position_reduction_factor: float = Field(default=0.5, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_fetch_time(self) -> EarningsCalendarConfig:
        """Validate fetch_time is in HH:MM format within 16:00-20:00."""
        if self.enabled:
            validate_time_range(self.fetch_time, "fetch_time", "after_hours")
        return self
