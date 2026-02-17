"""Stock screening and discovery configuration."""

from pydantic import BaseModel, Field, model_validator

from src.daemon.config._validators import validate_time_range


class DiscoveryConfig(BaseModel):
    """Configuration for discovery outcome tracking.

    Controls candidate TTL and outcome analytics (T+7d/30d returns).
    Continuous candidate discovery is handled by EventWatchers.
    """

    enabled: bool = False
    candidate_ttl_days: int = 7
    track_outcomes: bool = True
    outcome_lookback_days: int = 90
    max_watchlist_size: int = Field(default=10, ge=1)


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
