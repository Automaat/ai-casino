"""Risk management configuration."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from src.daemon.config._validators import validate_time_range


class RiskLimitsConfig(BaseModel):
    """Configuration for portfolio-level VaR risk limits."""

    enabled: bool = False
    max_var_95: float = Field(default=0.03, ge=0.001, le=0.20)
    max_cvar_99: float = Field(default=0.05, ge=0.001, le=0.30)
    lookback_days: int = Field(default=90, ge=20, le=365)
    adaptive_stop_loss: bool = True
    cdar_stop_threshold: float = Field(default=0.10, ge=0.01, le=0.50)
    atr_multiplier_min: float = Field(default=1.0, ge=0.5, le=2.0)
    report_dir: str = "~/.ai-casino/risk-reports"
    report_time: str = "16:30"


class PreTradeBacktestingConfig(BaseModel):
    """Configuration for pre-trade backtesting validation."""

    enabled: bool = False
    lookback_days: int = Field(default=180, ge=30, le=365)
    min_sharpe_threshold: float = Field(default=0.5, ge=-1.0, le=3.0)
    max_drawdown_threshold: float = Field(default=0.25, ge=0.05, le=0.50)
    confidence_penalty_multiplier: float = Field(default=0.7, ge=0.1, le=1.0)


class PositionSizingConfig(BaseModel):
    """Configuration for position sizing strategy."""

    primary_goal: Literal["maximize_returns", "minimize_risk", "balanced"] = "balanced"
    risk_tolerance: Literal["conservative", "moderate", "aggressive"] = "moderate"
    complexity: Literal["simple", "advanced"] = "simple"

    max_risk_per_trade_pct: float = Field(default=2.0, ge=0.1, le=10.0)
    max_single_position_pct: float = Field(default=20.0, ge=1.0, le=50.0)
    max_total_exposure_pct: float = Field(default=80.0, ge=10.0, le=100.0)
    min_reward_risk_ratio: float = Field(default=2.0, ge=1.0, le=10.0)

    blend_weight_optimization: float = Field(default=0.5, ge=0.0, le=1.0)
    blend_weight_risk_based: float = Field(default=0.5, ge=0.0, le=1.0)

    confidence_scaling_enabled: bool = False
    confidence_high_threshold: float = Field(default=0.8, ge=0.5, le=1.0)
    confidence_low_threshold: float = Field(default=0.6, ge=0.3, le=0.9)
    confidence_low_reduction_factor: float = Field(default=0.5, ge=0.1, le=0.9)

    use_monte_carlo_adjustment: bool = False
    monte_carlo_risk_multiplier: float = Field(default=0.7, ge=0.1, le=1.0)

    @model_validator(mode="after")
    def validate_blend_weights(self) -> PositionSizingConfig:
        """Validate blend weights sum to 1.0."""
        total = self.blend_weight_optimization + self.blend_weight_risk_based
        if not (0.99 <= total <= 1.01):
            msg = f"Blend weights must sum to 1.0, got {total:.2f}"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def validate_confidence_thresholds(self) -> PositionSizingConfig:
        """Validate confidence thresholds are properly ordered."""
        if self.confidence_low_threshold >= self.confidence_high_threshold:
            msg = (
                f"confidence_low_threshold ({self.confidence_low_threshold}) must be < "
                f"confidence_high_threshold ({self.confidence_high_threshold})"
            )
            raise ValueError(msg)
        return self


class PositionCircuitBreakerConfig(BaseModel):
    """Non-overridable drawdown limits for position management."""

    enabled: bool = False
    position_max_drawdown_pct: float = Field(default=8.0, ge=1.0, le=50.0)
    portfolio_daily_loss_limit_pct: float = Field(default=2.0, ge=0.5, le=10.0)
    portfolio_peak_drawdown_pct: float = Field(default=5.0, ge=1.0, le=20.0)
    portfolio_peak_size_reduction: float = Field(default=0.5, ge=0.1, le=1.0)


class PositionManagementConfig(BaseModel):
    """Configuration for position management."""

    enabled: bool = False
    trailing_stop_enabled: bool = True
    trailing_stop_percent: float = Field(default=3.0, ge=0.5, le=10.0)
    min_stop_gap_dollars: float = Field(default=0.10, ge=0.01, le=1.0)
    use_atr_trailing_stop: bool = False
    atr_trailing_multiplier: float = Field(default=2.5, ge=1.0, le=5.0)
    atr_ratchet_1r: float = Field(default=2.0, ge=0.5, le=4.0)
    atr_ratchet_2r: float = Field(default=1.5, ge=0.5, le=3.0)
    partial_profit_enabled: bool = True
    profit_target_1_percent: float = Field(default=5.0, ge=1.0, le=20.0)
    profit_target_1_sell_pct: float = Field(default=0.5, ge=0.1, le=1.0)
    profit_target_2_percent: float = Field(default=10.0, ge=5.0, le=50.0)
    profit_target_2_sell_pct: float = Field(default=1.0, ge=0.1, le=1.0)
    time_exit_enabled: bool = True
    max_holding_days: int = Field(default=30, ge=1, le=180)
    breakeven_enabled: bool = True
    breakeven_activation_percent: float = Field(default=5.0, ge=1.0, le=20.0)
    conviction_scaling_enabled: bool = True
    conviction_decrease_threshold: float = Field(default=0.15, ge=0.05, le=0.5)
    conviction_scale_out_percent: float = Field(default=0.5, ge=0.1, le=1.0)
    conviction_decay_enabled: bool = False
    conviction_decay_rate: float = Field(default=0.95, ge=0.5, le=1.0)
    conviction_exit_threshold: float = Field(default=0.35, ge=0.1, le=0.8)
    conviction_history_length: int = Field(default=10, ge=3, le=50)
    use_r_multiple_targets: bool = False
    r_target_1_sell_pct: float = Field(default=0.25, ge=0.1, le=1.0)
    r_target_2_sell_pct: float = Field(default=0.25, ge=0.1, le=1.0)
    r_high_conviction_delay: int = Field(default=1, ge=0, le=3)
    high_conviction_threshold: float = Field(default=0.8, ge=0.5, le=1.0)
    whipsaw_prevention_enabled: bool = False
    adx_exit_filter_threshold: float = Field(default=20.0, ge=5.0, le=50.0)
    sell_confirmation_cycles: int = Field(default=2, ge=1, le=5)
    re_entry_cooldown_hours: int = Field(default=24, ge=1, le=168)
    circuit_breaker: PositionCircuitBreakerConfig = Field(default_factory=PositionCircuitBreakerConfig)


class MonteCarloConfig(BaseModel):
    """Monte Carlo stress testing configuration."""

    enabled: bool = False
    schedule_time: str = "17:00"
    schedule_days: list[str] = Field(default_factory=lambda: ["sun"])

    # Simulation parameters
    num_simulations: int = Field(default=5000, ge=100, le=10000)
    horizon_days: int = Field(default=252, ge=1, le=504)
    simulation_method: str = "PARAMETRIC"
    min_historical_days: int = Field(default=90, ge=30)
    random_seed: int | None = None

    # Risk thresholds
    loss_threshold: float = Field(default=0.10, ge=0.0, le=1.0)
    max_acceptable_prob: float = Field(default=0.15, ge=0.0, le=1.0)

    # Position sizing integration (opt-in)
    adjust_position_sizing: bool = False

    # History retention
    max_history_records: int = Field(default=52, ge=1, le=520)

    @model_validator(mode="after")
    def validate_schedule_time(self) -> MonteCarloConfig:
        """Validate schedule_time is within 16:00-20:00 for after-hours or any time for weekends."""
        if not self.enabled:
            return self

        weekday_days = ["mon", "tue", "wed", "thu", "fri"]
        is_weekday_only = any(d.lower() in weekday_days for d in self.schedule_days)

        if is_weekday_only:
            validate_time_range(self.schedule_time, "schedule_time", "after_hours")

        return self
