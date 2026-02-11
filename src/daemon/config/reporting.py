"""Reporting and tracking configuration."""

from pydantic import BaseModel, Field, model_validator

from src.daemon.config._validators import validate_time_range


class HealthConfig(BaseModel):
    """Configuration for API health checks and state cleanup."""

    enabled: bool = True
    check_interval_seconds: int = Field(
        default=5,
        ge=1,
        description="Interval in seconds between health checks (must be >= 1)",
    )
    archive_days: int = 30
    log_max_size_mb: int = 5
    health_dir: str = "~/.ai-casino/health"
    archive_dir: str = "~/.ai-casino/archive"


class ReportingConfig(BaseModel):
    """Configuration for automated performance reporting."""

    enabled: bool = False
    tearsheet_time: str = "16:30"
    benchmark: str = "SPY"
    retention_days: int = 30

    @model_validator(mode="after")
    def validate_tearsheet_time(self) -> ReportingConfig:
        """Validate tearsheet_time is in HH:MM format within 16:00-20:00 and retention_days >= 1."""
        if not self.enabled:
            return self

        validate_time_range(self.tearsheet_time, "tearsheet_time", "after_hours")

        if self.retention_days < 1:
            msg = "retention_days must be >= 1 when reporting enabled"
            raise ValueError(msg)

        return self


class SignalTrackingConfig(BaseModel):
    """Configuration for signal accuracy tracking."""

    enabled: bool = True
    tracking_time: str = "17:00"

    @model_validator(mode="after")
    def validate_tracking_time(self) -> SignalTrackingConfig:
        """Validate tracking_time is in HH:MM format within 16:00-20:00."""
        if self.enabled:
            validate_time_range(self.tracking_time, "tracking_time", "after_hours")
        return self
