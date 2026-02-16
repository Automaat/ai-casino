"""Event watcher integration configuration."""

from pydantic import BaseModel, Field


class EventWatcherIntegrationConfig(BaseModel):
    """Event watcher → discovery integration config."""

    enable_discovery_integration: bool = True
    batch_evaluation_interval_minutes: int = 15
    max_candidates_per_batch: int = 10
    urgent_evaluation_threshold: float = 0.85
    urgent_bypass_batch: bool = True

    urgency_ttl_hours: dict[str, int] = Field(
        default_factory=lambda: {
            "IMMEDIATE": 4,
            "WATCHLIST": 24,
            "IGNORE": 0,
        }
    )

    news_watcher_use_discovery: bool = True
    social_watcher_use_discovery: bool = True
    anomaly_watcher_use_discovery: bool = False
    trump_watcher_use_discovery: bool = True

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"EventWatcherIntegrationConfig(enabled={self.enable_discovery_integration}, "
            f"batch_interval={self.batch_evaluation_interval_minutes}m)"
        )
