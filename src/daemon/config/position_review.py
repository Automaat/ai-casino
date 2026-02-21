"""Position review task configuration."""

from typing import Literal

from pydantic import BaseModel, Field


class PositionReviewConfig(BaseModel):
    """Configuration for scheduled position review."""

    enabled: bool = False
    run_during: Literal["regular_market"] = "regular_market"
    interval_minutes: int = Field(default=60, ge=15, le=480)

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"PositionReviewConfig(enabled={self.enabled}, "
            f"run_during={self.run_during}, interval={self.interval_minutes}m)"
        )
