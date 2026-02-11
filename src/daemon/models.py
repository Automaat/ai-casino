"""Typed models for daemon components."""

from pydantic import BaseModel, Field


class EarningsFlags(BaseModel):
    """Earnings proximity flags for trading decisions."""

    upcoming_earnings: bool = Field(description="Whether earnings report is upcoming")
    days_until_earnings: int | None = Field(default=None, description="Days until earnings report")
    pre_earnings_zone: str | None = Field(
        default=None,
        description="Pre-earnings proximity zone (T-1 or T-3)",
    )

    def __repr__(self) -> str:
        """String representation."""
        if not self.upcoming_earnings:
            return "EarningsFlags(upcoming=False)"
        return f"EarningsFlags(upcoming=True, days={self.days_until_earnings}, zone={self.pre_earnings_zone})"
