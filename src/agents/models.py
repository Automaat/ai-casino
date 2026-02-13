"""Typed models for agent components."""

from datetime import date

from pydantic import BaseModel, Field


class FundamentalMetrics(BaseModel):
    """Fundamental analysis metrics."""

    pe_ratio: float | None = Field(default=None, description="Price-to-Earnings ratio")
    eps: float | None = Field(default=None, description="Earnings Per Share")
    revenue_growth_yoy: float | None = Field(default=None, description="Year-over-year revenue growth")
    debt_to_equity: float | None = Field(default=None, description="Debt-to-Equity ratio")
    earnings_growth_yoy: float | None = Field(default=None, description="Year-over-year earnings growth")
    current_ratio: float | None = Field(default=None, description="Current ratio (liquidity)")

    @property
    def completeness_ratio(self) -> float:
        """Calculate data completeness (0.0-1.0).

        Returns:
            Ratio of non-None fields to total fields
        """
        all_fields = [
            self.pe_ratio,
            self.eps,
            self.revenue_growth_yoy,
            self.earnings_growth_yoy,
            self.debt_to_equity,
            self.current_ratio,
        ]
        non_none = sum(1 for v in all_fields if v is not None)
        return non_none / len(all_fields)

    def __repr__(self) -> str:
        """String representation."""
        return f"FundamentalMetrics(pe={self.pe_ratio}, eps={self.eps}, completeness={self.completeness_ratio:.1%})"


class EarningsFlags(BaseModel):
    """Earnings calendar flags."""

    upcoming_earnings: bool = Field(description="Whether earnings report is upcoming")
    days_until_earnings: int | None = Field(default=None, description="Days until earnings report")
    earnings_date: date | None = Field(default=None, description="Date of earnings report")
    estimate_eps: float | None = Field(default=None, description="Estimated EPS for earnings")

    def __repr__(self) -> str:
        """String representation."""
        if not self.upcoming_earnings:
            return "EarningsFlags(upcoming=False)"
        return (
            f"EarningsFlags(upcoming=True, days={self.days_until_earnings}, "
            f"date={self.earnings_date}, estimate_eps={self.estimate_eps})"
        )
