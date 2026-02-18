"""Comparative analysis models."""

from enum import StrEnum

from pydantic import BaseModel, Field


class RelativeValuation(StrEnum):
    """Relative valuation assessment."""

    RELATIVELY_UNDERVALUED = "RELATIVELY_UNDERVALUED"
    FAIRLY_VALUED = "FAIRLY_VALUED"
    RELATIVELY_OVERVALUED = "RELATIVELY_OVERVALUED"


class ComparativeAnalysis(BaseModel):
    """Comparative analysis result."""

    relative_valuation: RelativeValuation
    pe_vs_sector: float | None = Field(description="P/E ratio relative to sector (stock P/E / sector P/E)")
    pe_vs_market: float | None = Field(description="P/E ratio relative to market (stock P/E / market P/E)")
    perf_vs_sector_ytd: float | None = Field(
        description="YTD performance difference vs sector (stock - sector)"
    )
    perf_vs_sector_3m: float | None = Field(
        description="3M performance difference vs sector (stock - sector)"
    )
    perf_vs_market_ytd: float | None = Field(
        description="YTD performance difference vs market (stock - market)"
    )
    perf_vs_market_3m: float | None = Field(
        description="3M performance difference vs market (stock - market)"
    )
    sector_etf: str
    interpretation: str
    confidence: float = Field(ge=0.0, le=1.0)

    def __repr__(self) -> str:
        """String representation."""
        pe_str = f"{self.pe_vs_sector:.2f}" if self.pe_vs_sector else "N/A"
        return (
            f"ComparativeAnalysis(valuation={self.relative_valuation.value}, "
            f"pe_vs_sector={pe_str}, confidence={self.confidence:.2f})"
        )
