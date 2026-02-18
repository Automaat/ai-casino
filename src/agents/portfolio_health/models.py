"""Portfolio health check LLM response models."""

from pydantic import BaseModel, Field


class PortfolioHealthLLMResponse(BaseModel):
    """LLM response for portfolio health analysis."""

    recommendations: list[str] = Field(description="Actionable recommendations for portfolio improvement")
    constraints: list[str] = Field(
        description="Trading constraints to enforce (e.g. 'block_buy:TECH', 'reduce:AAPL')"
    )
    reasoning: str = Field(description="Analysis reasoning summary")

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"PortfolioHealthLLMResponse(recommendations={len(self.recommendations)}, "
            f"constraints={len(self.constraints)})"
        )
