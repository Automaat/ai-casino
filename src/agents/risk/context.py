"""Shared context for risk assessment."""

from dataclasses import dataclass

from src.metrics.portfolio_var import PortfolioVaRResult


@dataclass
class RiskContext:
    """Shared context for risk assessment, passed through component calls."""

    portfolio_cdar: float | None = None
    latest_portfolio_var: PortfolioVaRResult | None = None
