"""Sector attribution analysis for portfolio allocation and P&L contribution."""

from datetime import UTC, datetime

import yfinance as yf
from loguru import logger
from pydantic import BaseModel, Field

from src.daemon.positions import PositionRecord
from src.data.broker import BrokerPosition
from src.data.comparative import SECTOR_MAPPING, Sector

# Hardcoded SPY benchmark sector weights (approximate)
BENCHMARK_WEIGHTS: dict[str, float] = {
    "TECHNOLOGY": 0.29,
    "HEALTHCARE": 0.13,
    "FINANCIALS": 0.13,
    "CONSUMER_DISCRETIONARY": 0.11,
    "COMMUNICATION_SERVICES": 0.09,
    "INDUSTRIALS": 0.08,
    "CONSUMER_STAPLES": 0.06,
    "ENERGY": 0.04,
    "UTILITIES": 0.03,
    "REAL_ESTATE": 0.02,
    "MATERIALS": 0.02,
}


class SectorContribution(BaseModel):
    """Sector contribution metrics."""

    sector: str = Field(description="Sector name (e.g., TECHNOLOGY)")
    sector_etf: str = Field(description="Sector ETF ticker (e.g., XLK)")
    total_value: float = Field(description="Total market value in sector")
    portfolio_weight: float = Field(description="Portfolio weight (0.0-1.0)")
    benchmark_weight: float = Field(description="SPY benchmark weight (0.0-1.0)")
    over_under_weight: float = Field(description="Portfolio - benchmark weight")
    pnl: float = Field(description="Unrealized P&L for sector")
    return_pct: float = Field(description="Return percentage for sector")
    position_count: int = Field(description="Number of positions in sector")

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"SectorContribution(sector={self.sector}, weight={self.portfolio_weight:.2%}, "
            f"pnl=${self.pnl:,.2f})"
        )


class SectorAttributionAnalysis(BaseModel):
    """Complete sector attribution analysis."""

    contributions: list[SectorContribution] = Field(description="Sector-level contribution metrics")
    total_portfolio_value: float = Field(description="Total portfolio market value")
    benchmark_name: str = Field(default="SPY", description="Benchmark index name")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC), description="Analysis timestamp")

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"SectorAttributionAnalysis(sectors={len(self.contributions)}, "
            f"portfolio_value=${self.total_portfolio_value:,.2f})"
        )


class SectorAttributionAnalyzer:
    """Analyzes portfolio sector allocation and contribution vs benchmark."""

    def __init__(self) -> None:
        """Initialize sector attribution analyzer."""
        self._sector_cache: dict[str, Sector] = {}
        logger.info("Initialized SectorAttributionAnalyzer")

    async def analyze_attribution(
        self,
        positions: list[PositionRecord],
        broker_positions: dict[str, BrokerPosition],
    ) -> SectorAttributionAnalysis:
        """Analyze sector attribution for portfolio.

        Args:
            positions: List of position records from state
            broker_positions: Dict of broker positions with current prices

        Returns:
            SectorAttributionAnalysis with sector-level metrics
        """
        if not positions:
            logger.warning("No positions to analyze")
            return SectorAttributionAnalysis(
                contributions=[],
                total_portfolio_value=0.0,
            )

        logger.info(f"Analyzing sector attribution for {len(positions)} positions")

        # Aggregate by sector
        sector_pnl: dict[str, float] = {}
        sector_value: dict[str, float] = {}
        sector_cost: dict[str, float] = {}
        sector_count: dict[str, int] = {}

        for position in positions:
            # Get broker position for current price
            broker_pos = broker_positions.get(position.symbol)
            if not broker_pos:
                logger.warning(f"Broker position not found for {position.symbol}, skipping")
                continue

            # Lookup sector (with caching)
            sector = await self._get_position_sector(position.symbol)
            sector_name = sector.name

            # Calculate metrics
            current_price = broker_pos.avg_entry_price + (broker_pos.unrealized_pnl / broker_pos.qty)
            position_value = current_price * position.current_qty
            position_cost = position.entry_price * position.current_qty
            position_pnl = position_value - position_cost

            # Aggregate
            sector_pnl[sector_name] = sector_pnl.get(sector_name, 0.0) + position_pnl
            sector_value[sector_name] = sector_value.get(sector_name, 0.0) + position_value
            sector_cost[sector_name] = sector_cost.get(sector_name, 0.0) + position_cost
            sector_count[sector_name] = sector_count.get(sector_name, 0) + 1

        # Calculate total portfolio value
        total_value = sum(sector_value.values())
        if total_value == 0:
            logger.warning("Total portfolio value is zero")
            return SectorAttributionAnalysis(
                contributions=[],
                total_portfolio_value=0.0,
            )

        # Build contributions
        contributions = []
        for sector_name in sorted(sector_value.keys()):
            value = sector_value[sector_name]
            cost = sector_cost[sector_name]
            pnl = sector_pnl[sector_name]
            count = sector_count[sector_name]

            portfolio_weight = value / total_value
            benchmark_weight = BENCHMARK_WEIGHTS.get(sector_name, 0.0)
            over_under = portfolio_weight - benchmark_weight
            return_pct = (pnl / cost * 100) if cost > 0 else 0.0

            # Get sector ETF ticker
            sector_enum = Sector[sector_name]
            sector_etf = sector_enum.value

            contributions.append(
                SectorContribution(
                    sector=sector_name,
                    sector_etf=sector_etf,
                    total_value=value,
                    portfolio_weight=portfolio_weight,
                    benchmark_weight=benchmark_weight,
                    over_under_weight=over_under,
                    pnl=pnl,
                    return_pct=return_pct,
                    position_count=count,
                )
            )

        logger.info(f"Sector attribution: {len(contributions)} sectors, portfolio_value=${total_value:,.2f}")

        return SectorAttributionAnalysis(
            contributions=contributions,
            total_portfolio_value=total_value,
        )

    async def _get_position_sector(self, symbol: str) -> Sector:
        """Get sector for a position symbol with caching.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Sector enum (UNKNOWN if lookup fails)
        """
        # Check cache
        if symbol in self._sector_cache:
            return self._sector_cache[symbol]

        # Fetch from yfinance
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            if not info or "symbol" not in info:
                logger.warning(f"No yfinance data for {symbol}, using UNKNOWN sector")
                sector = Sector.UNKNOWN
            else:
                sector_name = info.get("sector")
                if not sector_name:
                    logger.warning(f"No sector in yfinance data for {symbol}")
                    sector = Sector.UNKNOWN
                else:
                    # Map to Sector enum
                    sector = SECTOR_MAPPING.get(sector_name.lower(), Sector.UNKNOWN)
                    logger.debug(f"Mapped {symbol} to sector {sector.name}")

        except Exception as e:
            logger.opt(exception=True).warning(f"Error fetching sector for {symbol}: {e}")
            sector = Sector.UNKNOWN

        # Cache and return
        self._sector_cache[symbol] = sector
        return sector
