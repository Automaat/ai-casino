"""Portfolio-aware filtering for discovery candidates."""

from loguru import logger
from pydantic import BaseModel, Field

from src.discovery.models import DiscoveryCandidate


class PortfolioFilterConfig(BaseModel):
    """Configuration for portfolio filters."""

    max_sector_concentration: float = 0.30  # 30% max per sector
    max_correlation_threshold: float = 0.75
    min_market_cap: float = 1e9  # $1B
    min_avg_volume: int = 1_000_000  # 1M shares/day
    price_range: tuple[float, float] = (10.0, 500.0)
    exclude_sectors: list[str] = Field(default_factory=list)
    max_watchlist_size: int = 50

    def __repr__(self) -> str:
        return f"PortfolioFilterConfig(sector_max={self.max_sector_concentration})"


class PortfolioFilterEngine:
    """Apply portfolio-aware filters to discovery candidates."""

    def __init__(self, config: PortfolioFilterConfig) -> None:
        self.config = config
        logger.info(f"Initialized PortfolioFilterEngine with {config}")

    async def filter_candidates(
        self,
        candidates: list[DiscoveryCandidate],
        current_positions: dict[str, object],
        current_watchlist: list[str],
    ) -> tuple[list[DiscoveryCandidate], list[str]]:
        """Filter candidates based on portfolio constraints.

        Args:
            candidates: Candidates to filter
            current_positions: Current portfolio positions
            current_watchlist: Current watchlist symbols

        Returns:
            Tuple of (accepted_candidates, rejection_reasons)
        """
        accepted: list[DiscoveryCandidate] = []
        rejection_reasons: list[str] = []

        # Calculate current sector exposure
        sector_exposure = self._calculate_sector_exposure(current_positions, current_watchlist, candidates)

        for candidate in candidates:
            # Check watchlist size limit
            if len(current_watchlist) + len(accepted) >= self.config.max_watchlist_size:
                rejection_reasons.append(f"{candidate.symbol}: watchlist size limit reached")
                continue

            # Check if already in watchlist
            if candidate.symbol in current_watchlist:
                rejection_reasons.append(f"{candidate.symbol}: already in watchlist")
                continue

            # Market cap filter
            market_cap = candidate.metadata.get("market_cap")
            if market_cap and isinstance(market_cap, (int, float)):
                if market_cap < self.config.min_market_cap:
                    rejection_reasons.append(
                        f"{candidate.symbol}: market cap ${market_cap / 1e9:.1f}B < min ${self.config.min_market_cap / 1e9:.1f}B"
                    )
                    continue

            # Volume filter
            avg_volume = candidate.metadata.get("avg_volume")
            if avg_volume and isinstance(avg_volume, (int, float)):
                if avg_volume < self.config.min_avg_volume:
                    rejection_reasons.append(
                        f"{candidate.symbol}: avg volume {avg_volume:,.0f} < min {self.config.min_avg_volume:,.0f}"
                    )
                    continue

            # Price range filter
            price = candidate.metadata.get("price")
            if price and isinstance(price, (int, float)):
                min_price, max_price = self.config.price_range
                if price < min_price or price > max_price:
                    rejection_reasons.append(
                        f"{candidate.symbol}: price ${price:.2f} outside range ${min_price}-${max_price}"
                    )
                    continue

            # Sector exclusion
            if candidate.sector in self.config.exclude_sectors:
                rejection_reasons.append(f"{candidate.symbol}: sector {candidate.sector} excluded")
                continue

            # Sector concentration check
            sector = candidate.sector
            current_sector_count = sector_exposure.get(sector, 0)
            total_count = len(current_watchlist) + len(accepted)

            if total_count > 0:
                projected_concentration = (current_sector_count + 1) / (total_count + 1)
                if projected_concentration > self.config.max_sector_concentration:
                    rejection_reasons.append(
                        f"{candidate.symbol}: sector {sector} concentration "
                        f"{projected_concentration:.1%} > max {self.config.max_sector_concentration:.1%}"
                    )
                    continue

            # Accept candidate
            accepted.append(candidate)
            sector_exposure[sector] = current_sector_count + 1

        logger.info(
            f"Portfolio filters: {len(accepted)}/{len(candidates)} accepted, {len(rejection_reasons)} rejected"
        )
        if rejection_reasons:
            for reason in rejection_reasons[:5]:  # Log first 5
                logger.debug(f"  Rejected: {reason}")

        return accepted, rejection_reasons

    def _calculate_sector_exposure(
        self,
        current_positions: dict[str, object],
        current_watchlist: list[str],
        candidates: list[DiscoveryCandidate],
    ) -> dict[str, int]:
        """Calculate current sector exposure from positions and watchlist.

        Args:
            current_positions: Current positions
            current_watchlist: Current watchlist
            candidates: Candidates with sector info

        Returns:
            Dict mapping sector to count
        """
        sector_counts: dict[str, int] = {}

        # Count sectors in watchlist (use candidate sector info)
        candidate_sectors = {c.symbol: c.sector for c in candidates}
        for symbol in current_watchlist:
            sector = candidate_sectors.get(symbol, "Unknown")
            sector_counts[sector] = sector_counts.get(sector, 0) + 1

        # Add positions (if sector info available)
        for symbol, position_data in current_positions.items():
            if isinstance(position_data, dict):
                sector = position_data.get("sector", "Unknown")
                sector_counts[sector] = sector_counts.get(sector, 0) + 1

        return sector_counts

    def __repr__(self) -> str:
        return f"PortfolioFilterEngine(config={self.config})"
