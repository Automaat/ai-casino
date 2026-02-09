"""Portfolio-aware filtering for discovery candidates."""

from loguru import logger
from pydantic import BaseModel, Field

from src.discovery.models import DiscoveryCandidate


class PortfolioFilterConfig(BaseModel):
    """Configuration for portfolio filters."""

    max_sector_concentration: float = 0.30  # 30% max per sector
    min_market_cap: float = 1e9  # $1B
    min_avg_volume: int = 1_000_000  # 1M shares/day
    price_range: tuple[float, float] = (10.0, 500.0)
    exclude_sectors: list[str] = Field(default_factory=list)
    max_watchlist_size: int = 50

    def __repr__(self) -> str:
        """Return string representation."""
        return f"PortfolioFilterConfig(sector_max={self.max_sector_concentration})"


class PortfolioFilterEngine:
    """Apply portfolio-aware filters to discovery candidates."""

    def __init__(self, config: PortfolioFilterConfig) -> None:
        """Initialize portfolio filter engine."""
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
            # Check if candidate passes all filters
            rejection = self._check_candidate_filters(candidate, current_watchlist, accepted, sector_exposure)

            if rejection:
                rejection_reasons.append(rejection)
            else:
                # Accept candidate
                accepted.append(candidate)
                sector_exposure[candidate.sector] = sector_exposure.get(candidate.sector, 0) + 1

        logger.info(
            f"Portfolio filters: {len(accepted)}/{len(candidates)} accepted, "
            f"{len(rejection_reasons)} rejected"
        )
        if rejection_reasons:
            for reason in rejection_reasons[:5]:  # Log first 5
                logger.debug(f"  Rejected: {reason}")

        return accepted, rejection_reasons

    def _check_candidate_filters(
        self,
        candidate: DiscoveryCandidate,
        current_watchlist: list[str],
        accepted: list[DiscoveryCandidate],
        sector_exposure: dict[str, int],
    ) -> str | None:
        """Check if candidate passes all filters.

        Returns rejection reason if rejected, None if accepted.
        """
        # Basic filters
        basic_rejection = self._check_basic_filters(candidate, current_watchlist, accepted)
        if basic_rejection:
            return basic_rejection

        # Market quality filters
        quality_rejection = self._check_quality_filters(candidate)
        if quality_rejection:
            return quality_rejection

        # Sector filters
        return self._check_sector_filters(candidate, current_watchlist, accepted, sector_exposure)

    def _check_basic_filters(
        self, candidate: DiscoveryCandidate, current_watchlist: list[str], accepted: list[DiscoveryCandidate]
    ) -> str | None:
        """Check basic filters (size limit, duplicates)."""
        if len(current_watchlist) + len(accepted) >= self.config.max_watchlist_size:
            return f"{candidate.symbol}: watchlist size limit reached"
        if candidate.symbol in current_watchlist:
            return f"{candidate.symbol}: already in watchlist"
        return None

    def _check_quality_filters(self, candidate: DiscoveryCandidate) -> str | None:  # noqa: PLR0911
        """Check market quality filters (cap, volume, price)."""
        market_cap = candidate.metadata.get("market_cap")
        avg_volume = candidate.metadata.get("avg_volume")
        price = candidate.metadata.get("price")

        # Validate all fields present and valid
        if not market_cap or not isinstance(market_cap, (int, float)) or market_cap <= 0:
            return f"{candidate.symbol}: missing or invalid market cap"
        if not avg_volume or not isinstance(avg_volume, (int, float)) or avg_volume <= 0:
            return f"{candidate.symbol}: missing or invalid avg volume"
        if not price or not isinstance(price, (int, float)) or price <= 0:
            return f"{candidate.symbol}: missing or invalid price"

        # Check thresholds
        if market_cap < self.config.min_market_cap:
            min_cap_b = self.config.min_market_cap / 1e9
            return f"{candidate.symbol}: market cap ${market_cap / 1e9:.1f}B < min ${min_cap_b:.1f}B"
        if avg_volume < self.config.min_avg_volume:
            return f"{candidate.symbol}: avg volume {avg_volume:,.0f} < min {self.config.min_avg_volume:,.0f}"

        min_price, max_price = self.config.price_range
        if price < min_price or price > max_price:
            return f"{candidate.symbol}: price ${price:.2f} outside range ${min_price}-${max_price}"

        return None

    def _check_sector_filters(
        self,
        candidate: DiscoveryCandidate,
        current_watchlist: list[str],
        accepted: list[DiscoveryCandidate],
        sector_exposure: dict[str, int],
    ) -> str | None:
        """Check sector-based filters (exclusion, concentration)."""
        # Sector exclusion
        if candidate.sector in self.config.exclude_sectors:
            return f"{candidate.symbol}: sector {candidate.sector} excluded"

        # Sector concentration check
        sector = candidate.sector
        current_sector_count = sector_exposure.get(sector, 0)
        total_count = len(current_watchlist) + len(accepted)

        if total_count > 0:
            projected_concentration = (current_sector_count + 1) / (total_count + 1)
            if projected_concentration > self.config.max_sector_concentration:
                return (
                    f"{candidate.symbol}: sector {sector} concentration "
                    f"{projected_concentration:.1%} > max {self.config.max_sector_concentration:.1%}"
                )
        return None

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

        # Fetch sectors for watchlist symbols
        candidate_sectors = {c.symbol: c.sector for c in candidates}
        for symbol in current_watchlist:
            # Try candidate sectors first, fallback to yfinance fetch
            if symbol in candidate_sectors:
                sector = candidate_sectors[symbol]
            else:
                sector = self._fetch_symbol_sector(symbol)
                if sector is None:
                    logger.warning(f"Could not fetch sector for watchlist symbol {symbol}, skipping")
                    continue
            sector_counts[sector] = sector_counts.get(sector, 0) + 1

        # Fetch sectors for positions (BrokerPosition doesn't have sector field)
        for symbol in current_positions:
            sector = self._fetch_symbol_sector(symbol)
            if sector is None:
                logger.warning(f"Could not fetch sector for position {symbol}, skipping")
                continue
            sector_counts[sector] = sector_counts.get(sector, 0) + 1

        return sector_counts

    def _fetch_symbol_sector(self, symbol: str) -> str | None:
        """Fetch sector for symbol via yfinance.

        Args:
            symbol: Stock symbol

        Returns:
            Sector name or None if unavailable
        """
        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            return ticker.info.get("sector")
        except Exception as e:
            logger.debug(f"Failed to fetch sector for {symbol}: {e}")
            return None

    def __repr__(self) -> str:
        """Return string representation."""
        return f"PortfolioFilterEngine(config={self.config})"
