"""Daemon integration for sector rotation analysis."""

import yfinance as yf
from loguru import logger

from src.metrics.sector_rotation import Momentum, SectorRotationAnalysis, SectorRotationAnalyzer
from src.screening.screener import ScreeningResult


class DaemonSectorRotation:
    """Daemon wrapper for sector rotation analysis with candidate weighting and position flagging."""

    def __init__(self) -> None:
        """Initialize daemon sector rotation."""
        self._analyzer = SectorRotationAnalyzer()
        logger.info("Initialized DaemonSectorRotation")

    def run(self) -> SectorRotationAnalysis:
        """Run sector rotation analysis.

        Returns:
            SectorRotationAnalysis with all sectors ranked
        """
        return self._analyzer.analyze()

    def weight_candidates(
        self,
        candidates: list[ScreeningResult],
        analysis: SectorRotationAnalysis,
        boost_factor: float = 0.15,
    ) -> list[ScreeningResult]:
        """Sort screening candidates by sector-adjusted score.

        Candidates in leading sectors get a boost, lagging sectors get a penalty.
        Returns a new sorted list without mutating the originals.

        Args:
            candidates: Screening results to weight
            analysis: Current sector rotation analysis
            boost_factor: Adjustment factor for leading/lagging sectors

        Returns:
            Candidates sorted by sector-adjusted score (descending)
        """
        # Build sector strength lookup by ETF
        strength_by_etf: dict[str, float] = {}
        for s in analysis.sectors:
            strength_by_etf[s.etf] = s.relative_strength

        leading_set = set(analysis.leading_sectors)
        lagging_set = set(analysis.lagging_sectors)

        # Build sector name lookup from screening results (sector field maps to sector names)
        # ScreeningResult.sector is a display name like "Technology"
        # We need to match against leading/lagging sector names (e.g., "TECHNOLOGY")

        def _adjusted_score(candidate: ScreeningResult) -> float:
            sector_upper = candidate.sector.upper().replace(" ", "_")
            base_score = candidate.score

            if sector_upper in leading_set:
                return base_score * (1 + boost_factor)
            if sector_upper in lagging_set:
                return base_score * (1 - boost_factor)
            return base_score

        return sorted(candidates, key=_adjusted_score, reverse=True)

    def flag_weak_positions(
        self,
        position_symbols: list[str],
        analysis: SectorRotationAnalysis,
    ) -> list[str]:
        """Flag position symbols in weakening or lagging sectors.

        Args:
            position_symbols: List of symbols currently held
            analysis: Current sector rotation analysis

        Returns:
            Symbols that are in decelerating momentum or lagging sectors
        """
        if not position_symbols:
            return []

        # Build lookup sets
        lagging_set = set(analysis.lagging_sectors)
        decelerating_sectors = {s.sector for s in analysis.sectors if s.momentum == Momentum.DECELERATING}
        weak_sectors = lagging_set | decelerating_sectors

        if not weak_sectors:
            return []

        # We need to determine each position's sector - requires yfinance lookup
        flagged: list[str] = []
        for symbol in position_symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                sector = info.get("sector", "")
                if not sector:
                    continue

                # Normalize sector name to match Sector enum names
                sector_upper = sector.upper().replace(" ", "_")

                # Map common yfinance sector names
                sector_map = {
                    "TECHNOLOGY": "TECHNOLOGY",
                    "HEALTHCARE": "HEALTHCARE",
                    "FINANCIAL_SERVICES": "FINANCIALS",
                    "FINANCIALS": "FINANCIALS",
                    "CONSUMER_CYCLICAL": "CONSUMER_DISCRETIONARY",
                    "CONSUMER_DISCRETIONARY": "CONSUMER_DISCRETIONARY",
                    "CONSUMER_DEFENSIVE": "CONSUMER_STAPLES",
                    "CONSUMER_STAPLES": "CONSUMER_STAPLES",
                    "ENERGY": "ENERGY",
                    "INDUSTRIALS": "INDUSTRIALS",
                    "BASIC_MATERIALS": "MATERIALS",
                    "MATERIALS": "MATERIALS",
                    "UTILITIES": "UTILITIES",
                    "REAL_ESTATE": "REAL_ESTATE",
                    "COMMUNICATION_SERVICES": "COMMUNICATION_SERVICES",
                }
                normalized = sector_map.get(sector_upper, sector_upper)

                if normalized in weak_sectors:
                    flagged.append(symbol)
                    logger.info(f"Flagged {symbol} in weak sector: {normalized}")

            except Exception as e:
                logger.warning(f"Failed to check sector for {symbol}: {e}")

        return flagged

    def __repr__(self) -> str:
        """String representation."""
        return "DaemonSectorRotation()"
