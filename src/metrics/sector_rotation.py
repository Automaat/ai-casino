"""Sector rotation analysis for tracking relative strength across GICS sectors."""

from datetime import UTC, datetime
from enum import StrEnum

import yfinance as yf
from loguru import logger
from pydantic import BaseModel

from src.data.comparative import MARKET_INDEX, Sector

# Exclude UNKNOWN (SPY fallback) from sector analysis
SECTOR_ETFS: list[tuple[str, str]] = [(s.name, s.value) for s in Sector if s != Sector.UNKNOWN]

# Trading days approximations
TRADING_DAYS_1W = 5
TRADING_DAYS_1M = 21
TRADING_DAYS_3M = 63

# Composite weighting: 20% 1w + 50% 1m + 30% 3m
WEIGHT_1W = 0.20
WEIGHT_1M = 0.50
WEIGHT_3M = 0.30

# Momentum threshold (percentage points)
MOMENTUM_THRESHOLD = 0.5


class Momentum(StrEnum):
    """Sector momentum classification."""

    ACCELERATING = "ACCELERATING"
    DECELERATING = "DECELERATING"
    NEUTRAL = "NEUTRAL"


class SectorStrength(BaseModel):
    """Relative strength metrics for a single sector."""

    sector: str
    etf: str
    return_1w: float
    return_1m: float
    return_3m: float
    relative_strength: float
    momentum: Momentum
    rank: int


class SectorRotationAnalysis(BaseModel):
    """Complete sector rotation analysis across all GICS sectors."""

    sectors: list[SectorStrength]
    leading_sectors: list[str]
    lagging_sectors: list[str]
    spy_return_1w: float
    spy_return_1m: float
    spy_return_3m: float
    timestamp: datetime


class SectorRotationAnalyzer:
    """Analyzes relative strength across all 11 GICS sectors via ETFs."""

    def __init__(self) -> None:
        """Initialize sector rotation analyzer."""
        logger.info("Initialized SectorRotationAnalyzer")

    def analyze(self) -> SectorRotationAnalysis:
        """Run sector rotation analysis.

        Returns:
            SectorRotationAnalysis with all sectors ranked by relative strength
        """
        logger.info("Starting sector rotation analysis")

        closes = self._fetch_sector_data()

        # Validate SPY data availability
        if MARKET_INDEX not in closes:
            msg = f"Missing market index data for {MARKET_INDEX}; cannot compute sector rotation"
            logger.error(msg)
            raise ValueError(msg)

        spy_close = closes[MARKET_INDEX]
        if len(spy_close) <= TRADING_DAYS_3M:
            msg = (
                f"Insufficient price history for {MARKET_INDEX}: "
                f"needed >{TRADING_DAYS_3M} points, got {len(spy_close)}"
            )
            logger.error(msg)
            raise ValueError(msg)

        spy_return_1w = self._calculate_return(spy_close, TRADING_DAYS_1W)
        spy_return_1m = self._calculate_return(spy_close, TRADING_DAYS_1M)
        spy_return_3m = self._calculate_return(spy_close, TRADING_DAYS_3M)

        strengths: list[SectorStrength] = []
        for sector_name, etf in SECTOR_ETFS:
            if etf not in closes:
                logger.warning(f"Missing data for {etf}, skipping {sector_name}")
                continue

            sector_close = closes[etf]
            rel_1w = self._calculate_relative_return(sector_close, spy_close, TRADING_DAYS_1W)
            rel_1m = self._calculate_relative_return(sector_close, spy_close, TRADING_DAYS_1M)
            rel_3m = self._calculate_relative_return(sector_close, spy_close, TRADING_DAYS_3M)

            composite = self._calculate_composite(rel_1w, rel_1m, rel_3m)
            momentum = self._calculate_momentum(rel_1w, rel_1m)

            strengths.append(
                SectorStrength(
                    sector=sector_name,
                    etf=etf,
                    return_1w=rel_1w,
                    return_1m=rel_1m,
                    return_3m=rel_3m,
                    relative_strength=composite,
                    momentum=momentum,
                    rank=0,
                )
            )

        # Rank by composite relative strength (highest = 1)
        strengths.sort(key=lambda s: s.relative_strength, reverse=True)
        for i, s in enumerate(strengths):
            s.rank = i + 1

        leading = [s.sector for s in strengths[:3]]
        lagging = [s.sector for s in strengths[-3:]]

        logger.info(f"Sector rotation complete: leading={leading}, lagging={lagging}")

        return SectorRotationAnalysis(
            sectors=strengths,
            leading_sectors=leading,
            lagging_sectors=lagging,
            spy_return_1w=spy_return_1w,
            spy_return_1m=spy_return_1m,
            spy_return_3m=spy_return_3m,
            timestamp=datetime.now(UTC),
        )

    def _fetch_sector_data(self) -> dict[str, list[float]]:
        """Fetch 6 months of close prices for all sector ETFs + SPY.

        Returns:
            Dict mapping ticker to list of close prices
        """
        tickers = [etf for _, etf in SECTOR_ETFS] + [MARKET_INDEX]

        logger.info(f"Fetching sector data for {len(tickers)} symbols")
        data = yf.download(tickers, period="6mo", progress=False, group_by="ticker")

        closes: dict[str, list[float]] = {}
        for ticker in tickers:
            try:
                col = data["Close"][ticker].dropna() if len(tickers) > 1 else data["Close"].dropna()
                closes[ticker] = col.tolist()
            except (KeyError, TypeError) as e:
                logger.opt(exception=True).warning(f"Failed to extract close prices for {ticker}: {e}")

        return closes

    def _calculate_return(self, prices: list[float], trading_days: int) -> float:
        """Calculate absolute return over a period.

        Args:
            prices: List of close prices (oldest first)
            trading_days: Number of trading days to look back

        Returns:
            Percentage return
        """
        if len(prices) < trading_days + 1:
            return 0.0
        return (prices[-1] / prices[-(trading_days + 1)] - 1) * 100

    def _calculate_relative_return(
        self, sector_prices: list[float], spy_prices: list[float], trading_days: int
    ) -> float:
        """Calculate sector return relative to SPY.

        Args:
            sector_prices: Sector ETF close prices
            spy_prices: SPY close prices
            trading_days: Number of trading days to look back

        Returns:
            Relative return (sector return minus SPY return) in percentage points
        """
        sector_ret = self._calculate_return(sector_prices, trading_days)
        spy_ret = self._calculate_return(spy_prices, trading_days)
        return sector_ret - spy_ret

    def _calculate_composite(self, return_1w: float, return_1m: float, return_3m: float) -> float:
        """Calculate composite relative strength score.

        Args:
            return_1w: 1-week relative return
            return_1m: 1-month relative return
            return_3m: 3-month relative return

        Returns:
            Weighted composite score
        """
        return WEIGHT_1W * return_1w + WEIGHT_1M * return_1m + WEIGHT_3M * return_3m

    def _calculate_momentum(self, return_1w: float, return_1m: float) -> Momentum:
        """Classify sector momentum based on short vs medium-term strength.

        Args:
            return_1w: 1-week relative return
            return_1m: 1-month relative return

        Returns:
            Momentum classification
        """
        diff = return_1w - return_1m
        if diff > MOMENTUM_THRESHOLD:
            return Momentum.ACCELERATING
        if diff < -MOMENTUM_THRESHOLD:
            return Momentum.DECELERATING
        return Momentum.NEUTRAL

    def format_context(self, analysis: SectorRotationAnalysis) -> str:
        """Format sector rotation analysis as text for trader prompt.

        Args:
            analysis: Sector rotation analysis results

        Returns:
            Formatted string for inclusion in trading prompt
        """
        lines = [
            f"Leading Sectors: {', '.join(analysis.leading_sectors)}",
            f"Lagging Sectors: {', '.join(analysis.lagging_sectors)}",
            "",
        ]

        for s in analysis.sectors:
            lines.append(
                f"  {s.rank}. {s.sector} ({s.etf}): "
                f"1w={s.return_1w:+.2f}% 1m={s.return_1m:+.2f}% 3m={s.return_3m:+.2f}% "
                f"composite={s.relative_strength:+.2f} [{s.momentum.value}]"
            )

        return "\n".join(lines)

    def __repr__(self) -> str:
        """String representation."""
        return "SectorRotationAnalyzer()"
