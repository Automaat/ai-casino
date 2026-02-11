"""Multi-factor scoring for discovery candidates."""

from loguru import logger
from pydantic import BaseModel

from src.discovery.models import DiscoveryCandidate

# Market cap thresholds (in dollars)
MARKET_CAP_SMALL = 1e9  # $1B
MARKET_CAP_LARGE = 10e9  # $10B

# Price range thresholds (in dollars)
PRICE_MIN = 10.0
PRICE_MAX = 500.0

# Earnings window (in days)
EARNINGS_SWEET_SPOT_MIN = 3
EARNINGS_SWEET_SPOT_MAX = 7
EARNINGS_LOOKBACK_MAX = 14

# Volatility thresholds (ATR percentage)
ATR_MIN = 2.0
ATR_MAX = 5.0

# Data quality thresholds
MIN_DATA_POINTS_VOLUME = 20
MIN_DATA_POINTS_GAP = 2
MIN_DATA_POINTS_ATR = 35


class ScoringWeights(BaseModel):
    """Weights for multi-factor scoring."""

    technical_weight: float = 0.35  # Screening score
    liquidity_weight: float = 0.25  # Volume, market cap
    timing_weight: float = 0.20  # Earnings, sector momentum
    social_weight: float = 0.15  # Reddit, news mentions
    volatility_weight: float = 0.05  # ATR-based scoring

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"ScoringWeights(tech={self.technical_weight}, liq={self.liquidity_weight}, "
            f"timing={self.timing_weight}, social={self.social_weight}, vol={self.volatility_weight})"
        )


class MultiFactorScorer:
    """Score discovery candidates using multiple factors."""

    def __init__(self, weights: ScoringWeights | None = None) -> None:
        """Initialize multi-factor scorer."""
        self.weights = weights or ScoringWeights()
        logger.info(f"Initialized MultiFactorScorer with {self.weights}")

    def score_candidate(self, candidate: DiscoveryCandidate) -> float:
        """Calculate composite score for candidate.

        Args:
            candidate: Discovery candidate to score

        Returns:
            Composite score 0-1
        """
        metadata = candidate.metadata
        scores: dict[str, float] = {}

        # Technical score (from screening or manual assignment)
        technical_score_raw = metadata.get("technical_score", 0.5)
        technical_score = float(technical_score_raw) if isinstance(technical_score_raw, (int, float)) else 0.5
        scores["technical"] = min(max(technical_score, 0.0), 1.0)

        # Liquidity score (volume and market cap)
        liquidity_score = self._score_liquidity(metadata)
        scores["liquidity"] = liquidity_score

        # Timing score (earnings proximity, sector momentum)
        timing_score = self._score_timing(metadata)
        scores["timing"] = timing_score

        # Social score (Reddit mentions, news trending)
        social_score = self._score_social(metadata)
        scores["social"] = social_score

        # Volatility score (ATR ratio - prefer moderate volatility)
        volatility_score = self._score_volatility(metadata)
        scores["volatility"] = volatility_score

        # Calculate weighted composite
        composite = (
            scores["technical"] * self.weights.technical_weight
            + scores["liquidity"] * self.weights.liquidity_weight
            + scores["timing"] * self.weights.timing_weight
            + scores["social"] * self.weights.social_weight
            + scores["volatility"] * self.weights.volatility_weight
        )

        logger.debug(
            f"{candidate.symbol}: technical={scores['technical']:.2f}, "
            f"liquidity={scores['liquidity']:.2f}, timing={scores['timing']:.2f}, "
            f"social={scores['social']:.2f}, volatility={scores['volatility']:.2f} "
            f"=> composite={composite:.2f}"
        )

        # Store individual scores in candidate
        candidate.source_scores = scores

        return min(max(composite, 0.0), 1.0)

    def _score_liquidity(self, metadata: dict[str, object]) -> float:
        """Score based on volume and market cap.

        Args:
            metadata: Candidate metadata with avg_volume, market_cap, price

        Returns:
            Liquidity score 0-1
        """
        score = 0.0

        # Volume component (50% of liquidity score)
        avg_volume = metadata.get("avg_volume")
        if avg_volume and isinstance(avg_volume, (int, float)):
            # Normalize to 1M shares as baseline, cap at 5M
            volume_score = min(float(avg_volume) / 5_000_000, 1.0)
            score += volume_score * 0.5

        # Market cap component (30% of liquidity score)
        market_cap = metadata.get("market_cap")
        if market_cap and isinstance(market_cap, (int, float)):
            # Normalize: $1B = 0.5, $10B = 1.0
            if market_cap >= MARKET_CAP_LARGE:
                score += 0.3
            elif market_cap >= MARKET_CAP_SMALL:
                mc_score = (float(market_cap) - MARKET_CAP_SMALL) / (MARKET_CAP_LARGE - MARKET_CAP_SMALL)
                score += mc_score * 0.3

        # Price component (20% - prefer $10-500 range)
        price = metadata.get("price")
        if price and isinstance(price, (int, float)) and PRICE_MIN <= price <= PRICE_MAX:
            score += 0.2

        return min(score, 1.0)

    def _score_timing(self, metadata: dict[str, object]) -> float:
        """Score based on earnings proximity and sector momentum.

        Args:
            metadata: Candidate metadata with days_to_earnings, sector_momentum

        Returns:
            Timing score 0-1
        """
        score = 0.0

        # Earnings proximity (60% of timing score)
        days_to_earnings = metadata.get("days_to_earnings")
        if days_to_earnings is not None and isinstance(days_to_earnings, (int, float)):
            # Prefer 3-7 days window (sweet spot for volatility)
            if EARNINGS_SWEET_SPOT_MIN <= days_to_earnings <= EARNINGS_SWEET_SPOT_MAX:
                score += 0.6
            elif 0 <= days_to_earnings <= EARNINGS_LOOKBACK_MAX:
                # Linear decay outside sweet spot
                score += 0.6 * (1 - abs(days_to_earnings - 5) / 10)

        # Sector momentum (40% of timing score)
        sector_momentum = metadata.get("sector_momentum")
        if sector_momentum and isinstance(sector_momentum, (int, float)):
            # Sector momentum already normalized 0-1
            score += min(max(float(sector_momentum), 0.0), 1.0) * 0.4

        return min(score, 1.0)

    def _score_social(self, metadata: dict[str, object]) -> float:
        """Score based on Reddit mentions and news trending.

        Args:
            metadata: Candidate metadata with reddit_mentions, news_article_count

        Returns:
            Social score 0-1
        """
        score = 0.0

        # Reddit mentions (60% of social score)
        reddit_mentions = metadata.get("reddit_mentions")
        if reddit_mentions and isinstance(reddit_mentions, (int, float)):
            # Normalize: 5 mentions = 0.5, 20+ mentions = 1.0
            mention_score = min(float(reddit_mentions) / 20.0, 1.0)
            score += mention_score * 0.6

        # News article count (40% of social score)
        news_count = metadata.get("news_article_count")
        if news_count and isinstance(news_count, (int, float)):
            # Normalize: 5 articles = 0.5, 15+ articles = 1.0
            news_score = min(float(news_count) / 15.0, 1.0)
            score += news_score * 0.4

        return min(score, 1.0)

    def _score_volatility(self, metadata: dict[str, object]) -> float:
        """Score based on ATR ratio (prefer moderate volatility).

        Args:
            metadata: Candidate metadata with atr_ratio (ATR/price)

        Returns:
            Volatility score 0-1
        """
        atr_ratio = metadata.get("atr_ratio")
        if not atr_ratio or not isinstance(atr_ratio, (int, float)):
            return 0.5  # Neutral score if missing

        # Prefer 2-5% daily range (moderate volatility)
        atr_pct = float(atr_ratio) * 100

        if ATR_MIN <= atr_pct <= ATR_MAX:
            return 1.0
        if atr_pct < ATR_MIN:
            # Too low volatility - linear penalty
            return max(atr_pct / ATR_MIN, 0.0)
        # Too high volatility - exponential penalty
        return max(1.0 - (atr_pct - ATR_MAX) / 10.0, 0.0)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"MultiFactorScorer(weights={self.weights})"
