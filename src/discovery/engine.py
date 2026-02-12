"""Stock discovery orchestration engine."""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pandas as pd
import yfinance as yf
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from src.data.market import MarketDataFetcher
from src.data.universe import StockUniverseFetcher
from src.discovery.filters import PortfolioFilterConfig, PortfolioFilterEngine
from src.discovery.models import DiscoveryCandidate, DiscoveryResult, DiscoverySource
from src.discovery.scoring import MultiFactorScorer, ScoringWeights
from src.discovery.triggers import TriggerDetector
from src.screening.screener import ScreeningCriteria, StockScreener


@dataclass
class OptionalServices:
    """Optional external services for discovery engine."""

    reddit_fetcher: object | None = None
    earnings_fetcher: object | None = None
    news_fetcher: object | None = None
    broker: object | None = None


@dataclass
class CoreDependencies:
    """Core dependencies for stock discovery engine."""

    screener: StockScreener
    market_fetcher: MarketDataFetcher
    universe_fetcher: StockUniverseFetcher
    trigger_detector: TriggerDetector


class DiscoveryEngineConfig(BaseModel):
    """Configuration for stock discovery engine."""

    # Source enablement
    enable_technical_screening: bool = True
    enable_reddit_trending: bool = False
    enable_earnings_calendar: bool = True
    enable_sector_rotation: bool = True
    enable_volume_spikes: bool = False
    enable_price_gaps: bool = False
    enable_news_trending: bool = False

    # Screening config
    screening_criteria: list[str] = Field(default_factory=lambda: ["momentum"])
    screening_universe: str = "COMBINED"
    screening_top_n: int = 20

    # Reddit config
    reddit_min_mentions: int = 5
    reddit_min_upvote_ratio: float = 0.75

    # Earnings config
    earnings_lookahead_days: int = 7

    # Trigger thresholds
    volume_spike_threshold: float = 2.0
    price_gap_threshold: float = 5.0

    # Scoring
    scoring_weights: ScoringWeights = Field(default_factory=ScoringWeights)
    max_discovered_per_cycle: int = 5
    min_composite_score: float = 0.60
    max_watchlist_size: int = 50

    # Portfolio filters
    portfolio_filters: PortfolioFilterConfig = Field(default_factory=PortfolioFilterConfig)

    # Lifecycle
    candidate_ttl_days: int = 7
    auto_remove_on_signal: bool = False

    # Tracking
    track_outcomes: bool = True
    outcome_lookback_days: int = 90

    model_config = ConfigDict(arbitrary_types_allowed=True)


class StockDiscoveryEngine:
    """Orchestrate multi-source stock discovery with intelligent filtering."""

    def __init__(
        self,
        deps: CoreDependencies,
        config: DiscoveryEngineConfig,
        services: OptionalServices | None = None,
    ) -> None:
        """Initialize stock discovery engine with dependencies."""
        self.screener = deps.screener
        self.market_fetcher = deps.market_fetcher
        self.universe_fetcher = deps.universe_fetcher
        self.trigger_detector = deps.trigger_detector
        self.config = config

        # Unpack optional services
        services = services or OptionalServices()
        self.reddit_fetcher = services.reddit_fetcher
        self.earnings_fetcher = services.earnings_fetcher
        self.news_fetcher = services.news_fetcher
        self.broker = services.broker

        self.scorer = MultiFactorScorer(config.scoring_weights)
        self.filter_engine = PortfolioFilterEngine(config.portfolio_filters)

        logger.info(f"Initialized StockDiscoveryEngine with {config}")

    async def discover(
        self,
        current_watchlist: list[str],
        current_positions: dict[str, object],
        sector_context: object | None = None,
    ) -> DiscoveryResult:
        """Run multi-source discovery with intelligent filtering.

        Args:
            current_watchlist: Current watchlist symbols
            current_positions: Current portfolio positions
            sector_context: Optional sector rotation context

        Returns:
            DiscoveryResult with ranked candidates
        """
        logger.info("Starting stock discovery")
        discovered_at = datetime.now(UTC)

        # Fetch from all enabled sources in parallel
        all_candidates: dict[str, DiscoveryCandidate] = {}
        source_breakdown: dict[str, int] = {}

        # Technical screening
        if self.config.enable_technical_screening:
            candidates = await self._fetch_technical_candidates()
            self._merge_candidates(all_candidates, candidates, DiscoverySource.TECHNICAL_SCREENING)
            source_breakdown["technical_screening"] = len(candidates)

        # Reddit trending (if enabled and fetcher available)
        if self.config.enable_reddit_trending and self.reddit_fetcher:
            candidates = await self._fetch_reddit_candidates()
            self._merge_candidates(all_candidates, candidates, DiscoverySource.REDDIT_TRENDING)
            source_breakdown["reddit_trending"] = len(candidates)

        # Earnings calendar (if enabled and fetcher available)
        if self.config.enable_earnings_calendar and self.earnings_fetcher:
            candidates = await self._fetch_earnings_candidates()
            self._merge_candidates(all_candidates, candidates, DiscoverySource.EARNINGS_UPCOMING)
            source_breakdown["earnings_upcoming"] = len(candidates)

        # Sector rotation (if enabled and context available)
        if self.config.enable_sector_rotation and sector_context:
            candidates = await self._fetch_sector_rotation_candidates(sector_context)
            self._merge_candidates(all_candidates, candidates, DiscoverySource.SECTOR_ROTATION)
            source_breakdown["sector_rotation"] = len(candidates)

        # Volume spikes (if enabled)
        if self.config.enable_volume_spikes:
            candidates = await self._fetch_volume_spike_candidates()
            self._merge_candidates(all_candidates, candidates, DiscoverySource.VOLUME_SPIKE)
            source_breakdown["volume_spike"] = len(candidates)

        # Price gaps (if enabled)
        if self.config.enable_price_gaps:
            candidates = await self._fetch_price_gap_candidates()
            self._merge_candidates(all_candidates, candidates, DiscoverySource.PRICE_GAP)
            source_breakdown["price_gap"] = len(candidates)

        # News trending (if enabled and fetcher available)
        if self.config.enable_news_trending and self.news_fetcher:
            candidates = await self._fetch_news_trending_candidates()
            self._merge_candidates(all_candidates, candidates, DiscoverySource.NEWS_TRENDING)
            source_breakdown["news_trending"] = len(candidates)

        total_discovered = len(all_candidates)
        logger.info(f"Discovered {total_discovered} unique candidates from {len(source_breakdown)} sources")

        # Score each candidate
        for candidate in all_candidates.values():
            # Boost score for multi-source agreement
            source_boost = 1.0 + (len(candidate.sources) - 1) * 0.1  # 10% per additional source
            base_score = self.scorer.score_candidate(candidate)
            candidate.composite_score = min(base_score * source_boost, 1.0)

            # Set TTL
            candidate.ttl_expires_at = discovered_at + timedelta(days=self.config.candidate_ttl_days)

        # Filter by minimum score
        scored_candidates = [
            c for c in all_candidates.values() if c.composite_score >= self.config.min_composite_score
        ]

        # Apply portfolio filters
        filtered_candidates, _ = await self.filter_engine.filter_candidates(
            scored_candidates, current_positions, current_watchlist
        )

        # Rank by composite score
        ranked_candidates = sorted(filtered_candidates, key=lambda c: c.composite_score, reverse=True)

        filtered_count = total_discovered - len(ranked_candidates)

        logger.info(
            f"Discovery complete: {len(ranked_candidates)} candidates after scoring/filtering "
            f"({filtered_count} filtered)"
        )

        return DiscoveryResult(
            candidates=ranked_candidates,
            total_discovered=total_discovered,
            filtered_count=filtered_count,
            discovered_at=discovered_at,
            source_breakdown=source_breakdown,
        )

    def _merge_candidates(
        self,
        all_candidates: dict[str, DiscoveryCandidate],
        new_candidates: list[DiscoveryCandidate],
        source: DiscoverySource,
    ) -> None:
        """Merge new candidates into aggregate dict.

        Args:
            all_candidates: Aggregate candidate dict
            new_candidates: New candidates to merge
            source: Discovery source
        """
        for candidate in new_candidates:
            if candidate.symbol in all_candidates:
                # Already exists - add source
                existing = all_candidates[candidate.symbol]
                if source not in existing.sources:
                    existing.sources.append(source)
                # Merge metadata
                existing.metadata.update(candidate.metadata)
            else:
                # New candidate
                candidate.sources = [source]
                all_candidates[candidate.symbol] = candidate

    async def _fetch_technical_candidates(self) -> list[DiscoveryCandidate]:
        """Fetch candidates from technical screening.

        Returns:
            List of discovery candidates
        """
        logger.debug("Fetching technical screening candidates")
        candidates: list[DiscoveryCandidate] = []

        try:
            for criteria_name in self.config.screening_criteria:
                try:
                    criteria = ScreeningCriteria[criteria_name.upper()]
                except KeyError:
                    logger.opt(exception=True).warning(
                        f"Invalid screening criteria: {criteria_name}, skipping"
                    )
                    continue
                result = self.screener.screen(
                    criteria=criteria,
                    universe=self.config.screening_universe,
                    top_n=self.config.screening_top_n,
                )

                for screening_result in result.results:
                    # Fetch stock info
                    stock_info = self._get_stock_info(screening_result.symbol)

                    candidate = DiscoveryCandidate(
                        symbol=screening_result.symbol,
                        name=str(stock_info.get("name", screening_result.symbol)),
                        sector=str(stock_info.get("sector", "Unknown")),
                        sources=[],  # Will be set by _merge_candidates
                        composite_score=0.0,  # Will be calculated later
                        discovery_timestamp=datetime.now(UTC),
                        metadata={
                            "technical_score": screening_result.score,
                            "screening_criteria": criteria_name,
                            **stock_info,
                        },
                        ttl_expires_at=datetime.now(UTC),  # Will be set later
                    )
                    candidates.append(candidate)

            logger.info(f"Technical screening: {len(candidates)} candidates")
        except Exception as e:
            logger.opt(exception=True).error(f"Technical screening failed: {e}", exc_info=True)

        return candidates

    async def _fetch_reddit_candidates(self) -> list[DiscoveryCandidate]:
        """Fetch candidates from Reddit trending.

        Returns:
            List of discovery candidates
        """
        logger.debug("Fetching Reddit trending candidates")
        candidates: list[DiscoveryCandidate] = []

        # TODO: Implement Reddit trending integration
        logger.warning("Reddit trending not implemented yet")

        return candidates

    async def _fetch_earnings_candidates(self) -> list[DiscoveryCandidate]:
        """Fetch candidates with upcoming earnings.

        Returns:
            List of discovery candidates
        """
        logger.debug("Fetching earnings calendar candidates")
        candidates: list[DiscoveryCandidate] = []

        # TODO: Implement earnings calendar integration
        logger.warning("Earnings calendar not implemented yet")

        return candidates

    async def _fetch_sector_rotation_candidates(self, _sector_context: object) -> list[DiscoveryCandidate]:
        """Fetch candidates from leading sectors.

        Args:
            sector_context: Sector rotation context

        Returns:
            List of discovery candidates
        """
        logger.debug("Fetching sector rotation candidates")
        candidates: list[DiscoveryCandidate] = []

        # TODO: Implement sector rotation integration
        logger.warning("Sector rotation not implemented yet")

        return candidates

    async def _fetch_volume_spike_candidates(self) -> list[DiscoveryCandidate]:
        """Fetch candidates with volume spikes.

        Returns:
            List of discovery candidates
        """
        logger.debug("Fetching volume spike candidates")
        candidates: list[DiscoveryCandidate] = []

        try:
            # Get universe
            universe = self._get_discovery_universe()

            # Detect volume spikes
            spike_symbols = self.trigger_detector.detect_volume_spikes(universe)

            for symbol in spike_symbols:
                stock_info = self._get_stock_info(symbol)
                candidate = DiscoveryCandidate(
                    symbol=symbol,
                    name=str(stock_info.get("name", symbol)),
                    sector=str(stock_info.get("sector", "Unknown")),
                    sources=[],
                    composite_score=0.0,
                    discovery_timestamp=datetime.now(UTC),
                    metadata={**stock_info, "trigger": "volume_spike"},
                    ttl_expires_at=datetime.now(UTC),
                )
                candidates.append(candidate)

            logger.info(f"Volume spike detection: {len(candidates)} candidates")
        except Exception as e:
            logger.opt(exception=True).error(f"Volume spike detection failed: {e}", exc_info=True)

        return candidates

    async def _fetch_price_gap_candidates(self) -> list[DiscoveryCandidate]:
        """Fetch candidates with price gaps.

        Returns:
            List of discovery candidates
        """
        logger.debug("Fetching price gap candidates")
        candidates: list[DiscoveryCandidate] = []

        try:
            # Get universe
            universe = self._get_discovery_universe()

            # Detect price gaps
            gap_symbols = self.trigger_detector.detect_price_gaps(universe)

            for symbol in gap_symbols:
                stock_info = self._get_stock_info(symbol)
                candidate = DiscoveryCandidate(
                    symbol=symbol,
                    name=str(stock_info.get("name", symbol)),
                    sector=str(stock_info.get("sector", "Unknown")),
                    sources=[],
                    composite_score=0.0,
                    discovery_timestamp=datetime.now(UTC),
                    metadata={**stock_info, "trigger": "price_gap"},
                    ttl_expires_at=datetime.now(UTC),
                )
                candidates.append(candidate)

            logger.info(f"Price gap detection: {len(candidates)} candidates")
        except Exception as e:
            logger.opt(exception=True).error(f"Price gap detection failed: {e}", exc_info=True)

        return candidates

    async def _fetch_news_trending_candidates(self) -> list[DiscoveryCandidate]:
        """Fetch candidates from trending news.

        Returns:
            List of discovery candidates
        """
        logger.debug("Fetching news trending candidates")
        candidates: list[DiscoveryCandidate] = []

        # TODO: Implement news trending integration
        logger.warning("News trending not implemented yet")

        return candidates

    def _get_discovery_universe(self) -> list[str]:
        """Get universe for discovery (SP500 + NASDAQ100).

        Returns:
            List of symbols
        """
        universe_type = self.config.screening_universe.upper()

        if universe_type == "SP500":
            universe = self.universe_fetcher.fetch_sp500()
            return [stock.symbol for stock in universe.stocks]
        if universe_type == "NASDAQ100":
            universe = self.universe_fetcher.fetch_nasdaq100()
            return [stock.symbol for stock in universe.stocks]
        if universe_type == "RUSSELL3000":
            universe = self.universe_fetcher.fetch_russell3000()
            return [stock.symbol for stock in universe.stocks]
        if universe_type == "US_LIQUID":
            # US_LIQUID requires liquidity_filters from config, which are not available
            # in the discovery engine. Failing fast here avoids silently using a different
            # universe than configured and returning low-quality/illiquid stocks.
            msg = (
                "Discovery engine does not support 'US_LIQUID' screening_universe. "
                "Please use a supported universe (e.g. SP500, NASDAQ100, RUSSELL3000) "
                "or configure liquidity filtering in a component that has access to "
                "liquidity_filters."
            )
            raise ValueError(msg)
        # Fallback to combined universe for any other value
        universe = self.universe_fetcher.fetch_combined()
        return [stock.symbol for stock in universe.stocks]

    def _fetch_stock_metadata(self, symbol: str) -> dict[str, object]:
        """Fetch stock metadata from yfinance Ticker.info.

        Args:
            symbol: Stock symbol

        Returns:
            Dict with name, sector, market_cap
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            return {
                "name": info.get("shortName", symbol),
                "sector": info.get("sector", "Unknown"),
                "market_cap": info.get("marketCap", 0),
            }
        except Exception as e:
            logger.debug(f"Failed to fetch metadata for {symbol}: {e}")
            return {
                "name": symbol,
                "sector": "Unknown",
                "market_cap": 0,
            }

    def _get_stock_info(self, symbol: str, cached_ohlcv: pd.DataFrame | None = None) -> dict[str, object]:
        """Fetch stock metadata (name, sector, market cap, etc).

        Args:
            symbol: Stock symbol
            cached_ohlcv: Optional pre-fetched OHLCV data to avoid duplicate API calls

        Returns:
            Dict with stock metadata
        """
        try:
            # Use cached OHLCV if available, otherwise fetch
            if cached_ohlcv is not None:
                df = cached_ohlcv
            else:
                market_data = self.market_fetcher.fetch_daily(symbol, period_days=30)
                df = market_data.data

            if df.empty:
                return {}

            # Calculate basic metrics
            latest = df.iloc[-1]
            price = float(latest["Close"])
            avg_volume = float(df["Volume"].mean())

            # Calculate ATR
            df.ta.atr(length=14, append=True)  # type: ignore[attr-defined]
            atr = (
                float(df["ATR_14"].iloc[-1])
                if "ATR_14" in df.columns and not df["ATR_14"].isna().all()
                else 0.0
            )
            atr_ratio = atr / price if price > 0 else 0.0

            # Fetch real metadata from yfinance
            metadata = self._fetch_stock_metadata(symbol)

            info: dict[str, object] = {
                "price": price,
                "avg_volume": avg_volume,
                "atr_ratio": atr_ratio,
                **metadata,  # Real name, sector, market_cap
            }
            return info

        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to fetch stock info for {symbol}: {e}")
            return {}

    def __repr__(self) -> str:
        """Return string representation."""
        enabled_sources = [s for s in dir(self.config) if s.startswith("enable_") and getattr(self.config, s)]
        return f"StockDiscoveryEngine(sources={len(enabled_sources)})"
