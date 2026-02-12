"""Deep peer benchmarking analysis for portfolio positions."""

import json
import time as time_mod
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import yfinance as yf
from loguru import logger
from pydantic import BaseModel

from src.cache.historical import HistoricalCache
from src.data.fundamental import FundamentalDataFetcher
from src.data.universe import StockInfo, StockUniverseFetcher


class PeerMetrics(BaseModel):
    """Fundamental metrics for a single peer."""

    symbol: str
    pe_ratio: float | None = None
    peg_ratio: float | None = None
    revenue_growth: float | None = None
    profit_margin: float | None = None
    operating_margin: float | None = None
    dividend_yield: float | None = None
    market_cap: float | None = None
    composite_score: float = 0.0


class PeerAnalysisResult(BaseModel):
    """Peer analysis result for a single position."""

    symbol: str
    sector: str
    peer_count: int
    rank: int
    peers: list[PeerMetrics]
    top_alternative: str | None = None
    swap_recommendation: str | None = None
    analyzed_at: datetime


class DeepPeerAnalysisResult(BaseModel):
    """Aggregate result for all positions analyzed."""

    analyses: list[PeerAnalysisResult]
    total_symbols: int
    total_peers_analyzed: int
    total_duration_seconds: float
    analyzed_at: datetime


# Composite score weights
_WEIGHTS = {
    "pe_inverted": 0.20,
    "peg_inverted": 0.15,
    "revenue_growth": 0.20,
    "profit_margin": 0.20,
    "operating_margin": 0.15,
    "dividend_yield": 0.10,
}


def _safe_float(value: str | float | None) -> float | None:
    """Parse AV overview value to float, returning None for missing/invalid."""
    if value is None or value in {"None", "-"}:
        return None
    try:
        return float(value)
    except ValueError, TypeError:
        return None


@dataclass
class PeerAnalyzerConfig:
    """Configuration for DeepPeerAnalyzer."""

    output_dir: str = "~/.ai-casino/peer-analysis"
    max_peers: int = 10
    rate_limit_sleep: float = 13.0


class DeepPeerAnalyzer:
    """Weekly deep peer analysis comparing positions against sector peers."""

    def __init__(
        self,
        fundamental_fetcher: FundamentalDataFetcher | None = None,
        universe_fetcher: StockUniverseFetcher | None = None,
        historical_cache: HistoricalCache | None = None,
        config: PeerAnalyzerConfig | None = None,
        **deprecated_kwargs: str | int | float | None,
    ) -> None:
        """Initialize deep peer analyzer.

        Args:
            fundamental_fetcher: Alpha Vantage fundamental data fetcher (required for analyze_positions)
            universe_fetcher: Stock universe fetcher (required for analyze_positions)
            historical_cache: Optional cache for dedup
            config: Configuration (uses defaults if not provided)
            **deprecated_kwargs: Deprecated params (output_dir, max_peers, rate_limit_sleep). Use config.
        """
        # Backward compat: construct config from individual params if provided
        output_dir = deprecated_kwargs.get("output_dir")
        max_peers = deprecated_kwargs.get("max_peers")
        rate_limit_sleep = deprecated_kwargs.get("rate_limit_sleep")

        if config is None and output_dir is not None:
            config = PeerAnalyzerConfig(
                output_dir=str(output_dir),
                max_peers=int(max_peers) if max_peers else 10,
                rate_limit_sleep=float(rate_limit_sleep) if rate_limit_sleep else 13.0,
            )

        cfg = config or PeerAnalyzerConfig()
        self._fundamental = fundamental_fetcher
        self._universe = universe_fetcher
        self._output_dir = Path(cfg.output_dir).expanduser()
        self._max_peers = cfg.max_peers
        self._rate_limit_sleep = cfg.rate_limit_sleep
        self._cache = historical_cache
        self._ticker_cache: dict[str, yf.Ticker] = {}
        logger.info(f"Initialized DeepPeerAnalyzer (max_peers={cfg.max_peers})")

    def analyze_positions(self, symbols: list[str]) -> DeepPeerAnalysisResult:
        """Run deep peer analysis for all positions.

        Args:
            symbols: List of position symbols to analyze

        Returns:
            DeepPeerAnalysisResult with all analyses
        """
        if self._fundamental is None or self._universe is None:
            msg = "analyze_positions requires fundamental_fetcher and universe_fetcher"
            raise ValueError(msg)

        start = time_mod.time()
        analyses: list[PeerAnalysisResult] = []
        total_peers = 0

        # Fetch universe once
        universe = self._universe.fetch_combined()

        for symbol in symbols:
            try:
                result = self._analyze_single(symbol, universe.stocks)
                analyses.append(result)
                total_peers += result.peer_count
            except Exception as e:
                logger.opt(exception=True).error(f"Peer analysis failed for {symbol}: {e}")

        duration = time_mod.time() - start
        result = DeepPeerAnalysisResult(
            analyses=analyses,
            total_symbols=len(symbols),
            total_peers_analyzed=total_peers,
            total_duration_seconds=duration,
            analyzed_at=datetime.now(UTC),
        )

        self.persist(result)
        return result

    def _analyze_single(self, symbol: str, universe: list[StockInfo]) -> PeerAnalysisResult:
        """Analyze a single position against its sector peers.

        Args:
            symbol: Stock ticker
            universe: Full stock universe for peer identification

        Returns:
            PeerAnalysisResult with ranking and swap recommendation
        """
        sector = self._get_sector(symbol)
        peer_symbols = self._get_peers(symbol, sector, universe)

        # Fetch metrics for position + peers
        all_symbols = [symbol, *peer_symbols]
        all_metrics: list[PeerMetrics] = []

        for sym in all_symbols:
            metrics = self._fetch_peer_metrics(sym)
            if metrics:
                all_metrics.append(metrics)
            time_mod.sleep(self._rate_limit_sleep)

        return self._rank_peers(symbol, sector, all_metrics)

    def _get_cached_ticker(self, symbol: str) -> yf.Ticker:
        """Get yf.Ticker with session-level caching.

        Args:
            symbol: Stock ticker

        Returns:
            Cached yf.Ticker instance
        """
        if symbol not in self._ticker_cache:
            self._ticker_cache[symbol] = yf.Ticker(symbol)
        return self._ticker_cache[symbol]

    def _get_sector(self, symbol: str) -> str:
        """Get sector for a symbol via yfinance.

        Args:
            symbol: Stock ticker

        Returns:
            Sector name string
        """
        try:
            ticker = self._get_cached_ticker(symbol)
            info = ticker.info
            sector = info.get("sector", "")
            if not sector:
                logger.warning(f"No sector found for {symbol}")
                return "Unknown"
            return sector
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to get sector for {symbol}: {e}")
            return "Unknown"

    def _get_peers(self, symbol: str, sector: str, universe: list[StockInfo]) -> list[str]:
        """Identify same-sector peers, capped by market cap proximity.

        Args:
            symbol: Position symbol (excluded from peers)
            sector: Target sector name
            universe: Full stock universe

        Returns:
            List of peer symbols (max max_peers)
        """
        sector_lower = sector.casefold()
        same_sector = [s for s in universe if s.sector.casefold() == sector_lower and s.symbol != symbol]

        if len(same_sector) <= self._max_peers:
            return [s.symbol for s in same_sector]

        # Cap at max_peers closest by market cap
        # Get position market cap for proximity sorting
        try:
            ticker = self._get_cached_ticker(symbol)
            position_mcap = ticker.info.get("marketCap", 0) or 0
        except Exception:
            position_mcap = 0

        if position_mcap == 0:
            return [s.symbol for s in same_sector[: self._max_peers]]

        # Sort by market cap proximity
        peers_with_mcap: list[tuple[str, float]] = []
        for stock in same_sector:
            try:
                t = self._get_cached_ticker(stock.symbol)
                mcap = t.info.get("marketCap", 0) or 0
                peers_with_mcap.append((stock.symbol, abs(mcap - position_mcap)))
            except Exception:
                peers_with_mcap.append((stock.symbol, float("inf")))

        peers_with_mcap.sort(key=lambda x: x[1])
        return [p[0] for p in peers_with_mcap[: self._max_peers]]

    def _fetch_peer_metrics(self, symbol: str) -> PeerMetrics | None:
        """Fetch fundamental metrics for a single symbol via AV overview.

        Args:
            symbol: Stock ticker

        Returns:
            PeerMetrics or None on failure
        """
        if not self._fundamental:
            return None
        try:
            overview = self._fundamental.fetch_overview(symbol)

            pe = _safe_float(overview.get("PERatio"))
            peg = _safe_float(overview.get("PEGRatio"))
            rev_growth = _safe_float(overview.get("QuarterlyRevenueGrowthYOY"))
            profit_margin = _safe_float(overview.get("ProfitMargin"))
            op_margin = _safe_float(overview.get("OperatingMarginTTM"))
            div_yield = _safe_float(overview.get("DividendYield"))
            market_cap = _safe_float(overview.get("MarketCapitalization"))

            metrics = PeerMetrics(
                symbol=symbol,
                pe_ratio=pe,
                peg_ratio=peg,
                revenue_growth=rev_growth,
                profit_margin=profit_margin,
                operating_margin=op_margin,
                dividend_yield=div_yield,
                market_cap=market_cap,
            )
            metrics.composite_score = self._composite_score(metrics)
            return metrics

        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to fetch metrics for {symbol}: {e}")
            return None

    def _composite_score(self, metrics: PeerMetrics) -> float:
        """Calculate weighted composite score for ranking.

        Lower PE/PEG = better (inverted), higher growth/margins = better.
        Missing values contribute 0.5 (neutral).

        Args:
            metrics: Peer fundamental metrics

        Returns:
            Composite score (0.0-1.0 range, higher = better)
        """
        scores: dict[str, float] = {}

        # PE: lower is better, normalize with inversion
        if metrics.pe_ratio is not None and metrics.pe_ratio > 0:
            scores["pe_inverted"] = max(0.0, min(1.0, 1.0 - (metrics.pe_ratio / 100.0)))
        else:
            scores["pe_inverted"] = 0.5

        # Lower PEG ratio = better value
        if metrics.peg_ratio is not None and metrics.peg_ratio > 0:
            scores["peg_inverted"] = max(0.0, min(1.0, 1.0 - (metrics.peg_ratio / 5.0)))
        else:
            scores["peg_inverted"] = 0.5

        # Revenue growth: higher is better (AV returns decimal like 0.15 for 15%)
        if metrics.revenue_growth is not None:
            scores["revenue_growth"] = max(0.0, min(1.0, (metrics.revenue_growth + 0.5) / 1.0))
        else:
            scores["revenue_growth"] = 0.5

        # Profit margin: higher is better
        if metrics.profit_margin is not None:
            scores["profit_margin"] = max(0.0, min(1.0, (metrics.profit_margin + 0.5) / 1.0))
        else:
            scores["profit_margin"] = 0.5

        # Operating margin: higher is better
        if metrics.operating_margin is not None:
            scores["operating_margin"] = max(0.0, min(1.0, (metrics.operating_margin + 0.5) / 1.0))
        else:
            scores["operating_margin"] = 0.5

        # Dividend yield: higher is better (small bonus)
        if metrics.dividend_yield is not None:
            scores["dividend_yield"] = max(0.0, min(1.0, metrics.dividend_yield / 0.10))
        else:
            scores["dividend_yield"] = 0.5

        total = sum(scores[k] * _WEIGHTS[k] for k in _WEIGHTS)
        return round(total, 4)

    def _rank_peers(self, position_symbol: str, sector: str, peers: list[PeerMetrics]) -> PeerAnalysisResult:
        """Rank position among peers and generate swap recommendation.

        Args:
            position_symbol: The position being analyzed
            sector: Sector name
            peers: All metrics (position + peers) sorted by composite_score desc

        Returns:
            PeerAnalysisResult with ranking
        """
        sorted_peers = sorted(peers, key=lambda p: p.composite_score, reverse=True)
        rank = next(
            (i + 1 for i, p in enumerate(sorted_peers) if p.symbol == position_symbol),
            0,
        )

        top_alternative: str | None = None
        swap_recommendation: str | None = None
        if rank > 0 and sorted_peers and sorted_peers[0].symbol != position_symbol:
            top_alternative = sorted_peers[0].symbol
            swap_recommendation = (
                f"{position_symbol} ranks #{rank} of {len(sorted_peers)} "
                f"in {sector}, consider {top_alternative} (#{1})"
            )

        return PeerAnalysisResult(
            symbol=position_symbol,
            sector=sector,
            peer_count=len(sorted_peers),
            rank=rank,
            peers=sorted_peers,
            top_alternative=top_alternative,
            swap_recommendation=swap_recommendation,
            analyzed_at=datetime.now(UTC),
        )

    def persist(self, result: DeepPeerAnalysisResult) -> Path:
        """Write analysis result to JSON file.

        Args:
            result: Analysis result to persist

        Returns:
            Path to written file
        """
        self._output_dir.mkdir(parents=True, exist_ok=True)
        date_str = datetime.now(UTC).strftime("%Y-%m-%d")
        file_path = self._output_dir / f"{date_str}.json"

        with file_path.open("w") as f:
            json.dump(result.model_dump(mode="json"), f, indent=2, default=str)

        logger.info(f"Persisted peer analysis to {file_path}")
        return file_path

    def load_latest(self, symbol: str) -> PeerAnalysisResult | None:
        """Load most recent analysis result for a symbol.

        Args:
            symbol: Stock ticker

        Returns:
            PeerAnalysisResult or None if no data
        """
        if not self._output_dir.exists():
            return None

        json_files = sorted(self._output_dir.glob("*.json"), reverse=True)
        if not json_files:
            return None

        try:
            with json_files[0].open() as f:
                data = json.load(f)

            result = DeepPeerAnalysisResult.model_validate(data)
            for analysis in result.analyses:
                if analysis.symbol == symbol:
                    return analysis
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to load peer analysis for {symbol}: {e}")

        return None

    def format_context(self, symbol: str) -> str | None:
        """Format peer analysis as context string for trader prompt.

        Args:
            symbol: Stock ticker

        Returns:
            Formatted context string or None if no data
        """
        analysis = self.load_latest(symbol)
        if not analysis:
            return None

        metrics_lines = []
        for i, peer in enumerate(analysis.peers[:5], 1):
            pe_str = f"PE={peer.pe_ratio:.1f}" if peer.pe_ratio is not None else "PE=N/A"
            margin_str = (
                f"margin={peer.profit_margin:.1%}" if peer.profit_margin is not None else "margin=N/A"
            )
            line = f"  {i}. {peer.symbol}: score={peer.composite_score:.3f} {pe_str} {margin_str}"
            metrics_lines.append(line)

        metrics_summary = "\n".join(metrics_lines)
        swap = analysis.swap_recommendation or "Position is top-ranked in sector"

        return (
            f"Sector: {analysis.sector}\n"
            f"Rank: #{analysis.rank} of {analysis.peer_count} peers\n"
            f"{metrics_summary}\n"
            f"{swap}"
        )

    def __repr__(self) -> str:
        """String representation."""
        return f"DeepPeerAnalyzer(max_peers={self._max_peers})"
