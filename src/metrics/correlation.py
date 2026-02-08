"""Portfolio correlation analysis and audit."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yfinance as yf
from loguru import logger
from pydantic import BaseModel, Field

from src.data.broker import BrokerPosition
from src.data.market import MarketDataFetcher
from src.screening.screener import ScreeningResult

# Constants
MIN_POSITIONS_FOR_CORRELATION = 2
MIN_DATA_POINTS = 30
MAX_SUBSTITUTION_SUGGESTIONS = 5


class CorrelationPair(BaseModel):
    """Highly correlated position pair."""

    symbol_a: str
    symbol_b: str
    correlation: float
    sector_a: str
    sector_b: str
    same_sector: bool


class SubstitutionSuggestion(BaseModel):
    """Suggestion to replace correlated position."""

    symbol_to_replace: str
    reason: str
    alternatives: list[str] = Field(description="Top 3 low-correlation candidates")
    alternative_correlations: list[float]
    sector: str


class CorrelationAuditResult(BaseModel):
    """Portfolio correlation audit result."""

    audit_date: datetime
    num_positions: int
    correlation_matrix: dict[str, dict[str, float]]
    highly_correlated_pairs: list[CorrelationPair]
    diversification_ratio: float
    max_correlation: float
    avg_correlation: float
    substitution_suggestions: list[SubstitutionSuggestion]
    warnings: list[str]
    lookback_days: int


class CorrelationAuditor:
    """Audit portfolio correlation and suggest diversification improvements."""

    def __init__(
        self,
        market_fetcher: MarketDataFetcher | None = None,
        correlation_threshold: float = 0.8,
        lookback_days: int = 90,
        output_dir: str = "~/.ai-casino/correlation-audits",
    ) -> None:
        """Initialize correlation auditor.

        Args:
            market_fetcher: Market data fetcher (optional for load_latest)
            correlation_threshold: Minimum correlation to flag (0.5-0.95)
            lookback_days: Historical period for correlation (30-180 days)
            output_dir: Directory for persisting audit results
        """
        self.market_fetcher = market_fetcher
        self.correlation_threshold = correlation_threshold
        self.lookback_days = lookback_days
        self.output_dir = Path(output_dir).expanduser()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._returns_cache: dict[str, pd.Series] = {}
        self._ticker_cache: dict[str, yf.Ticker] = {}

        logger.info(
            f"Initialized CorrelationAuditor (threshold={correlation_threshold:.2f}, "
            f"lookback={lookback_days}d)"
        )

    def audit(
        self,
        positions: dict[str, BrokerPosition],
        screening_results: list[ScreeningResult] | None = None,
    ) -> CorrelationAuditResult:
        """Run portfolio correlation audit.

        Args:
            positions: Current broker positions
            screening_results: Latest screening results for substitution suggestions

        Returns:
            Correlation audit result with pairs, diversification ratio, substitutions
        """
        if self.market_fetcher is None:
            msg = "market_fetcher required for audit()"
            raise ValueError(msg)

        warnings = []
        symbols = list(positions.keys())

        if len(symbols) < MIN_POSITIONS_FOR_CORRELATION:
            logger.warning(f"Insufficient positions for correlation audit: {len(symbols)}")
            return CorrelationAuditResult(
                audit_date=datetime.now(UTC),
                num_positions=len(symbols),
                correlation_matrix={},
                highly_correlated_pairs=[],
                diversification_ratio=1.0,
                max_correlation=0.0,
                avg_correlation=0.0,
                substitution_suggestions=[],
                warnings=["Insufficient positions (<2)"],
                lookback_days=self.lookback_days,
            )

        # Fetch aligned returns
        logger.info(f"Fetching returns for {len(symbols)} positions over {self.lookback_days} days")
        self._returns_cache.clear()  # Clear cache at start of audit
        returns_df = self._fetch_position_returns(symbols, warnings)

        # Populate cache with portfolio returns for substitution calculations
        for symbol in returns_df.columns:
            self._returns_cache[symbol] = returns_df[symbol]

        if returns_df.empty:
            warnings.append("No return data available")
            return CorrelationAuditResult(
                audit_date=datetime.now(UTC),
                num_positions=len(symbols),
                correlation_matrix={},
                highly_correlated_pairs=[],
                diversification_ratio=1.0,
                max_correlation=0.0,
                avg_correlation=0.0,
                substitution_suggestions=[],
                warnings=warnings,
                lookback_days=self.lookback_days,
            )

        if len(returns_df) < MIN_DATA_POINTS:
            warnings.append(f"Limited data points: {len(returns_df)}")

        # Compute correlation matrix
        corr_matrix_dict = self._compute_correlation_matrix(returns_df)

        # Identify highly correlated pairs
        correlated_pairs = self._identify_correlated_pairs(corr_matrix_dict, warnings)

        # Calculate diversification ratio
        diversification_ratio = self._calculate_diversification_ratio(returns_df, positions)

        # Calculate statistics
        correlations = []
        for sym_a, row in corr_matrix_dict.items():
            for sym_b, corr in row.items():
                if sym_a < sym_b:  # Avoid duplicates
                    correlations.append(corr)

        finite_correlations = [c for c in correlations if pd.notna(c)]
        max_corr = max(finite_correlations) if finite_correlations else 0.0
        avg_corr = sum(finite_correlations) / len(finite_correlations) if finite_correlations else 0.0

        # Generate substitution suggestions
        substitutions = self._generate_substitutions(
            correlated_pairs, screening_results, corr_matrix_dict, warnings
        )

        result = CorrelationAuditResult(
            audit_date=datetime.now(UTC),
            num_positions=len(symbols),
            correlation_matrix=corr_matrix_dict,
            highly_correlated_pairs=correlated_pairs,
            diversification_ratio=diversification_ratio,
            max_correlation=max_corr,
            avg_correlation=avg_corr,
            substitution_suggestions=substitutions,
            warnings=warnings,
            lookback_days=self.lookback_days,
        )

        # Persist result
        self.persist(result)

        logger.info(
            f"Correlation audit complete: {len(correlated_pairs)} pairs, "
            f"diversification={diversification_ratio:.3f}, {len(substitutions)} suggestions"
        )

        return result

    def load_latest(self) -> CorrelationAuditResult | None:
        """Load most recent audit result.

        Returns:
            Latest audit result or None if no results exist
        """
        audit_files = sorted(self.output_dir.glob("correlation-audit-*.json"), reverse=True)
        if not audit_files:
            return None

        latest = audit_files[0]
        logger.debug(f"Loading latest audit: {latest.name}")

        with latest.open() as f:
            data = json.load(f)

        return CorrelationAuditResult(**data)

    def persist(self, result: CorrelationAuditResult) -> Path:
        """Persist audit result to disk.

        Args:
            result: Audit result to save

        Returns:
            Path to saved file
        """
        timestamp = result.audit_date.strftime("%Y%m%d-%H%M%S")
        filename = f"correlation-audit-{timestamp}.json"
        filepath = self.output_dir / filename

        with filepath.open("w") as f:
            json.dump(result.model_dump(mode="json"), f, indent=2)

        logger.info(f"Persisted correlation audit: {filepath}")
        return filepath

    def _fetch_position_returns(self, symbols: list[str], warnings: list[str]) -> pd.DataFrame:
        """Fetch aligned daily returns for positions.

        Args:
            symbols: Position symbols
            warnings: List to append warnings to

        Returns:
            DataFrame with aligned returns (symbols as columns, dates as index)
        """
        if not self.market_fetcher:
            warnings.append("Market fetcher not available")
            return pd.DataFrame()

        returns_data = {}

        for symbol in symbols:
            try:
                market_data = self.market_fetcher.fetch_daily(symbol, period_days=self.lookback_days)
                df = market_data.data

                if df.empty or "Close" not in df.columns:
                    warnings.append(f"No Close data for {symbol}")
                    continue

                # Calculate daily returns
                returns = df["Close"].pct_change().dropna()
                returns_data[symbol] = returns

            except Exception as e:
                logger.warning(f"Failed to fetch {symbol}: {e}")
                warnings.append(f"Failed to fetch {symbol}")

        if not returns_data:
            return pd.DataFrame()

        # Align all series to common dates
        returns_df = pd.DataFrame(returns_data)
        returns_df = returns_df.dropna()  # Drop rows with any missing data

        logger.debug(f"Aligned returns: {len(returns_df)} days x {len(returns_df.columns)} symbols")
        return returns_df

    def _compute_correlation_matrix(self, returns_df: pd.DataFrame) -> dict[str, dict[str, float]]:
        """Compute pairwise correlation matrix.

        Args:
            returns_df: Aligned returns DataFrame

        Returns:
            Nested dict {symbol_a: {symbol_b: correlation}}
        """
        corr_matrix = returns_df.corr()

        # Convert to nested dict format
        result = {}
        for sym_a in corr_matrix.index:
            result[sym_a] = {}
            for sym_b in corr_matrix.columns:
                result[sym_a][sym_b] = float(corr_matrix.loc[sym_a, sym_b])

        return result

    def _identify_correlated_pairs(
        self, corr_matrix: dict[str, dict[str, float]], warnings: list[str]
    ) -> list[CorrelationPair]:
        """Identify highly correlated position pairs.

        Args:
            corr_matrix: Correlation matrix
            warnings: List to append warnings to

        Returns:
            List of correlated pairs above threshold
        """
        pairs = []

        symbols = list(corr_matrix.keys())
        for i, sym_a in enumerate(symbols):
            for sym_b in symbols[i + 1 :]:  # Avoid duplicates
                corr = corr_matrix[sym_a][sym_b]

                if corr >= self.correlation_threshold:
                    # Get sectors
                    sector_a = self._get_sector(sym_a, warnings)
                    sector_b = self._get_sector(sym_b, warnings)

                    pairs.append(
                        CorrelationPair(
                            symbol_a=sym_a,
                            symbol_b=sym_b,
                            correlation=corr,
                            sector_a=sector_a,
                            sector_b=sector_b,
                            same_sector=(sector_a == sector_b),
                        )
                    )

        # Sort by correlation descending
        pairs.sort(key=lambda p: p.correlation, reverse=True)

        return pairs

    def _calculate_diversification_ratio(
        self, returns_df: pd.DataFrame, positions: dict[str, BrokerPosition]
    ) -> float:
        """Calculate portfolio diversification ratio.

        Args:
            returns_df: Aligned returns DataFrame
            positions: Position dict for weights

        Returns:
            Diversification ratio (portfolio_vol / weighted_avg_individual_vol)
        """
        # Calculate position weights
        total_value = sum(pos.market_value for pos in positions.values())
        if total_value == 0:
            return 1.0

        weights = {}
        for symbol in returns_df.columns:
            if symbol in positions:
                weights[symbol] = positions[symbol].market_value / total_value
            else:
                weights[symbol] = 0.0

        # Convert to array in same order as returns_df columns
        weight_array = pd.Series([weights[sym] for sym in returns_df.columns], index=returns_df.columns)

        # Calculate individual volatilities (annualized)
        individual_vols = returns_df.std() * (252**0.5)

        # Weighted average individual volatility
        weighted_avg_vol = (weight_array * individual_vols).sum()

        if weighted_avg_vol == 0:
            return 1.0

        # Portfolio volatility
        cov_matrix = returns_df.cov() * 252  # Annualized
        portfolio_variance = weight_array @ cov_matrix @ weight_array
        portfolio_vol = portfolio_variance**0.5

        diversification_ratio = portfolio_vol / weighted_avg_vol

        return float(diversification_ratio)

    def _generate_substitutions(
        self,
        correlated_pairs: list[CorrelationPair],
        screening_results: list[ScreeningResult] | None,
        corr_matrix: dict[str, dict[str, float]],
        warnings: list[str],
    ) -> list[SubstitutionSuggestion]:
        """Generate substitution suggestions for correlated positions.

        Args:
            correlated_pairs: Highly correlated pairs
            screening_results: Latest screening candidates
            corr_matrix: Full correlation matrix
            warnings: List to append warnings to

        Returns:
            List of substitution suggestions
        """
        if not screening_results:
            warnings.append("No screening results for substitutions")
            return []

        suggestions = []

        # Group pairs by symbol to find most correlated positions
        symbol_correlations: dict[str, list[tuple[str, float]]] = {}
        for pair in correlated_pairs:
            if pair.symbol_a not in symbol_correlations:
                symbol_correlations[pair.symbol_a] = []
            if pair.symbol_b not in symbol_correlations:
                symbol_correlations[pair.symbol_b] = []

            symbol_correlations[pair.symbol_a].append((pair.symbol_b, pair.correlation))
            symbol_correlations[pair.symbol_b].append((pair.symbol_a, pair.correlation))

        # Find symbols with most correlation issues
        for symbol, correlations in sorted(
            symbol_correlations.items(), key=lambda x: len(x[1]), reverse=True
        ):
            if len(suggestions) >= MAX_SUBSTITUTION_SUGGESTIONS:
                break

            sector = self._get_sector(symbol, warnings)
            avg_corr = sum(c[1] for c in correlations) / len(correlations)

            top_alts, top_corrs = self._find_low_correlation_alternatives(
                avg_corr, screening_results, corr_matrix
            )

            if top_alts:
                suggestions.append(
                    SubstitutionSuggestion(
                        symbol_to_replace=symbol,
                        reason=f"High correlation with {len(correlations)} positions (avg {avg_corr:.2f})",
                        alternatives=top_alts,
                        alternative_correlations=top_corrs,
                        sector=sector,
                    )
                )

        return suggestions

    def _find_low_correlation_alternatives(
        self,
        avg_corr: float,
        screening_results: list[ScreeningResult],
        corr_matrix: dict[str, dict[str, float]],
    ) -> tuple[list[str], list[float]]:
        """Find low-correlation alternative positions from screening results.

        Args:
            avg_corr: Average correlation with other positions
            screening_results: Latest screening candidates
            corr_matrix: Full correlation matrix

        Returns:
            Tuple of (alternatives, correlations) sorted by lowest correlation
        """
        alternatives = []
        alt_correlations = []

        for result in screening_results:
            candidate = result.symbol

            if candidate in corr_matrix:
                continue

            candidate_avg_corr = self._calculate_avg_correlation_with_portfolio(
                candidate, list(corr_matrix.keys())
            )

            if candidate_avg_corr < avg_corr:
                alternatives.append(candidate)
                alt_correlations.append(candidate_avg_corr)

        if alternatives:
            sorted_alts = sorted(zip(alternatives, alt_correlations, strict=False), key=lambda x: x[1])
            top_alts = [alt for alt, _ in sorted_alts[:3]]
            top_corrs = [corr for _, corr in sorted_alts[:3]]
            return top_alts, top_corrs

        return [], []

    def _get_cached_returns(self, symbol: str) -> pd.Series | None:
        """Get cached returns or fetch and cache if not available.

        Args:
            symbol: Stock symbol

        Returns:
            Daily returns series or None if fetch fails
        """
        if symbol in self._returns_cache:
            return self._returns_cache[symbol]

        try:
            market_data = self.market_fetcher.fetch_daily(symbol, period_days=self.lookback_days)
            if market_data.data.empty:
                return None

            returns = market_data.data["Close"].pct_change().dropna()
            self._returns_cache[symbol] = returns
            return returns

        except Exception as e:
            logger.debug(f"Failed to fetch returns for {symbol}: {e}")
            return None

    def _calculate_avg_correlation_with_portfolio(
        self, candidate: str, portfolio_symbols: list[str]
    ) -> float:
        """Calculate average correlation between candidate and portfolio positions.

        Args:
            candidate: Candidate symbol
            portfolio_symbols: Current portfolio symbols

        Returns:
            Average correlation with portfolio
        """
        try:
            # Fetch candidate returns (uses cache)
            candidate_returns = self._get_cached_returns(candidate)
            if candidate_returns is None:
                return 1.0  # Assume high correlation if no data

            correlations = []
            for symbol in portfolio_symbols:
                try:
                    # Use cached portfolio returns
                    portfolio_returns = self._get_cached_returns(symbol)
                    if portfolio_returns is None:
                        continue

                    # Align and calculate correlation
                    aligned = pd.DataFrame({"candidate": candidate_returns, "portfolio": portfolio_returns})
                    aligned = aligned.dropna()

                    if len(aligned) >= MIN_DATA_POINTS:
                        corr = aligned["candidate"].corr(aligned["portfolio"])
                        if pd.notna(corr):
                            correlations.append(abs(corr))
                        else:
                            correlations.append(1.0)  # Treat undefined as high correlation

                except Exception as e:
                    logger.debug(f"Failed to calculate correlation {candidate}-{symbol}: {e}")
                    continue

            return sum(correlations) / len(correlations) if correlations else 1.0

        except Exception as e:
            logger.warning(f"Failed to calculate avg correlation for {candidate}: {e}")
            return 1.0

    def _get_sector(self, symbol: str, warnings: list[str]) -> str:
        """Get sector for symbol using yfinance.

        Args:
            symbol: Stock symbol
            warnings: List to append warnings to

        Returns:
            Sector name or "Unknown"
        """
        if symbol not in self._ticker_cache:
            self._ticker_cache[symbol] = yf.Ticker(symbol)

        try:
            info = self._ticker_cache[symbol].info
            return info.get("sector", "Unknown")
        except Exception as e:
            logger.debug(f"Failed to get sector for {symbol}: {e}")
            warnings.append(f"Missing sector: {symbol}")
            return "Unknown"
