"""Pattern analyzer for detecting trading patterns from historical data."""

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Final

from loguru import logger

from src.coordinator.pattern_models import PatternInsight, PatternType

if TYPE_CHECKING:
    from src.coordinator.memory import CoordinatorMemory
    from src.database.repositories.analysis import AnalysisRecordRepository
    from src.database.repositories.trade import TradeRepository

# Pattern detection thresholds
HIGH_WIN_RATE_THRESHOLD: Final[float] = 0.6
LOW_WIN_RATE_THRESHOLD: Final[float] = 0.4


class PatternAnalyzer:
    """Detect patterns from historical analysis and trade data."""

    def __init__(
        self,
        analysis_repo: AnalysisRecordRepository,
        trade_repo: TradeRepository,
        memory: CoordinatorMemory,
        min_sample_size: int = 10,
    ) -> None:
        """Initialize pattern analyzer.

        Args:
            analysis_repo: Analysis record repository
            trade_repo: Trade repository
            memory: Coordinator memory for saving insights
            min_sample_size: Minimum observations for pattern confidence
        """
        self._analysis_repo = analysis_repo
        self._trade_repo = trade_repo
        self._memory = memory
        self._min_sample_size = min_sample_size

    async def analyze_patterns(self, lookback_days: int = 30) -> list[PatternInsight]:
        """Run all pattern detections.

        Args:
            lookback_days: Days to look back for pattern analysis

        Returns:
            List of detected pattern insights
        """
        patterns: list[PatternInsight] = []

        try:
            # Run all pattern detection methods
            patterns.extend(await self._analyze_symbol_performance(lookback_days))
            patterns.extend(await self._analyze_confidence_calibration(lookback_days))
            patterns.extend(await self._analyze_timing_patterns(lookback_days))
            patterns.extend(await self._analyze_technical_indicators(lookback_days))
            patterns.extend(await self._analyze_execution_gaps(lookback_days))

            logger.info(f"Pattern analysis complete: {len(patterns)} insights detected")
            return patterns

        except Exception as e:
            logger.opt(exception=True).error(f"Pattern analysis failed: {e}")
            return []

    async def _analyze_symbol_performance(self, days: int) -> list[PatternInsight]:
        """Detect which symbols have highest/lowest win rates.

        Args:
            days: Days to look back

        Returns:
            List of performance pattern insights
        """
        try:
            start_date = datetime.now(UTC) - timedelta(days=days)

            # Get closed trades since start_date (efficient SQL filtering)
            trades = await self._trade_repo.get_closed_since(start_date)

            if not trades:
                return []

            # Calculate metrics per symbol
            symbol_metrics = self._calculate_symbol_metrics(trades)

            # Generate insights
            return self._generate_performance_insights(symbol_metrics)

        except Exception as e:
            logger.opt(exception=True).warning(f"Symbol performance analysis failed: {e}")
            return []

    def _calculate_symbol_metrics(self, trades: list) -> dict[str, dict]:
        """Calculate performance metrics per symbol.

        Args:
            trades: List of trade records

        Returns:
            Dictionary of symbol metrics
        """
        symbol_metrics: dict[str, dict] = {}
        for trade in trades:
            symbol = trade.symbol

            # Only count CLOSED trades with exit_price for accurate metrics
            if not (trade.exit_price and trade.entry_price):
                continue

            if symbol not in symbol_metrics:
                symbol_metrics[symbol] = {"wins": 0, "losses": 0, "total": 0}

            symbol_metrics[symbol]["total"] += 1

            # Use trade.pnl if available, otherwise compute from prices
            pnl = getattr(trade, "pnl", None)
            if pnl is None:
                # TradeRecord uses shares field
                pnl = (trade.exit_price - trade.entry_price) * trade.shares

            if pnl > 0:
                symbol_metrics[symbol]["wins"] += 1
            else:
                symbol_metrics[symbol]["losses"] += 1

        return symbol_metrics

    def _generate_performance_insights(self, symbol_metrics: dict[str, dict]) -> list[PatternInsight]:
        """Generate insights from symbol metrics.

        Args:
            symbol_metrics: Dictionary of symbol performance metrics

        Returns:
            List of pattern insights
        """
        insights: list[PatternInsight] = []
        for symbol, metrics in symbol_metrics.items():
            if metrics["total"] < self._min_sample_size:
                continue

            win_rate = metrics["wins"] / metrics["total"] if metrics["total"] > 0 else 0.0

            # Flag high performers
            if win_rate > HIGH_WIN_RATE_THRESHOLD:
                insights.append(
                    PatternInsight(
                        pattern_type=PatternType.SYMBOL_PERFORMANCE,
                        symbol=symbol,
                        confidence=0.8,
                        sample_size=metrics["total"],
                        insight_text=f"{symbol} shows strong performance with {win_rate:.0%} win rate",
                        recommendation=f"Consider increasing position sizing for {symbol} trades",
                        detected_at=datetime.now(UTC),
                    )
                )

            # Flag poor performers
            elif win_rate < LOW_WIN_RATE_THRESHOLD:
                insights.append(
                    PatternInsight(
                        pattern_type=PatternType.SYMBOL_PERFORMANCE,
                        symbol=symbol,
                        confidence=0.7,
                        sample_size=metrics["total"],
                        insight_text=f"{symbol} shows weak performance with {win_rate:.0%} win rate",
                        recommendation=f"Reduce exposure to {symbol} or require higher confidence",
                        detected_at=datetime.now(UTC),
                    )
                )

        return insights

    async def _analyze_confidence_calibration(self, _days: int) -> list[PatternInsight]:
        """Check if confidence correlates with actual outcomes.

        Args:
            _days: Days to look back (currently unused - placeholder)

        Returns:
            List of calibration pattern insights
        """
        # Placeholder for future implementation
        # Would bucket confidence ranges and measure actual win rates
        return []

    async def _analyze_timing_patterns(self, _days: int) -> list[PatternInsight]:
        """Detect time-of-day and session patterns.

        Args:
            _days: Days to look back (currently unused - placeholder)

        Returns:
            List of timing pattern insights
        """
        # Placeholder for future implementation
        # Would compare PRE_MARKET vs REGULAR session outcomes
        return []

    async def _analyze_technical_indicators(self, _days: int) -> list[PatternInsight]:
        """Detect RSI/MACD ranges correlated with wins.

        Args:
            _days: Days to look back (currently unused - placeholder)

        Returns:
            List of indicator pattern insights
        """
        # Placeholder for future implementation
        # Would group by RSI/MACD ranges and calculate win rates
        return []

    async def _analyze_execution_gaps(self, _days: int) -> list[PatternInsight]:
        """Analyze recommended vs executed trades.

        Args:
            _days: Days to look back (currently unused - placeholder)

        Returns:
            List of execution gap insights
        """
        # Placeholder for future implementation
        # Would count recommendations vs executions to identify filtering patterns
        return []

    def __repr__(self) -> str:
        """String representation."""
        return f"PatternAnalyzer(min_sample_size={self._min_sample_size})"
