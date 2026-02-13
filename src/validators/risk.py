"""Pre-decision risk validation using deterministic rules."""

from datetime import UTC, datetime

import pandas as pd
from loguru import logger

from src.agents.fundamental import FundamentalAnalysis
from src.agents.news import NewsAnalysis
from src.agents.sentiment import SentimentAnalysis
from src.agents.technical import TechnicalAnalysis
from src.agents.thesis_researcher import BearishResearchAnalysis, BullishResearchAnalysis
from src.daemon.config.risk_validation import RiskValidationConfig
from src.daemon.degradation import DegradationContext
from src.strategies.session import TradingSession
from src.strategies.signal import Signal
from src.strategies.timeframe import MultiTimeframeData
from src.workflows.models.risk_validation import SignalConsistency, ValidationResult


class RiskValidator:
    """Pre-decision risk validation using deterministic rules."""

    def __init__(self, config: RiskValidationConfig | None = None) -> None:
        """Initialize risk validator.

        Args:
            config: Risk validation configuration (uses defaults if None)
        """
        self.config = config or RiskValidationConfig()
        logger.info("Initialized RiskValidator")

    def validate(
        self,
        symbol: str,
        trading_session: TradingSession,
        technical: TechnicalAnalysis | None,
        _sentiment: SentimentAnalysis | None,
        _news: NewsAnalysis | None,
        fundamental: FundamentalAnalysis | None,
        bullish: BullishResearchAnalysis | None,
        bearish: BearishResearchAnalysis | None,
        market_data: pd.DataFrame | MultiTimeframeData | None,
        degradation_context: DegradationContext | None,
    ) -> ValidationResult:
        """Validate analyses - synchronous, deterministic logic.

        Args:
            symbol: Stock ticker symbol
            trading_session: Current trading session (REGULAR or PRE_MARKET)
            technical: Technical analysis result
            sentiment: Sentiment analysis result
            news: News analysis result
            fundamental: Fundamental analysis result
            bullish: Bullish research result
            bearish: Bearish research result
            market_data: Market data (for freshness check)
            degradation_context: Degradation context (for circuit breaker status)

        Returns:
            ValidationResult with approval status, warnings, and risk level
        """
        warnings: list[str] = []
        constraints_met: dict[str, bool] = {}
        blocking_issues: list[str] = []

        # Check degradation context (highest priority)
        from src.daemon.degradation import DegradationTier

        if degradation_context and degradation_context.tier == DegradationTier.HALTED:
            reason = degradation_context.halt_reason or "Unknown reason"
            blocking_issues.append(f"Trading halted due to {reason}")
            constraints_met["degradation_check"] = False
        else:
            constraints_met["degradation_check"] = True

        # Check confidence thresholds
        confidence_check = self._check_confidence_thresholds(
            technical, _sentiment, _news, fundamental, bullish, bearish, warnings
        )
        constraints_met["confidence_thresholds"] = confidence_check

        # Check signal consistency
        signal_consistency = self._check_signal_consistency(technical, _sentiment, _news, warnings)
        constraints_met["signal_consistency"] = not signal_consistency.conflicting_signals or (
            len(signal_consistency.conflict_details) <= self.config.max_conflicting_signals
        )

        # Check trading session rules
        session_check = self._check_trading_session(trading_session, technical, _sentiment, _news, warnings)
        constraints_met["trading_session"] = session_check

        # Check data freshness
        freshness_check = self._check_data_freshness(market_data, warnings)
        constraints_met["data_freshness"] = freshness_check

        # Check suspicious patterns
        suspicious_check = self._check_suspicious_patterns(technical, _sentiment, _news, warnings)
        constraints_met["suspicious_patterns"] = suspicious_check

        # Calculate aggregate confidence
        aggregate_confidence = self._calculate_aggregate_confidence(
            technical, sentiment, news, fundamental, bullish, bearish
        )

        # Determine risk level
        risk_level = self._determine_risk_level(aggregate_confidence, signal_consistency, degradation_context)

        # Determine approval (warn but continue unless blocking issues)
        approved = len(blocking_issues) == 0

        return ValidationResult(
            approved=approved,
            risk_level=risk_level,
            confidence_score=aggregate_confidence,
            warnings=warnings,
            constraints_met=constraints_met,
            blocking_issues=blocking_issues,
            signal_consistency=signal_consistency,
        )

    def _check_confidence_thresholds(
        self,
        technical: TechnicalAnalysis | None,
        _sentiment: SentimentAnalysis | None,
        _news: NewsAnalysis | None,
        fundamental: FundamentalAnalysis | None,
        bullish: BullishResearchAnalysis | None,
        bearish: BearishResearchAnalysis | None,
        warnings: list[str],
    ) -> bool:
        """Check confidence thresholds for each analysis type.

        Args:
            technical: Technical analysis result
            _sentiment: Sentiment analysis result (no confidence field, unused)
            _news: News analysis result (no confidence field, unused)
            fundamental: Fundamental analysis result
            bullish: Bullish research result
            bearish: Bearish research result
            warnings: List to append warnings to

        Returns:
            True if all thresholds met, False otherwise
        """
        all_pass = True

        if technical and technical.confidence < self.config.min_technical_confidence:
            warnings.append(
                f"Technical confidence ({technical.confidence:.2f}) below threshold "
                f"({self.config.min_technical_confidence:.2f})"
            )
            all_pass = False

        # Note: sentiment and news analyses don't have confidence fields in current implementation

        # Check research analyses
        if bullish and bullish.confidence < self.config.min_research_confidence:
            warnings.append(
                f"Bullish research confidence ({bullish.confidence:.2f}) below threshold "
                f"({self.config.min_research_confidence:.2f})"
            )
            all_pass = False

        if bearish and bearish.confidence < self.config.min_research_confidence:
            warnings.append(
                f"Bearish research confidence ({bearish.confidence:.2f}) below threshold "
                f"({self.config.min_research_confidence:.2f})"
            )
            all_pass = False

        return all_pass

    def _check_signal_consistency(
        self,
        technical: TechnicalAnalysis | None,
        _sentiment: SentimentAnalysis | None,
        _news: NewsAnalysis | None,
        warnings: list[str],
    ) -> SignalConsistency:
        """Check signal consistency across analyses.

        Args:
            technical: Technical analysis result
            _sentiment: Sentiment analysis result (no signal field, unused)
            _news: News analysis result (no signal field, unused)
            warnings: List to append warnings to

        Returns:
            SignalConsistency with conflict analysis
        """
        signal_distribution: dict[Signal, int] = {}
        conflict_details: list[str] = []

        # Collect signals (only technical has signal field in current implementation)
        signals: list[tuple[str, Signal]] = []
        if technical:
            signals.append(("technical", technical.signal))
            signal_distribution[technical.signal] = signal_distribution.get(technical.signal, 0) + 1

        # Note: sentiment and news analyses don't have signal fields in current implementation

        # Check for conflicts (BUY vs SELL)
        has_buy = Signal.BUY in signal_distribution
        has_sell = Signal.SELL in signal_distribution

        conflicting = has_buy and has_sell

        if conflicting:
            buy_sources = [name for name, sig in signals if sig == Signal.BUY]
            sell_sources = [name for name, sig in signals if sig == Signal.SELL]
            conflict_details.append(f"BUY signals: {', '.join(buy_sources)}")
            conflict_details.append(f"SELL signals: {', '.join(sell_sources)}")

            if not self.config.allow_conflicting_signals:
                warnings.append("Conflicting signals detected (BUY vs SELL)")
            elif len(conflict_details) > self.config.max_conflicting_signals:
                warnings.append(
                    f"Excessive conflicting signals ({len(conflict_details)} > "
                    f"{self.config.max_conflicting_signals})"
                )

        return SignalConsistency(
            conflicting_signals=conflicting,
            signal_distribution=signal_distribution,
            conflict_details=conflict_details,
        )

    def _check_trading_session(
        self,
        trading_session: TradingSession,
        technical: TechnicalAnalysis | None,
        _sentiment: SentimentAnalysis | None,
        _news: NewsAnalysis | None,
        warnings: list[str],
    ) -> bool:
        """Check trading session-specific rules.

        Args:
            trading_session: Current trading session
            technical: Technical analysis result
            _sentiment: Sentiment analysis result (no confidence field, unused)
            _news: News analysis result (no confidence field, unused)
            warnings: List to append warnings to

        Returns:
            True if session rules met, False otherwise
        """
        if trading_session != TradingSession.PRE_MARKET:
            return True  # No special rules for regular session

        # Pre-market: require higher confidence (only technical has confidence)
        confidences = [technical.confidence] if technical else []

        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        if avg_confidence < self.config.pre_market_min_confidence:
            warnings.append(
                f"Pre-market trading requires higher confidence ({avg_confidence:.2f} < "
                f"{self.config.pre_market_min_confidence:.2f})"
            )
            return False

        return True

    def _check_data_freshness(
        self,
        market_data: pd.DataFrame | MultiTimeframeData | None,
        warnings: list[str],
    ) -> bool:
        """Check data freshness (market data age).

        Args:
            market_data: Market data
            warnings: List to append warnings to

        Returns:
            True if data is fresh, False otherwise
        """
        if market_data is None:
            warnings.append("No market data available")
            return False

        # Extract timestamp
        if isinstance(market_data, MultiTimeframeData):
            last_updated = market_data.last_updated
        else:
            # Assume DataFrame has a timestamp index or last_updated attribute
            if not hasattr(market_data, "index") or len(market_data) == 0:
                warnings.append("Market data is empty or malformed")
                return False
            # Get last timestamp from index (assuming it's a DatetimeIndex)
            try:
                last_updated = pd.to_datetime(market_data.index[-1])
            except Exception:
                # Can't parse timestamp, skip freshness check
                return True

        # Check age
        now = datetime.now(UTC)
        if last_updated.tzinfo is None:
            last_updated = last_updated.replace(tzinfo=UTC)

        age_minutes = (now - last_updated).total_seconds() / 60

        if age_minutes > self.config.max_data_age_minutes:
            warnings.append(
                f"Market data is stale ({age_minutes:.0f} minutes old, threshold: "
                f"{self.config.max_data_age_minutes} minutes)"
            )
            return False

        return True

    def _check_suspicious_patterns(
        self,
        technical: TechnicalAnalysis | None,
        _sentiment: SentimentAnalysis | None,
        _news: NewsAnalysis | None,
        warnings: list[str],
    ) -> bool:
        """Check for suspicious patterns (e.g., all confidences >0.95).

        Args:
            technical: Technical analysis result
            _sentiment: Sentiment analysis result (no confidence field, unused)
            _news: News analysis result (no confidence field, unused)
            warnings: List to append warnings to

        Returns:
            True if no suspicious patterns, False otherwise
        """
        # Only technical has confidence in current implementation
        confidences = [technical.confidence] if technical else []

        non_zero = [c for c in confidences if c > 0]
        if len(non_zero) >= 2 and all(c > 0.95 for c in non_zero):
            warnings.append("Suspicious: all confidences >0.95 (possible overfitting)")
            return False

        return True

    def _calculate_aggregate_confidence(
        self,
        technical: TechnicalAnalysis | None,
        sentiment: SentimentAnalysis | None,
        news: NewsAnalysis | None,
        fundamental: FundamentalAnalysis | None,
        bullish: BullishResearchAnalysis | None,
        bearish: BearishResearchAnalysis | None,
    ) -> float:
        """Calculate aggregate confidence across all analyses.

        Args:
            technical: Technical analysis result
            sentiment: Sentiment analysis result
            news: News analysis result
            fundamental: Fundamental analysis result
            bullish: Bullish research result
            bearish: Bearish research result

        Returns:
            Weighted average confidence (0.0-1.0)
        """
        confidences = []
        if technical:
            confidences.append(technical.confidence)
        if sentiment:
            confidences.append(sentiment.confidence)
        if news:
            confidences.append(news.confidence)
        if fundamental:
            confidences.append(fundamental.confidence)
        if bullish:
            confidences.append(bullish.confidence)
        if bearish:
            confidences.append(bearish.confidence)

        if not confidences:
            return 0.5  # Default if no analyses available

        return sum(confidences) / len(confidences)

    def _determine_risk_level(
        self,
        aggregate_confidence: float,
        signal_consistency: SignalConsistency,
        degradation_context: DegradationContext | None,
    ) -> str:
        """Determine risk level based on validation results.

        Args:
            aggregate_confidence: Aggregate confidence score
            signal_consistency: Signal consistency analysis
            degradation_context: Degradation context

        Returns:
            Risk level: LOW, MEDIUM, or HIGH
        """
        from src.daemon.degradation import DegradationTier

        # High risk if degraded or halted
        if degradation_context and degradation_context.tier in {
            DegradationTier.DEGRADED,
            DegradationTier.HALTED,
        }:
            return "HIGH"

        # High risk if low confidence or conflicting signals
        if aggregate_confidence < 0.5 or signal_consistency.conflicting_signals:
            return "HIGH"

        # Medium risk if moderate confidence
        if aggregate_confidence < 0.75:
            return "MEDIUM"

        # Low risk otherwise
        return "LOW"

    def __repr__(self) -> str:
        """String representation."""
        return f"RiskValidator(enabled={self.config.enabled}, min_confidence={self.config.min_overall_confidence})"
