"""Technical Analysis Agent."""

import pandas as pd
from loguru import logger
from pydantic import BaseModel

from src.models.llm import LLMClient
from src.strategies.ensemble import EnsembleResult, EnsembleStrategy
from src.strategies.mean_reversion import MeanReversionIndicators, MeanReversionStrategy
from src.strategies.momentum import MomentumIndicators, MomentumStrategy, Signal
from src.strategies.trend_following import TrendFollowingIndicators, TrendFollowingStrategy

StrategyType = MomentumStrategy | MeanReversionStrategy | TrendFollowingStrategy | EnsembleStrategy
IndicatorsType = MomentumIndicators | MeanReversionIndicators | TrendFollowingIndicators | EnsembleResult


class TechnicalAnalysis(BaseModel):
    """Technical analysis result."""

    signal: Signal
    rsi: float | None = None
    macd_hist: float | None = None
    interpretation: str
    confidence: float
    ensemble_result: EnsembleResult | None = None


class TechnicalAnalyst:
    """Agent for technical analysis of price data."""

    def __init__(self, llm_client: LLMClient, strategy: StrategyType) -> None:
        """Initialize technical analyst.

        Args:
            llm_client: LLM client for generating interpretations
            strategy: Trading strategy for indicator generation
        """
        self.llm = llm_client
        self.strategy = strategy
        self._strategy_type = type(strategy).__name__
        logger.info(f"Initialized TechnicalAnalyst (strategy={self._strategy_type})")

    async def analyze(self, symbol: str, market_data: pd.DataFrame) -> TechnicalAnalysis:
        """Perform technical analysis on market data.

        Args:
            symbol: Stock ticker symbol
            market_data: OHLCV dataframe

        Returns:
            TechnicalAnalysis with signal and interpretation
        """
        logger.info(f"Analyzing {symbol} technicals with {self._strategy_type}")

        signal, indicators = self.strategy.generate_signal(market_data)
        latest_close = float(market_data["Close"].iloc[-1])

        prompt, system_prompt = self._build_prompt(symbol, latest_close, signal, indicators)
        response = await self.llm.acomplete(prompt, system=system_prompt, temperature=0.3)

        # Extract RSI/MACD if available (for downstream agents)
        rsi, macd_hist, ensemble_result = self._extract_indicator_values(indicators)
        confidence = self._calculate_confidence(response, indicators)

        logger.info(f"Technical analysis complete: {signal.value} (confidence={confidence:.2f})")

        return TechnicalAnalysis(
            signal=signal,
            rsi=rsi,
            macd_hist=macd_hist,
            interpretation=response,
            confidence=confidence,
            ensemble_result=ensemble_result,
        )

    def _build_prompt(
        self, symbol: str, latest_close: float, signal: Signal, indicators: IndicatorsType
    ) -> tuple[str, str]:
        """Build appropriate prompt based on strategy type."""
        if isinstance(self.strategy, EnsembleStrategy):
            prompt = self._build_ensemble_prompt(symbol, latest_close, signal, indicators)
            system = (
                "You are a technical analyst specializing in multi-strategy ensemble analysis. "
                "Provide clear, actionable interpretations considering all strategy signals."
            )
        elif isinstance(self.strategy, TrendFollowingStrategy):
            prompt = self._build_trend_following_prompt(symbol, latest_close, signal, indicators)
            system = (
                "You are a technical analyst specializing in trend following strategies. "
                "Provide clear, actionable interpretations based on SMA crossovers and ADX."
            )
        elif isinstance(self.strategy, MeanReversionStrategy):
            prompt = self._build_mean_reversion_prompt(symbol, latest_close, signal, indicators)
            system = (
                "You are a technical analyst specializing in mean reversion strategies. "
                "Provide clear, actionable interpretations based on Bollinger Bands."
            )
        else:
            prompt = self._build_momentum_prompt(symbol, latest_close, signal, indicators)
            system = (
                "You are a technical analyst specializing in momentum indicators. "
                "Provide clear, actionable interpretations."
            )
        return prompt, system

    def _extract_indicator_values(
        self, indicators: IndicatorsType
    ) -> tuple[float | None, float | None, EnsembleResult | None]:
        """Extract RSI, MACD, and ensemble result from indicators."""
        if isinstance(indicators, EnsembleResult):
            rsi, macd_hist = None, None
            for sr in indicators.strategy_results:
                if sr.name == "momentum":
                    rsi = sr.indicators.rsi
                    macd_hist = sr.indicators.macd_hist
                    break
            return rsi, macd_hist, indicators
        if isinstance(indicators, MomentumIndicators):
            return indicators.rsi, indicators.macd_hist, None
        return None, None, None

    def _calculate_confidence(self, response: str, indicators: IndicatorsType) -> float:
        """Calculate confidence based on indicator type."""
        if isinstance(indicators, EnsembleResult):
            return indicators.confidence
        if isinstance(indicators, MomentumIndicators):
            return self._extract_momentum_confidence(response, indicators)
        if isinstance(indicators, TrendFollowingIndicators):
            return self._extract_trend_confidence(indicators)
        if isinstance(indicators, MeanReversionIndicators):
            return self._extract_mean_reversion_confidence(indicators)
        return 0.5

    def _build_momentum_prompt(
        self, symbol: str, latest_close: float, signal: Signal, indicators: MomentumIndicators
    ) -> str:
        """Build LLM prompt for momentum strategy."""
        strategy = self.strategy
        return f"""Analyze these technical indicators for {symbol}:

Current Price: ${latest_close:.2f}

RSI ({strategy.rsi_period}): {indicators.rsi:.2f}
- Oversold threshold: {strategy.rsi_oversold}
- Overbought threshold: {strategy.rsi_overbought}
- Status: {"OVERSOLD" if indicators.rsi_oversold else "OVERBOUGHT" if indicators.rsi_overbought else "NEUTRAL"}

MACD:
- MACD Line: {indicators.macd:.4f}
- Signal Line: {indicators.macd_signal:.4f}
- Histogram: {indicators.macd_hist:.4f}
- Trend: {"BULLISH" if indicators.macd_bullish else "BEARISH"}

Generated Signal: {signal.value}

Provide a concise 2-3 sentence interpretation of these indicators and their implications for trading.
"""

    def _build_trend_following_prompt(
        self, symbol: str, latest_close: float, signal: Signal, indicators: TrendFollowingIndicators
    ) -> str:
        """Build LLM prompt for trend following strategy."""
        strategy = self.strategy
        cross_status = (
            "GOLDEN CROSS"
            if indicators.sma_bullish_cross
            else ("DEATH CROSS" if indicators.sma_bearish_cross else "NO CROSSOVER")
        )
        return f"""Analyze these trend following indicators for {symbol}:

Current Price: ${latest_close:.2f}

SMA Crossover:
- Fast SMA ({strategy.sma_fast}): {indicators.sma_fast:.2f}
- Slow SMA ({strategy.sma_slow}): {indicators.sma_slow:.2f}
- Crossover: {cross_status}

ADX ({strategy.adx_period}): {indicators.adx:.2f}
- Strong trend threshold: {strategy.adx_threshold}
- Trend strength: {"STRONG" if indicators.strong_trend else "WEAK"}
- +DI: {indicators.plus_di:.2f}
- -DI: {indicators.minus_di:.2f}
- Direction: {indicators.trend_direction.upper()}

Generated Signal: {signal.value}

Provide a concise 2-3 sentence interpretation of the trend and its implications for trading.
"""

    def _build_mean_reversion_prompt(
        self, symbol: str, latest_close: float, signal: Signal, indicators: MeanReversionIndicators
    ) -> str:
        """Build LLM prompt for mean reversion strategy."""
        bb_status = (
            "OVERSOLD" if indicators.oversold else ("OVERBOUGHT" if indicators.overbought else "NEUTRAL")
        )
        return f"""Analyze these mean reversion indicators for {symbol}:

Current Price: ${latest_close:.2f}

Bollinger Bands:
- Upper Band: {indicators.bb_upper:.2f}
- Middle Band (SMA): {indicators.bb_middle:.2f}
- Lower Band: {indicators.bb_lower:.2f}
- Band Width: {indicators.bb_width:.2f}%
- %B: {indicators.bb_percent:.2f}
- Status: {bb_status}

Generated Signal: {signal.value}

Provide a concise 2-3 sentence interpretation of the mean reversion setup and its implications for trading.
"""

    def _build_ensemble_prompt(
        self, symbol: str, latest_close: float, signal: Signal, result: EnsembleResult
    ) -> str:
        """Build LLM prompt for ensemble strategy.

        Args:
            symbol: Stock ticker
            latest_close: Latest closing price
            signal: Aggregated signal
            result: Ensemble result with strategy breakdowns

        Returns:
            Prompt string
        """
        strategy_breakdown = []
        for sr in result.strategy_results:
            strategy_breakdown.append(f"- {sr.name}: {sr.signal.value} (weight={sr.weight:.2f})")

        return f"""Analyze this multi-strategy ensemble result for {symbol}:

Current Price: ${latest_close:.2f}

Strategy Signals:
{chr(10).join(strategy_breakdown)}

Aggregated Signal: {signal.value}
Agreement Ratio: {result.agreement_ratio:.2f}
Confidence: {result.confidence:.2f}
Conflict Resolved: {result.conflict_resolved}

Provide a concise 2-3 sentence interpretation of the ensemble analysis, noting any disagreements between strategies and their implications.
"""

    def _extract_momentum_confidence(self, response: str, indicators: MomentumIndicators) -> float:
        """Calculate confidence for momentum indicators."""
        confidence = 0.5

        if (indicators.rsi_oversold and indicators.macd_bullish) or (
            indicators.rsi_overbought and indicators.macd_bearish
        ):
            confidence = 0.8
        elif (
            indicators.rsi_oversold
            or indicators.macd_bullish
            or indicators.rsi_overbought
            or indicators.macd_bearish
        ):
            confidence = 0.6

        if "high confidence" in response.lower() or "strong signal" in response.lower():
            confidence = min(confidence + 0.1, 1.0)

        return confidence

    def _extract_trend_confidence(self, indicators: TrendFollowingIndicators) -> float:
        """Calculate confidence for trend following indicators."""
        confidence = 0.5

        if indicators.strong_trend:
            confidence += 0.2
        if indicators.sma_bullish_cross or indicators.sma_bearish_cross:
            confidence += 0.2
        # DI divergence adds confidence
        di_diff = abs(indicators.plus_di - indicators.minus_di)
        confidence += min(di_diff * 0.01, 0.1)

        return min(confidence, 0.95)

    def _extract_mean_reversion_confidence(self, indicators: MeanReversionIndicators) -> float:
        """Calculate confidence for mean reversion indicators."""
        confidence = 0.5

        if indicators.oversold or indicators.overbought:
            confidence += 0.25
        # Further from middle = higher confidence
        bb_deviation = abs(indicators.bb_percent - 0.5)
        confidence += min(bb_deviation * 0.3, 0.15)

        return min(confidence, 0.95)

    def __repr__(self) -> str:
        """String representation."""
        return f"TechnicalAnalyst(strategy={self.strategy})"
