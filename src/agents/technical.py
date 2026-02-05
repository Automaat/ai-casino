"""Technical Analysis Agent."""

from typing import cast

import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field

from src.models.llm import LLMClient
from src.models.providers.base import StructuredOutputError
from src.prompts import PromptLoader
from src.strategies.ensemble import EnsembleResult, EnsembleStrategy
from src.strategies.mean_reversion import MeanReversionIndicators, MeanReversionStrategy
from src.strategies.momentum import MomentumIndicators, MomentumStrategy, Signal
from src.strategies.trend_following import TrendFollowingIndicators, TrendFollowingStrategy


class TechnicalLLMResponse(BaseModel):
    """LLM response structure for technical analysis."""

    interpretation: str = Field(description="Technical analysis interpretation")
    confidence_keywords: list[str] = Field(
        description="Keywords indicating confidence: 'high confidence', 'strong signal', 'weak', 'uncertain', etc."
    )


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
        self._prompt_loader = PromptLoader("technical")
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

        try:
            llm_response = await self.llm.astructured(
                prompt, TechnicalLLMResponse, system=system_prompt, temperature=0.3
            )
            interpretation = llm_response.interpretation
            confidence_keywords = llm_response.confidence_keywords
        except StructuredOutputError as e:
            logger.warning(f"Structured output failed, falling back to text parsing: {e}")
            interpretation = await self.llm.acomplete(prompt, system=system_prompt, temperature=0.3)
            confidence_keywords = []

        # Extract RSI/MACD if available (for downstream agents)
        rsi, macd_hist, ensemble_result = self._extract_indicator_values(indicators)
        confidence = self._calculate_confidence_with_keywords(interpretation, indicators, confidence_keywords)

        logger.info(f"Technical analysis complete: {signal.value} (confidence={confidence:.2f})")

        return TechnicalAnalysis(
            signal=signal,
            rsi=rsi,
            macd_hist=macd_hist,
            interpretation=interpretation,
            confidence=confidence,
            ensemble_result=ensemble_result,
        )

    def _build_prompt(
        self, symbol: str, latest_close: float, signal: Signal, indicators: IndicatorsType
    ) -> tuple[str, str]:
        """Build appropriate prompt based on strategy type."""
        if isinstance(self.strategy, EnsembleStrategy):
            prompt_type = "ensemble"
            prompt_vars = self._build_ensemble_vars(symbol, latest_close, signal, indicators)
        elif isinstance(self.strategy, TrendFollowingStrategy):
            prompt_type = "trend_following"
            prompt_vars = self._build_trend_following_vars(symbol, latest_close, signal, indicators)
        elif isinstance(self.strategy, MeanReversionStrategy):
            prompt_type = "mean_reversion"
            prompt_vars = self._build_mean_reversion_vars(symbol, latest_close, signal, indicators)
        else:
            prompt_type = "momentum"
            prompt_vars = self._build_momentum_vars(symbol, latest_close, signal, indicators)

        system = self._prompt_loader.load(f"system_{prompt_type}")
        user = self._prompt_loader.load(f"user_{prompt_type}", **prompt_vars)
        return user, system

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

    def _calculate_confidence_with_keywords(
        self, interpretation: str, indicators: IndicatorsType, keywords: list[str]
    ) -> float:
        """Calculate confidence using structured keywords when available."""
        if isinstance(indicators, EnsembleResult):
            return indicators.confidence

        # Use keywords if available, otherwise fall back to text parsing
        if keywords:
            keywords_lower = [k.lower() for k in keywords]
            has_high_confidence = any(
                word in keywords_lower for word in ["high confidence", "strong signal", "strong"]
            )
        else:
            has_high_confidence = (
                "high confidence" in interpretation.lower() or "strong signal" in interpretation.lower()
            )

        if isinstance(indicators, MomentumIndicators):
            confidence = self._calculate_base_momentum_confidence(indicators)
            if has_high_confidence:
                confidence = min(confidence + 0.1, 1.0)
            return confidence
        if isinstance(indicators, TrendFollowingIndicators):
            return self._extract_trend_confidence(indicators)
        if isinstance(indicators, MeanReversionIndicators):
            return self._extract_mean_reversion_confidence(indicators)
        return 0.5

    def _calculate_base_momentum_confidence(self, indicators: MomentumIndicators) -> float:
        """Calculate base confidence for momentum indicators without text parsing."""
        if (indicators.rsi_oversold and indicators.macd_bullish) or (
            indicators.rsi_overbought and indicators.macd_bearish
        ):
            return 0.8
        if (
            indicators.rsi_oversold
            or indicators.macd_bullish
            or indicators.rsi_overbought
            or indicators.macd_bearish
        ):
            return 0.6
        return 0.5

    def _build_momentum_vars(
        self, symbol: str, latest_close: float, signal: Signal, indicators: MomentumIndicators
    ) -> dict[str, str]:
        """Build variables for momentum prompt."""
        strategy = cast("MomentumStrategy", self.strategy)
        rsi_status = (
            "OVERSOLD"
            if indicators.rsi_oversold
            else "OVERBOUGHT"
            if indicators.rsi_overbought
            else "NEUTRAL"
        )
        macd_trend = "BULLISH" if indicators.macd_bullish else "BEARISH"

        return {
            "symbol": symbol,
            "latest_close": f"{latest_close:.2f}",
            "rsi_period": str(strategy.rsi_period),
            "rsi": f"{indicators.rsi:.2f}",
            "rsi_oversold": str(strategy.rsi_oversold),
            "rsi_overbought": str(strategy.rsi_overbought),
            "rsi_status": rsi_status,
            "macd": f"{indicators.macd:.4f}",
            "macd_signal": f"{indicators.macd_signal:.4f}",
            "macd_hist": f"{indicators.macd_hist:.4f}",
            "macd_trend": macd_trend,
            "signal": signal.value,
        }

    def _build_trend_following_vars(
        self, symbol: str, latest_close: float, signal: Signal, indicators: TrendFollowingIndicators
    ) -> dict[str, str]:
        """Build variables for trend following prompt."""
        strategy = cast("TrendFollowingStrategy", self.strategy)
        cross_status = (
            "GOLDEN CROSS"
            if indicators.sma_bullish_cross
            else ("DEATH CROSS" if indicators.sma_bearish_cross else "NO CROSSOVER")
        )
        trend_strength = "STRONG" if indicators.strong_trend else "WEAK"

        return {
            "symbol": symbol,
            "latest_close": f"{latest_close:.2f}",
            "sma_fast": str(strategy.sma_fast),
            "sma_fast_value": f"{indicators.sma_fast:.2f}",
            "sma_slow": str(strategy.sma_slow),
            "sma_slow_value": f"{indicators.sma_slow:.2f}",
            "cross_status": cross_status,
            "adx_period": str(strategy.adx_period),
            "adx": f"{indicators.adx:.2f}",
            "adx_threshold": str(strategy.adx_threshold),
            "trend_strength": trend_strength,
            "plus_di": f"{indicators.plus_di:.2f}",
            "minus_di": f"{indicators.minus_di:.2f}",
            "trend_direction": indicators.trend_direction.upper(),
            "signal": signal.value,
        }

    def _build_mean_reversion_vars(
        self, symbol: str, latest_close: float, signal: Signal, indicators: MeanReversionIndicators
    ) -> dict[str, str]:
        """Build variables for mean reversion prompt."""
        bb_status = (
            "OVERSOLD" if indicators.oversold else ("OVERBOUGHT" if indicators.overbought else "NEUTRAL")
        )

        return {
            "symbol": symbol,
            "latest_close": f"{latest_close:.2f}",
            "bb_upper": f"{indicators.bb_upper:.2f}",
            "bb_middle": f"{indicators.bb_middle:.2f}",
            "bb_lower": f"{indicators.bb_lower:.2f}",
            "bb_width": f"{indicators.bb_width:.2f}",
            "bb_percent": f"{indicators.bb_percent:.2f}",
            "bb_status": bb_status,
            "signal": signal.value,
        }

    def _build_ensemble_vars(
        self, symbol: str, latest_close: float, signal: Signal, result: EnsembleResult
    ) -> dict[str, str]:
        """Build variables for ensemble prompt.

        Args:
            symbol: Stock ticker
            latest_close: Latest closing price
            signal: Aggregated signal
            result: Ensemble result with strategy breakdowns

        Returns:
            Variables dictionary
        """
        strategy_breakdown = []
        for sr in result.strategy_results:
            strategy_breakdown.append(f"- {sr.name}: {sr.signal.value} (weight={sr.weight:.2f})")

        return {
            "symbol": symbol,
            "latest_close": f"{latest_close:.2f}",
            "strategy_breakdown": "\n".join(strategy_breakdown),
            "signal": signal.value,
            "agreement_ratio": f"{result.agreement_ratio:.2f}",
            "confidence": f"{result.confidence:.2f}",
            "conflict_resolved": str(result.conflict_resolved),
        }

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
