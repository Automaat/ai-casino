"""Technical Analysis Worker - Pydantic AI migration POC."""

import asyncio
from typing import cast

import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field

from src.agents.technical import TechnicalAnalysis
from src.models.llm import LLMClient
from src.models.providers.base import StructuredOutputError
from src.prompts import PromptLoader
from src.strategies.confluence import ConfluenceCalculator
from src.strategies.ensemble import EnsembleResult, EnsembleStrategy
from src.strategies.mean_reversion import MeanReversionIndicators, MeanReversionStrategy
from src.strategies.momentum import ExhaustionSignals, MomentumIndicators, MomentumStrategy
from src.strategies.signal import Signal
from src.strategies.timeframe import MultiTimeframeAnalysis, MultiTimeframeData, Timeframe, TimeframeResult
from src.strategies.trend_following import TrendFollowingIndicators, TrendFollowingStrategy
from src.tools.models import ToolDefinition, ToolFunction, ToolParameter, ToolParametersSchema


class TechnicalLLMResponse(BaseModel):
    """LLM response structure for technical analysis."""

    interpretation: str = Field(description="Technical analysis interpretation")
    confidence_keywords: list[str] = Field(
        description="Confidence keywords: 'high confidence', 'strong signal', 'weak', 'uncertain', etc."
    )


_DECLINING_BARS_WARNING_THRESHOLD = 3


def _build_exhaustion_warnings(exhaust: ExhaustionSignals) -> list[str]:
    """Build human-readable exhaustion warning strings."""
    warnings: list[str] = []
    if exhaust.rsi_bearish_divergence:
        warnings.append("RSI bearish divergence detected")
    if exhaust.macd_hist_declining_bars >= _DECLINING_BARS_WARNING_THRESHOLD:
        warnings.append(f"MACD histogram declining {exhaust.macd_hist_declining_bars} bars")
    if exhaust.adx_turning_down:
        warnings.append("ADX turning down (trend weakening)")
    return warnings


StrategyType = MomentumStrategy | MeanReversionStrategy | TrendFollowingStrategy | EnsembleStrategy
IndicatorsType = MomentumIndicators | MeanReversionIndicators | TrendFollowingIndicators | EnsembleResult


class TechnicalWorker:
    """Technical analysis worker - Pydantic AI migration POC.

    This is a simplified implementation that establishes the worker pattern:
    - Stateless: strategy passed to analyze(), not stored in __init__
    - Compatible interface for future Pydantic AI Agent integration
    - Uses existing LLMClient for now (full Pydantic AI Agent integration deferred)
    """

    def __init__(self, llm_client: LLMClient) -> None:
        """Initialize technical worker.

        Args:
            llm_client: LLM client for generating interpretations
        """
        self.llm_client = llm_client
        self._prompts = PromptLoader("technical")
        logger.info("Initialized TechnicalWorker (stateless, POC)")

    async def analyze(
        self,
        symbol: str,
        market_data: pd.DataFrame | MultiTimeframeData,
        strategy: StrategyType | None,
        enable_multi_timeframe: bool = False,
    ) -> TechnicalAnalysis:
        """Perform technical analysis on market data.

        Args:
            symbol: Stock ticker symbol
            market_data: OHLCV dataframe or MultiTimeframeData
            strategy: Trading strategy for indicator generation
            enable_multi_timeframe: Enable multi-timeframe analysis

        Returns:
            TechnicalAnalysis with signal and interpretation
        """
        if strategy is None:
            msg = "strategy is required for technical analysis"
            raise ValueError(msg)

        # Multi-timeframe analysis path
        if isinstance(market_data, MultiTimeframeData) and enable_multi_timeframe:
            return await self._analyze_multi_timeframe(symbol, market_data, strategy)

        # Extract daily data for single-timeframe analysis
        if isinstance(market_data, MultiTimeframeData):
            if Timeframe.DAILY not in market_data.timeframes:
                msg = "Daily timeframe required for single-timeframe analysis"
                raise ValueError(msg)
            daily_data = market_data.timeframes[Timeframe.DAILY]
        else:
            daily_data = market_data

        logger.info(f"Analyzing {symbol} technicals with {type(strategy).__name__}")

        # Offload CPU-intensive indicator calculation to thread
        from collections.abc import Callable

        generate_signal = cast(
            "Callable[[pd.DataFrame], tuple[Signal, IndicatorsType]]",
            strategy.generate_signal,
        )
        signal, indicators = await asyncio.to_thread(generate_signal, daily_data)
        latest_close = float(daily_data["Close"].iloc[-1])

        # Build prompt
        prompt, system_prompt = self._build_prompt(symbol, latest_close, signal, indicators, strategy)

        # Try structured output with fallback
        try:
            llm_response = await self.llm_client.astructured(
                prompt, TechnicalLLMResponse, system=system_prompt, temperature=0.3, max_tokens=2048
            )
            interpretation = llm_response.interpretation
            confidence_keywords = llm_response.confidence_keywords
        except StructuredOutputError as e:
            logger.opt(exception=True).warning(f"Structured output failed, falling back: {e}")
            interpretation = await self.llm_client.acomplete(
                prompt, system=system_prompt, temperature=0.3, max_tokens=2048
            )
            confidence_keywords = []

        # Extract indicator values and calculate confidence
        rsi, macd_hist, atr_14, exhaust_score, exhaust_warnings, ensemble_result = (
            self._extract_indicator_values(indicators)
        )
        confidence = self._calculate_confidence_with_keywords(interpretation, indicators, confidence_keywords)

        logger.info(f"Technical analysis complete: {signal.value} (confidence={confidence:.2f})")

        adx_val = indicators.adx if isinstance(indicators, MomentumIndicators) else None

        return TechnicalAnalysis(
            signal=signal,
            rsi=rsi,
            macd_hist=macd_hist,
            atr_14=atr_14,
            adx=adx_val,
            exhaustion_score=exhaust_score,
            exhaustion_warnings=exhaust_warnings,
            interpretation=interpretation,
            confidence=confidence,
            ensemble_result=ensemble_result,
        )

    async def _analyze_multi_timeframe(
        self, symbol: str, multi_data: MultiTimeframeData, strategy: StrategyType
    ) -> TechnicalAnalysis:
        """Perform multi-timeframe analysis.

        Args:
            symbol: Stock ticker symbol
            multi_data: Multi-timeframe market data
            strategy: Trading strategy

        Returns:
            TechnicalAnalysis with multi-timeframe results
        """
        logger.info(f"Multi-timeframe analysis for {symbol} ({list(multi_data.timeframes.keys())})")

        timeframe_results: dict[Timeframe, TimeframeResult] = {}

        from collections.abc import Callable

        generate_signal = cast(
            "Callable[[pd.DataFrame], tuple[Signal, IndicatorsType]]",
            strategy.generate_signal,
        )

        for timeframe, data in multi_data.timeframes.items():
            # Offload CPU-intensive indicator calculation to thread
            signal, indicators = await asyncio.to_thread(generate_signal, data)
            latest_close = float(data["Close"].iloc[-1])

            prompt, system_prompt = self._build_prompt(symbol, latest_close, signal, indicators, strategy)

            try:
                llm_response = await self.llm_client.astructured(
                    prompt, TechnicalLLMResponse, system=system_prompt, temperature=0.3, max_tokens=2048
                )
                interpretation = llm_response.interpretation
                confidence_keywords = llm_response.confidence_keywords
            except StructuredOutputError as e:
                logger.opt(exception=True).warning(
                    f"Structured output failed for {timeframe}, falling back: {e}"
                )
                interpretation = await self.llm_client.acomplete(
                    prompt, system=system_prompt, temperature=0.3, max_tokens=2048
                )
                confidence_keywords = []

            rsi, macd_hist, _atr, _exhaust, _exhaust_w, _ = self._extract_indicator_values(indicators)
            confidence = self._calculate_confidence_with_keywords(
                interpretation, indicators, confidence_keywords
            )

            indicators_dict = {}
            if rsi is not None:
                indicators_dict["rsi"] = rsi
            if macd_hist is not None:
                indicators_dict["macd_hist"] = macd_hist

            timeframe_results[timeframe] = TimeframeResult(
                timeframe=timeframe,
                signal=signal,
                rsi=rsi,
                macd_hist=macd_hist,
                interpretation=interpretation,
                confidence=confidence,
                indicators=indicators_dict,
            )

            logger.debug(f"{timeframe}: {signal.value} (confidence={confidence:.2f})")

        # Calculate confluence
        calculator = ConfluenceCalculator()
        final_signal, confluence_score, conflict_detected = calculator.calculate_confluence(timeframe_results)

        primary_timeframe = calculator.select_dominant_timeframe(final_signal, timeframe_results)
        base_confidence = timeframe_results[primary_timeframe].confidence
        final_confidence = calculator.adjust_confidence(base_confidence, confluence_score)

        # Generate aggregated interpretation
        multi_interpretation = await self._generate_multi_timeframe_interpretation(
            symbol, timeframe_results, final_signal, confluence_score, conflict_detected
        )

        multi_analysis = MultiTimeframeAnalysis(
            signal=final_signal,
            confidence=final_confidence,
            confluence_score=confluence_score,
            timeframe_results=timeframe_results,
            primary_timeframe=primary_timeframe,
            conflict_detected=conflict_detected,
            interpretation=multi_interpretation,
        )

        primary_result = timeframe_results[primary_timeframe]

        logger.info(
            f"Multi-timeframe complete: {final_signal.value} "
            f"(confidence={final_confidence:.2f}, confluence={confluence_score:.2f})"
        )

        return TechnicalAnalysis(
            signal=final_signal,
            rsi=primary_result.rsi,
            macd_hist=primary_result.macd_hist,
            interpretation=multi_interpretation,
            confidence=final_confidence,
            ensemble_result=None,
            multi_timeframe=multi_analysis,
        )

    async def _generate_multi_timeframe_interpretation(
        self,
        symbol: str,
        timeframe_results: dict[Timeframe, TimeframeResult],
        final_signal: Signal,
        confluence_score: float,
        conflict_detected: bool,
    ) -> str:
        """Generate aggregated interpretation for multi-timeframe analysis.

        Args:
            symbol: Stock ticker symbol
            timeframe_results: Results for each timeframe
            final_signal: Final aggregated signal
            confluence_score: Confluence score (0-1)
            conflict_detected: Whether conflicting signals detected

        Returns:
            LLM-generated interpretation
        """
        timeframe_summaries = []
        for tf, result in timeframe_results.items():
            timeframe_summaries.append(
                f"{tf.value}: {result.signal.value} (confidence={result.confidence:.2f})\n"
                f"  Interpretation: {result.interpretation}"
            )

        prompt_vars = {
            "symbol": symbol,
            "final_signal": final_signal.value,
            "confluence_score": f"{confluence_score:.2f}",
            "conflict_detected": str(conflict_detected),
            "timeframe_summaries": "\n\n".join(timeframe_summaries),
        }

        system = self._prompts.load("system_multi_timeframe")
        user = self._prompts.load("user_multi_timeframe", **prompt_vars)

        try:
            return await self.llm_client.acomplete(user, system=system, temperature=0.3)
        except Exception as e:
            logger.opt(exception=True).error(f"Failed to generate multi-timeframe interpretation: {e}")
            raise

    def _build_prompt(
        self,
        symbol: str,
        latest_close: float,
        signal: Signal,
        indicators: IndicatorsType,
        strategy: StrategyType,
    ) -> tuple[str, str]:
        """Build appropriate prompt based on strategy type.

        Args:
            symbol: Stock ticker
            latest_close: Latest closing price
            signal: Trading signal
            indicators: Strategy indicators
            strategy: Trading strategy (for extracting configured parameters)

        Returns:
            Tuple of (user_prompt, system_prompt)
        """
        if isinstance(indicators, EnsembleResult):
            prompt_type = "ensemble"
            prompt_vars = self._build_ensemble_vars(symbol, latest_close, signal, indicators)
        elif isinstance(indicators, TrendFollowingIndicators) and isinstance(
            strategy, TrendFollowingStrategy
        ):
            prompt_type = "trend_following"
            prompt_vars = self._build_trend_following_vars(symbol, latest_close, signal, indicators, strategy)
        elif isinstance(indicators, MeanReversionIndicators):
            prompt_type = "mean_reversion"
            prompt_vars = self._build_mean_reversion_vars(symbol, latest_close, signal, indicators)
        elif isinstance(indicators, MomentumIndicators) and isinstance(strategy, MomentumStrategy):
            prompt_type = "momentum"
            prompt_vars = self._build_momentum_vars(symbol, latest_close, signal, indicators, strategy)
        else:
            msg = f"Unexpected indicators type: {type(indicators)}"
            raise TypeError(msg)

        system = self._prompts.load(f"system_{prompt_type}")
        user = self._prompts.load(f"user_{prompt_type}", **prompt_vars)
        return user, system

    def _extract_indicator_values(
        self, indicators: IndicatorsType
    ) -> tuple[float | None, float | None, float | None, float | None, list[str], EnsembleResult | None]:
        """Extract RSI, MACD, ATR, exhaustion, and ensemble result from indicators.

        Args:
            indicators: Strategy indicators

        Returns:
            Tuple of (rsi, macd_hist, atr_14, exhaustion_score, exhaustion_warnings, ensemble_result)
        """
        if isinstance(indicators, EnsembleResult):
            rsi, macd_hist, atr_14, exhaust_score = None, None, None, None
            exhaust_warnings: list[str] = []
            for sr in indicators.strategy_results:
                if sr.name == "momentum" and isinstance(sr.indicators, MomentumIndicators):
                    rsi = sr.indicators.rsi
                    macd_hist = sr.indicators.macd_hist
                    atr_14 = sr.indicators.atr_14
                    if sr.indicators.exhaustion:
                        exhaust_score = sr.indicators.exhaustion.exhaustion_score
                        exhaust_warnings = _build_exhaustion_warnings(sr.indicators.exhaustion)
                    break
            return rsi, macd_hist, atr_14, exhaust_score, exhaust_warnings, indicators
        if isinstance(indicators, MomentumIndicators):
            exhaust_score_val = None
            exhaust_warn: list[str] = []
            if indicators.exhaustion:
                exhaust_score_val = indicators.exhaustion.exhaustion_score
                exhaust_warn = _build_exhaustion_warnings(indicators.exhaustion)
            return (
                indicators.rsi,
                indicators.macd_hist,
                indicators.atr_14,
                exhaust_score_val,
                exhaust_warn,
                None,
            )
        return None, None, None, None, [], None

    def _calculate_confidence_with_keywords(
        self, interpretation: str, indicators: IndicatorsType, keywords: list[str]
    ) -> float:
        """Calculate confidence using structured keywords when available.

        Args:
            interpretation: LLM interpretation text
            indicators: Strategy indicators
            keywords: Confidence keywords from structured output

        Returns:
            Confidence score (0.0-1.0)
        """
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
        """Calculate base confidence for momentum indicators.

        Args:
            indicators: Momentum indicators

        Returns:
            Base confidence score
        """
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
        self,
        symbol: str,
        latest_close: float,
        signal: Signal,
        indicators: MomentumIndicators,
        strategy: MomentumStrategy,
    ) -> dict[str, str]:
        """Build variables for momentum prompt.

        Args:
            symbol: Stock ticker
            latest_close: Latest closing price
            signal: Trading signal
            indicators: Momentum indicators
            strategy: Momentum strategy (for configured parameters)

        Returns:
            Variables dictionary
        """
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
            "rsi_oversold": str(int(strategy.rsi_oversold)),
            "rsi_overbought": str(int(strategy.rsi_overbought)),
            "rsi_status": rsi_status,
            "macd": f"{indicators.macd:.4f}",
            "macd_signal": f"{indicators.macd_signal:.4f}",
            "macd_hist": f"{indicators.macd_hist:.4f}",
            "macd_trend": macd_trend,
            "signal": signal.value,
        }

    def _build_trend_following_vars(
        self,
        symbol: str,
        latest_close: float,
        signal: Signal,
        indicators: TrendFollowingIndicators,
        strategy: TrendFollowingStrategy,
    ) -> dict[str, str]:
        """Build variables for trend following prompt.

        Args:
            symbol: Stock ticker
            latest_close: Latest closing price
            signal: Trading signal
            indicators: Trend following indicators
            strategy: Trend following strategy (for configured parameters)

        Returns:
            Variables dictionary
        """
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
            "adx_threshold": str(int(strategy.adx_threshold)),
            "trend_strength": trend_strength,
            "plus_di": f"{indicators.plus_di:.2f}",
            "minus_di": f"{indicators.minus_di:.2f}",
            "trend_direction": indicators.trend_direction.upper(),
            "signal": signal.value,
        }

    def _build_mean_reversion_vars(
        self, symbol: str, latest_close: float, signal: Signal, indicators: MeanReversionIndicators
    ) -> dict[str, str]:
        """Build variables for mean reversion prompt.

        Args:
            symbol: Stock ticker
            latest_close: Latest closing price
            signal: Trading signal
            indicators: Mean reversion indicators

        Returns:
            Variables dictionary
        """
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

    def _extract_trend_confidence(self, indicators: TrendFollowingIndicators) -> float:
        """Calculate confidence for trend following indicators.

        Args:
            indicators: Trend following indicators

        Returns:
            Confidence score
        """
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
        """Calculate confidence for mean reversion indicators.

        Args:
            indicators: Mean reversion indicators

        Returns:
            Confidence score
        """
        confidence = 0.5

        if indicators.oversold or indicators.overbought:
            confidence += 0.25
        # Further from middle = higher confidence
        bb_deviation = abs(indicators.bb_percent - 0.5)
        confidence += min(bb_deviation * 0.3, 0.15)

        return min(confidence, 0.95)

    def get_tool_definition(self) -> ToolDefinition:
        """Get tool definition for supervisor integration.

        Returns:
            Tool definition
        """
        return ToolDefinition(
            type="function",
            function=ToolFunction(
                name="analyze_technical",
                description="Perform technical analysis using RSI/MACD/SMA indicators",
                parameters=ToolParametersSchema(
                    type="object",
                    properties={
                        "symbol": ToolParameter(type="string", description="Stock ticker symbol"),
                        "strategy": ToolParameter(
                            type="string",
                            enum=["momentum", "trend_following", "mean_reversion", "ensemble"],
                            description="Trading strategy to use",
                        ),
                        "enable_multi_timeframe": ToolParameter(
                            type="boolean",
                            default=False,
                            description="Enable multi-timeframe analysis",
                        ),
                    },
                    required=["symbol", "strategy"],
                ),
            ),
        )

    def __repr__(self) -> str:
        """String representation."""
        return "TechnicalWorker(pydantic_ai)"
