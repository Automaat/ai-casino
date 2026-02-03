"""Technical Analysis Agent."""

import pandas as pd
from loguru import logger
from pydantic import BaseModel

from src.models.llm import LLMClient
from src.strategies.ensemble import EnsembleResult, EnsembleStrategy
from src.strategies.momentum import MomentumIndicators, MomentumStrategy, Signal


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

    def __init__(self, llm_client: LLMClient, strategy: MomentumStrategy | EnsembleStrategy) -> None:
        """Initialize technical analyst.

        Args:
            llm_client: LLM client for generating interpretations
            strategy: Momentum or Ensemble strategy for indicators
        """
        self.llm = llm_client
        self.strategy = strategy
        self._is_ensemble = isinstance(strategy, EnsembleStrategy)
        logger.info(f"Initialized TechnicalAnalyst (ensemble={self._is_ensemble})")

    async def analyze(self, symbol: str, market_data: pd.DataFrame) -> TechnicalAnalysis:
        """Perform technical analysis on market data.

        Args:
            symbol: Stock ticker symbol
            market_data: OHLCV dataframe

        Returns:
            TechnicalAnalysis with signal and interpretation
        """
        logger.info(f"Analyzing {symbol} technicals")

        signal, indicators = self.strategy.generate_signal(market_data)
        latest_close = float(market_data["Close"].iloc[-1])

        if self._is_ensemble:
            prompt = self._build_ensemble_prompt(symbol, latest_close, signal, indicators)
            system_prompt = (
                "You are a technical analyst specializing in multi-strategy ensemble analysis. "
                "Provide clear, actionable interpretations considering all strategy signals."
            )
        else:
            prompt = self._build_momentum_prompt(symbol, latest_close, signal, indicators)
            system_prompt = (
                "You are a technical analyst specializing in momentum indicators. "
                "Provide clear, actionable interpretations."
            )

        response = await self.llm.acomplete(prompt, system=system_prompt, temperature=0.3)

        if self._is_ensemble:
            confidence = indicators.confidence
            logger.info(f"Technical analysis complete: {signal.value} (confidence={confidence:.2f})")
            # Extract RSI/MACD from momentum sub-strategy for downstream agents
            rsi = None
            macd_hist = None
            for sr in indicators.strategy_results:
                if sr.name == "momentum":
                    rsi = sr.indicators.rsi
                    macd_hist = sr.indicators.macd_hist
                    break
            return TechnicalAnalysis(
                signal=signal,
                rsi=rsi,
                macd_hist=macd_hist,
                interpretation=response,
                confidence=confidence,
                ensemble_result=indicators,
            )

        confidence = self._extract_confidence(response, indicators)
        logger.info(f"Technical analysis complete: {signal.value} (confidence={confidence:.2f})")

        return TechnicalAnalysis(
            signal=signal,
            rsi=indicators.rsi,
            macd_hist=indicators.macd_hist,
            interpretation=response,
            confidence=confidence,
        )

    def _build_momentum_prompt(
        self, symbol: str, latest_close: float, signal: Signal, indicators: MomentumIndicators
    ) -> str:
        """Build LLM prompt for momentum strategy.

        Args:
            symbol: Stock ticker
            latest_close: Latest closing price
            signal: Generated signal
            indicators: Momentum indicators

        Returns:
            Prompt string
        """
        return f"""Analyze these technical indicators for {symbol}:

Current Price: ${latest_close:.2f}

RSI ({self.strategy.rsi_period}): {indicators.rsi:.2f}
- Oversold threshold: {self.strategy.rsi_oversold}
- Overbought threshold: {self.strategy.rsi_overbought}
- Status: {"OVERSOLD" if indicators.rsi_oversold else "OVERBOUGHT" if indicators.rsi_overbought else "NEUTRAL"}

MACD:
- MACD Line: {indicators.macd:.4f}
- Signal Line: {indicators.macd_signal:.4f}
- Histogram: {indicators.macd_hist:.4f}
- Trend: {"BULLISH" if indicators.macd_bullish else "BEARISH"}

Generated Signal: {signal.value}

Provide a concise 2-3 sentence interpretation of these indicators and their implications for trading.
Rate your confidence (0.0-1.0) based on indicator alignment.
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

    def _extract_confidence(
        self,
        response: str,
        indicators,  # noqa: ANN001
    ) -> float:
        """Calculate confidence score based on indicator alignment.

        Args:
            response: LLM response text
            indicators: MomentumIndicators

        Returns:
            Confidence score (0.0-1.0)
        """
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

    def __repr__(self) -> str:
        """String representation."""
        return f"TechnicalAnalyst(strategy={self.strategy})"
