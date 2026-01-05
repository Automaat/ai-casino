"""Technical Analysis Agent."""

import pandas as pd
from loguru import logger
from pydantic import BaseModel

from src.models.llm import LLMClient
from src.strategies.momentum import MomentumStrategy, Signal


class TechnicalAnalysis(BaseModel):
    """Technical analysis result."""

    signal: Signal
    rsi: float
    macd_hist: float
    interpretation: str
    confidence: float


class TechnicalAnalyst:
    """Agent for technical analysis of price data."""

    def __init__(self, llm_client: LLMClient, strategy: MomentumStrategy) -> None:
        """Initialize technical analyst.

        Args:
            llm_client: LLM client for generating interpretations
            strategy: Momentum strategy for indicators
        """
        self.llm = llm_client
        self.strategy = strategy
        logger.info("Initialized TechnicalAnalyst")

    def analyze(self, symbol: str, market_data: pd.DataFrame) -> TechnicalAnalysis:
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

        prompt = f"""Analyze these technical indicators for {symbol}:

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

        system_prompt = (
            "You are a technical analyst specializing in momentum indicators. "
            "Provide clear, actionable interpretations."
        )

        response = self.llm.complete(prompt, system=system_prompt, temperature=0.3)

        confidence = self._extract_confidence(response, indicators)

        logger.info(f"Technical analysis complete: {signal.value} (confidence={confidence:.2f})")

        return TechnicalAnalysis(
            signal=signal,
            rsi=indicators.rsi,
            macd_hist=indicators.macd_hist,
            interpretation=response,
            confidence=confidence,
        )

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

        if (indicators.rsi_oversold and indicators.macd_bullish) or (indicators.rsi_overbought and indicators.macd_bearish):
            confidence = 0.8
        elif indicators.rsi_oversold or indicators.macd_bullish or indicators.rsi_overbought or indicators.macd_bearish:
            confidence = 0.6

        if "high confidence" in response.lower() or "strong signal" in response.lower():
            confidence = min(confidence + 0.1, 1.0)

        return confidence

    def __repr__(self) -> str:
        """String representation."""
        return f"TechnicalAnalyst(strategy={self.strategy})"
