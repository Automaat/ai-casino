"""Market regime detection using ADX and ATR indicators."""

from enum import StrEnum

import pandas as pd
import pandas_ta_classic  # noqa: F401 - Required to register .ta accessor on DataFrame
from loguru import logger
from pydantic import BaseModel


class MarketRegime(StrEnum):
    """Market regime classification."""

    TRENDING_BULLISH = "TRENDING_BULLISH"
    TRENDING_BEARISH = "TRENDING_BEARISH"
    RANGING = "RANGING"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"


class RegimeIndicators(BaseModel):
    """Technical indicators for regime detection."""

    adx: float
    plus_di: float
    minus_di: float
    atr: float
    atr_ratio: float
    bb_width: float


class RegimeAnalysis(BaseModel):
    """Market regime analysis result."""

    regime: MarketRegime
    indicators: RegimeIndicators
    confidence: float
    reasoning: str


MIN_ROWS_FOR_REGIME = 35


class MarketRegimeDetector:
    """Detect market regime using ADX, ATR, and Bollinger Bands."""

    ADX_TREND_THRESHOLD = 25.0
    ATR_HIGH_VOL_RATIO = 1.5
    ADX_PERIOD = 14
    ATR_PERIOD = 14
    ATR_MA_PERIOD = 20
    BB_PERIOD = 20
    BB_STD = 2.0

    def __init__(
        self,
        adx_threshold: float | None = None,
        atr_vol_ratio: float | None = None,
    ) -> None:
        """Initialize regime detector.

        Args:
            adx_threshold: ADX threshold for trending (default 25.0)
            atr_vol_ratio: ATR ratio threshold for high volatility (default 1.5)
        """
        self.adx_threshold = adx_threshold or self.ADX_TREND_THRESHOLD
        self.atr_vol_ratio = atr_vol_ratio or self.ATR_HIGH_VOL_RATIO

        logger.info(
            f"Initialized MarketRegimeDetector: adx_threshold={self.adx_threshold}, "
            f"atr_vol_ratio={self.atr_vol_ratio}"
        )

    def _calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate regime detection indicators.

        Args:
            data: OHLCV dataframe

        Returns:
            DataFrame with indicator columns
        """
        df = data.copy()

        df.ta.adx(length=self.ADX_PERIOD, append=True)
        df.ta.atr(length=self.ATR_PERIOD, append=True)
        df.ta.bbands(length=self.BB_PERIOD, std=self.BB_STD, append=True)

        atr_col = f"ATRr_{self.ATR_PERIOD}"
        df[f"ATR_MA_{self.ATR_MA_PERIOD}"] = df[atr_col].rolling(window=self.ATR_MA_PERIOD).mean()

        return df

    def _get_indicators(self, data: pd.DataFrame) -> RegimeIndicators:
        """Extract latest indicator values.

        Args:
            data: DataFrame with calculated indicators

        Returns:
            RegimeIndicators with latest values
        """
        latest = data.iloc[-1]

        def find_col(prefix: str) -> str:
            matches = [c for c in data.columns if c.startswith(prefix)]
            if not matches:
                msg = f"Column {prefix}* not found. Run _calculate_indicators first."
                raise ValueError(msg)
            return matches[0]

        adx_col = find_col(f"ADX_{self.ADX_PERIOD}")
        dmp_col = find_col(f"DMP_{self.ADX_PERIOD}")
        dmn_col = find_col(f"DMN_{self.ADX_PERIOD}")
        atr_col = find_col(f"ATRr_{self.ATR_PERIOD}")
        atr_ma_col = f"ATR_MA_{self.ATR_MA_PERIOD}"
        bb_width_col = find_col(f"BBB_{self.BB_PERIOD}")

        adx = float(latest[adx_col])
        plus_di = float(latest[dmp_col])
        minus_di = float(latest[dmn_col])
        atr = float(latest[atr_col])
        atr_ma = float(latest[atr_ma_col])
        bb_width = float(latest[bb_width_col])

        atr_ratio = atr / atr_ma if atr_ma > 0 else 1.0

        return RegimeIndicators(
            adx=adx,
            plus_di=plus_di,
            minus_di=minus_di,
            atr=atr,
            atr_ratio=atr_ratio,
            bb_width=bb_width,
        )

    def _classify_regime(self, indicators: RegimeIndicators) -> tuple[MarketRegime, str]:
        """Classify market regime based on indicators.

        Args:
            indicators: Calculated regime indicators

        Returns:
            Tuple of (MarketRegime, reasoning string)
        """
        adx = indicators.adx
        plus_di = indicators.plus_di
        minus_di = indicators.minus_di
        atr_ratio = indicators.atr_ratio

        # High volatility takes precedence
        if atr_ratio > self.atr_vol_ratio:
            reasoning = (
                f"High volatility detected: ATR ratio={atr_ratio:.2f} > {self.atr_vol_ratio}. "
                f"ADX={adx:.2f}, +DI={plus_di:.2f}, -DI={minus_di:.2f}"
            )
            return MarketRegime.HIGH_VOLATILITY, reasoning

        # Trending markets
        if adx >= self.adx_threshold:
            if plus_di > minus_di:
                reasoning = (
                    f"Bullish trend: ADX={adx:.2f} >= {self.adx_threshold}, "
                    f"+DI={plus_di:.2f} > -DI={minus_di:.2f}"
                )
                return MarketRegime.TRENDING_BULLISH, reasoning
            reasoning = (
                f"Bearish trend: ADX={adx:.2f} >= {self.adx_threshold}, "
                f"-DI={minus_di:.2f} > +DI={plus_di:.2f}"
            )
            return MarketRegime.TRENDING_BEARISH, reasoning

        # Ranging market
        reasoning = (
            f"Ranging market: ADX={adx:.2f} < {self.adx_threshold}. "
            f"+DI={plus_di:.2f}, -DI={minus_di:.2f}, ATR ratio={atr_ratio:.2f}"
        )
        return MarketRegime.RANGING, reasoning

    def _calculate_confidence(self, regime: MarketRegime, indicators: RegimeIndicators) -> float:
        """Calculate confidence in regime classification.

        Args:
            regime: Classified regime
            indicators: Regime indicators

        Returns:
            Confidence score (0.0-1.0)
        """
        if regime == MarketRegime.HIGH_VOLATILITY:
            # Higher ATR ratio = higher confidence
            excess = indicators.atr_ratio - self.atr_vol_ratio
            confidence = min(0.6 + excess * 0.2, 0.95)

        elif regime in (MarketRegime.TRENDING_BULLISH, MarketRegime.TRENDING_BEARISH):
            # Stronger ADX = higher confidence
            excess = indicators.adx - self.adx_threshold
            confidence = min(0.6 + excess * 0.01, 0.95)
            # DI divergence adds confidence
            di_diff = abs(indicators.plus_di - indicators.minus_di)
            confidence = min(confidence + di_diff * 0.005, 0.95)

        else:
            # Ranging: lower ADX = higher confidence in ranging
            below_threshold = self.adx_threshold - indicators.adx
            confidence = min(0.5 + below_threshold * 0.02, 0.85)

        return round(confidence, 2)

    def detect_regime(self, data: pd.DataFrame) -> RegimeAnalysis:
        """Detect market regime from price data.

        Args:
            data: OHLCV dataframe (minimum 35 rows required, 50+ recommended)

        Returns:
            RegimeAnalysis with regime classification and confidence
        """
        if len(data) < MIN_ROWS_FOR_REGIME:
            msg = f"Need at least {MIN_ROWS_FOR_REGIME} rows for regime detection, got {len(data)}"
            raise ValueError(msg)

        df = self._calculate_indicators(data)
        indicators = self._get_indicators(df)
        regime, reasoning = self._classify_regime(indicators)
        confidence = self._calculate_confidence(regime, indicators)

        logger.info(f"Regime: {regime.value} (confidence={confidence:.2f})")

        return RegimeAnalysis(
            regime=regime,
            indicators=indicators,
            confidence=confidence,
            reasoning=reasoning,
        )

    def __repr__(self) -> str:
        """String representation."""
        return f"MarketRegimeDetector(adx_threshold={self.adx_threshold}, atr_vol_ratio={self.atr_vol_ratio})"
