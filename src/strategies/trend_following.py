"""Trend following strategy using SMA crossover and ADX."""

import pandas as pd
import pandas_ta_classic  # noqa: F401 - Required to register .ta accessor on DataFrame
from loguru import logger
from pydantic import BaseModel

from src.strategies.signal import Signal

MIN_ROWS_FOR_CROSSOVER = 2


class TrendFollowingIndicators(BaseModel):
    """Technical indicators for trend following strategy."""

    close: float
    sma_fast: float
    sma_slow: float
    sma_bullish_cross: bool
    sma_bearish_cross: bool
    adx: float
    plus_di: float
    minus_di: float
    strong_trend: bool
    trend_direction: str  # "bullish", "bearish", "neutral"


class TrendFollowingStrategy:
    """Trend following strategy using SMA crossover and ADX."""

    def __init__(
        self,
        sma_fast: int = 50,
        sma_slow: int = 200,
        adx_period: int = 14,
        adx_threshold: float = 25.0,
        adx_threshold_weak: float | None = None,
    ) -> None:
        """Initialize trend following strategy.

        Args:
            sma_fast: Fast SMA period (default 50)
            sma_slow: Slow SMA period (default 200)
            adx_period: ADX calculation period (default 14)
            adx_threshold: ADX threshold for strong trend entries (>25 = strong)
            adx_threshold_weak: ADX threshold for exits (default: adx_threshold - 5)
        """
        self.sma_fast = sma_fast
        self.sma_slow = sma_slow
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.adx_threshold_weak = (
            adx_threshold_weak if adx_threshold_weak is not None else adx_threshold - 5.0
        )

        logger.info(
            f"Initialized TrendFollowingStrategy: SMA=({sma_fast},{sma_slow}), "
            f"ADX={adx_period} (strong>={adx_threshold}, weak>={self.adx_threshold_weak})"
        )

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate SMA and ADX indicators.

        Args:
            data: OHLCV dataframe with OHLC columns

        Returns:
            DataFrame with added indicator columns
        """
        df = data.copy()

        df.ta.sma(length=self.sma_fast, append=True)
        df.ta.sma(length=self.sma_slow, append=True)
        df.ta.adx(length=self.adx_period, append=True)

        logger.debug(f"Calculated indicators for {len(df)} rows")
        return df

    def get_latest_indicators(self, data: pd.DataFrame) -> TrendFollowingIndicators:
        """Get latest indicator values.

        Args:
            data: DataFrame with calculated indicators

        Returns:
            TrendFollowingIndicators with latest values
        """
        if len(data) < MIN_ROWS_FOR_CROSSOVER:
            msg = f"Need at least {MIN_ROWS_FOR_CROSSOVER} rows to detect crossovers"
            raise ValueError(msg)

        latest = data.iloc[-1]
        previous = data.iloc[-2]

        def find_col(prefix: str) -> str:
            matches = [c for c in data.columns if c.startswith(prefix)]
            if not matches:
                msg = f"Column {prefix}* not found. Run calculate_indicators first."
                raise ValueError(msg)
            return matches[0]

        sma_fast_col = find_col(f"SMA_{self.sma_fast}")
        sma_slow_col = find_col(f"SMA_{self.sma_slow}")
        adx_col = find_col(f"ADX_{self.adx_period}")
        dmp_col = find_col(f"DMP_{self.adx_period}")
        dmn_col = find_col(f"DMN_{self.adx_period}")

        close = float(latest["Close"])
        sma_fast_val = float(latest[sma_fast_col])
        sma_slow_val = float(latest[sma_slow_col])
        adx = float(latest[adx_col])
        plus_di = float(latest[dmp_col])
        minus_di = float(latest[dmn_col])

        # Detect crossovers
        prev_fast = float(previous[sma_fast_col])
        prev_slow = float(previous[sma_slow_col])
        sma_bullish_cross = prev_fast <= prev_slow and sma_fast_val > sma_slow_val
        sma_bearish_cross = prev_fast >= prev_slow and sma_fast_val < sma_slow_val

        # Trend direction based on DI
        if plus_di > minus_di:
            trend_direction = "bullish"
        elif minus_di > plus_di:
            trend_direction = "bearish"
        else:
            trend_direction = "neutral"

        return TrendFollowingIndicators(
            close=close,
            sma_fast=sma_fast_val,
            sma_slow=sma_slow_val,
            sma_bullish_cross=sma_bullish_cross,
            sma_bearish_cross=sma_bearish_cross,
            adx=adx,
            plus_di=plus_di,
            minus_di=minus_di,
            strong_trend=adx >= self.adx_threshold,
            trend_direction=trend_direction,
        )

    def generate_signal(self, data: pd.DataFrame) -> tuple[Signal, TrendFollowingIndicators]:
        """Generate trading signal based on trend following indicators.

        Signal logic:
        - BUY: SMA bullish crossover with bullish trend confirmation (via DI) OR
          (strong uptrend + price > fast SMA)
        - SELL: SMA bearish crossover with bearish trend confirmation (via DI) OR
          (strong downtrend + price < fast SMA)
        - HOLD: Otherwise

        Args:
            data: OHLCV dataframe

        Returns:
            Tuple of (Signal, TrendFollowingIndicators)
        """
        df = self.calculate_indicators(data)
        indicators = self.get_latest_indicators(df)

        # Golden cross (fast crosses above slow) with trend confirmation
        bullish_cross = indicators.sma_bullish_cross and indicators.trend_direction == "bullish"
        # Strong uptrend: ADX strong, +DI > -DI, price above fast SMA
        strong_uptrend = (
            indicators.strong_trend
            and indicators.trend_direction == "bullish"
            and indicators.close > indicators.sma_fast
        )

        # Death cross (fast crosses below slow) with trend confirmation
        bearish_cross = indicators.sma_bearish_cross and indicators.trend_direction == "bearish"
        # Strong downtrend: ADX strong, -DI > +DI, price below fast SMA
        strong_downtrend = (
            indicators.strong_trend
            and indicators.trend_direction == "bearish"
            and indicators.close < indicators.sma_fast
        )

        if bullish_cross or strong_uptrend:
            signal = Signal.BUY
        elif bearish_cross or strong_downtrend:
            signal = Signal.SELL
        else:
            signal = Signal.HOLD

        logger.info(
            f"Signal: {signal.value} | Close={indicators.close:.2f} | "
            f"SMA({self.sma_fast})={indicators.sma_fast:.2f} | "
            f"SMA({self.sma_slow})={indicators.sma_slow:.2f} | "
            f"ADX={indicators.adx:.2f} | Trend={indicators.trend_direction}"
        )

        return signal, indicators

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"TrendFollowingStrategy(sma_fast={self.sma_fast}, sma_slow={self.sma_slow}, "
            f"adx_period={self.adx_period}, adx_threshold={self.adx_threshold}, "
            f"adx_threshold_weak={self.adx_threshold_weak})"
        )
