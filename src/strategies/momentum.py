"""Momentum strategy using RSI and MACD indicators."""

import pandas as pd
import pandas_ta_classic  # noqa: F401 - Required to register .ta accessor on DataFrame
from loguru import logger
from pydantic import BaseModel, Field

from src.metrics.execution import timed_operation
from src.strategies.signal import Signal


class ExhaustionSignals(BaseModel):
    """Momentum exhaustion detection signals."""

    rsi_bearish_divergence: bool = False
    macd_hist_declining_bars: int = 0
    adx_turning_down: bool = False
    exhaustion_score: float = Field(default=0.0, ge=0.0, le=1.0)


class MomentumIndicators(BaseModel):
    """Technical indicators for momentum strategy."""

    rsi: float
    rsi_oversold: bool
    rsi_overbought: bool
    macd: float
    macd_signal: float
    macd_hist: float
    macd_bullish: bool
    macd_bearish: bool
    atr_14: float | None = None
    adx: float | None = None
    exhaustion: ExhaustionSignals | None = None


_EXHAUSTION_MIN_BARS = 5
_EXHAUSTION_MIN_HIST_BARS = 3
_EXHAUSTION_LOOKBACK = 6
# Weights sum to 1.0 at max signal strength:
# divergence (0.4) + hist_bars * 0.12 capped at 3*0.12=0.36 + adx_down (0.24) = 1.0
_DIVERGENCE_WEIGHT = 0.4
_HIST_BAR_WEIGHT = 0.12
_ADX_DOWN_WEIGHT = 0.24


def _detect_rsi_divergence(data: pd.DataFrame) -> bool:
    """Detect RSI bearish divergence: price higher highs, RSI lower highs."""
    rsi_cols = [c for c in data.columns if c.startswith("RSI_")]
    if not rsi_cols:
        return False
    rsi_series = data[rsi_cols[0]].dropna()
    if len(rsi_series) < _EXHAUSTION_MIN_BARS:
        return False
    recent_close = data["Close"].iloc[-_EXHAUSTION_MIN_BARS:]
    recent_rsi = rsi_series.iloc[-_EXHAUSTION_MIN_BARS:]
    price_rising = float(recent_close.iloc[-1]) > float(recent_close.iloc[0])
    rsi_falling = float(recent_rsi.iloc[-1]) < float(recent_rsi.iloc[0])
    return price_rising and rsi_falling


def _count_declining_hist_bars(data: pd.DataFrame, hist_col: str) -> int:
    """Count consecutive declining MACD histogram bars from the end."""
    hist_series = data[hist_col].dropna()
    if len(hist_series) < _EXHAUSTION_MIN_HIST_BARS:
        return 0
    count = 0
    for i in range(len(hist_series) - 1, max(len(hist_series) - _EXHAUSTION_LOOKBACK, 0), -1):
        if float(hist_series.iloc[i]) < float(hist_series.iloc[i - 1]):
            count += 1
        else:
            break
    return count


def _detect_adx_turning_down(data: pd.DataFrame) -> bool:
    """Detect ADX turning down (trend weakening)."""
    adx_cols = [c for c in data.columns if c.startswith("ADX_")]
    if not adx_cols:
        return False
    adx_series = data[adx_cols[0]].dropna()
    if len(adx_series) < _EXHAUSTION_MIN_HIST_BARS:
        return False
    return float(adx_series.iloc[-1]) < float(adx_series.iloc[-2])


def detect_exhaustion(data: pd.DataFrame, hist_col: str) -> ExhaustionSignals:
    """Detect momentum exhaustion from existing indicator data.

    Args:
        data: DataFrame with calculated indicators
        hist_col: Name of the MACD histogram column

    Returns:
        ExhaustionSignals with detected warnings
    """
    if len(data) < _EXHAUSTION_MIN_BARS:
        return ExhaustionSignals()

    rsi_divergence = _detect_rsi_divergence(data)
    declining_bars = _count_declining_hist_bars(data, hist_col)
    adx_down = _detect_adx_turning_down(data)

    score = 0.0
    if rsi_divergence:
        score += _DIVERGENCE_WEIGHT
    score += min(declining_bars * _HIST_BAR_WEIGHT, _HIST_BAR_WEIGHT * _EXHAUSTION_MIN_HIST_BARS)
    if adx_down:
        score += _ADX_DOWN_WEIGHT
    score = min(score, 1.0)

    return ExhaustionSignals(
        rsi_bearish_divergence=rsi_divergence,
        macd_hist_declining_bars=declining_bars,
        adx_turning_down=adx_down,
        exhaustion_score=score,
    )


class MomentumStrategy:
    """Momentum trading strategy using RSI and MACD."""

    def __init__(
        self,
        rsi_period: int = 14,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
    ) -> None:
        """Initialize momentum strategy.

        Args:
            rsi_period: RSI calculation period
            rsi_oversold: RSI oversold threshold
            rsi_overbought: RSI overbought threshold
            macd_fast: MACD fast EMA period
            macd_slow: MACD slow EMA period
            macd_signal: MACD signal line period
        """
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal

        logger.info(
            f"Initialized MomentumStrategy: RSI={rsi_period}, MACD=({macd_fast},{macd_slow},{macd_signal})"
        )

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate RSI and MACD indicators.

        Args:
            data: OHLCV dataframe with 'Close' column

        Returns:
            DataFrame with added indicator columns
        """
        df = data.copy()

        with timed_operation("pandas_ta_indicators", rows=len(df)):
            df.ta.rsi(length=self.rsi_period, append=True)

            df.ta.macd(
                fast=self.macd_fast,
                slow=self.macd_slow,
                signal=self.macd_signal,
                append=True,
            )

            df.ta.atr(length=14, append=True)
            df.ta.adx(length=14, append=True)

        logger.debug(f"Calculated indicators for {len(df)} rows")
        return df

    def get_latest_indicators(self, data: pd.DataFrame) -> MomentumIndicators:
        """Get latest indicator values.

        Args:
            data: DataFrame with calculated indicators

        Returns:
            MomentumIndicators with latest values
        """
        latest = data.iloc[-1]

        def find_col(prefix: str) -> str:
            matches = [c for c in data.columns if c.startswith(prefix)]
            if not matches:
                msg = f"Column {prefix}* not found. Run calculate_indicators first."
                raise ValueError(msg)
            return matches[0]

        rsi_col = find_col(f"RSI_{self.rsi_period}")
        macd_col = find_col(f"MACD_{self.macd_fast}_{self.macd_slow}")
        signal_col = find_col(f"MACDs_{self.macd_fast}_{self.macd_slow}")
        hist_col = find_col(f"MACDh_{self.macd_fast}_{self.macd_slow}")

        rsi = float(latest[rsi_col])
        macd = float(latest[macd_col])
        macd_signal_val = float(latest[signal_col])
        macd_hist = float(latest[hist_col])

        atr_14: float | None = None
        atr_matches = [c for c in data.columns if c.startswith("ATRr_14")]
        if atr_matches:
            raw = latest[atr_matches[0]]
            if pd.notna(raw):
                atr_14 = float(raw)

        adx: float | None = None
        adx_matches = [c for c in data.columns if c.startswith("ADX_14")]
        if adx_matches:
            raw_adx = latest[adx_matches[0]]
            if pd.notna(raw_adx):
                adx = float(raw_adx)

        exhaustion = detect_exhaustion(data, hist_col)

        return MomentumIndicators(
            rsi=rsi,
            rsi_oversold=rsi < self.rsi_oversold,
            rsi_overbought=rsi > self.rsi_overbought,
            macd=macd,
            macd_signal=macd_signal_val,
            macd_hist=macd_hist,
            macd_bullish=macd > macd_signal_val,
            macd_bearish=macd < macd_signal_val,
            atr_14=atr_14,
            adx=adx,
            exhaustion=exhaustion,
        )

    def generate_signal(self, data: pd.DataFrame) -> tuple[Signal, MomentumIndicators]:
        """Generate trading signal based on momentum indicators.

        Args:
            data: OHLCV dataframe

        Returns:
            Tuple of (Signal, MomentumIndicators)
        """
        df = self.calculate_indicators(data)
        indicators = self.get_latest_indicators(df)

        if indicators.rsi_oversold and indicators.macd_bullish:
            signal = Signal.BUY
        elif indicators.rsi_overbought and indicators.macd_bearish:
            signal = Signal.SELL
        else:
            signal = Signal.HOLD

        logger.info(
            f"Signal: {signal.value} | RSI={indicators.rsi:.2f} | "
            f"MACD={indicators.macd:.4f} | MACD_Hist={indicators.macd_hist:.4f}"
        )

        return signal, indicators

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"MomentumStrategy(rsi_period={self.rsi_period}, "
            f"oversold={self.rsi_oversold}, overbought={self.rsi_overbought})"
        )
