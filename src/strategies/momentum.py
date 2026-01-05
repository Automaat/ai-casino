"""Momentum strategy using RSI and MACD indicators."""

from enum import Enum

import pandas as pd
import pandas_ta  # noqa: F401 - Required to register .ta accessor on DataFrame
from loguru import logger
from pydantic import BaseModel


class Signal(str, Enum):
    """Trading signal."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


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

        df.ta.rsi(length=self.rsi_period, append=True)

        df.ta.macd(
            fast=self.macd_fast,
            slow=self.macd_slow,
            signal=self.macd_signal,
            append=True,
        )

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

        rsi_col = f"RSI_{self.rsi_period}"
        macd_col = f"MACD_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}"
        signal_col = f"MACDs_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}"
        hist_col = f"MACDh_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}"

        rsi = float(latest[rsi_col])
        macd = float(latest[macd_col])
        macd_signal_val = float(latest[signal_col])
        macd_hist = float(latest[hist_col])

        return MomentumIndicators(
            rsi=rsi,
            rsi_oversold=rsi < self.rsi_oversold,
            rsi_overbought=rsi > self.rsi_overbought,
            macd=macd,
            macd_signal=macd_signal_val,
            macd_hist=macd_hist,
            macd_bullish=macd > macd_signal_val,
            macd_bearish=macd < macd_signal_val,
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
