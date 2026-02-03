"""Mean reversion strategy using Bollinger Bands."""

import pandas as pd
import pandas_ta  # noqa: F401 - Required to register .ta accessor on DataFrame
from loguru import logger
from pydantic import BaseModel

from src.strategies.momentum import Signal


class MeanReversionIndicators(BaseModel):
    """Technical indicators for mean reversion strategy."""

    close: float
    bb_upper: float
    bb_middle: float
    bb_lower: float
    bb_width: float
    bb_percent: float
    oversold: bool
    overbought: bool


class MeanReversionStrategy:
    """Mean reversion trading strategy using Bollinger Bands."""

    def __init__(self, bb_period: int = 20, bb_std: float = 2.0) -> None:
        """Initialize mean reversion strategy.

        Args:
            bb_period: Bollinger Bands moving average period
            bb_std: Number of standard deviations for bands
        """
        self.bb_period = bb_period
        self.bb_std = bb_std

        logger.info(f"Initialized MeanReversionStrategy: BB=({bb_period}, {bb_std})")

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Bands indicators.

        Args:
            data: OHLCV dataframe with 'Close' column

        Returns:
            DataFrame with added indicator columns
        """
        df = data.copy()

        df.ta.bbands(length=self.bb_period, std=self.bb_std, append=True)

        logger.debug(f"Calculated indicators for {len(df)} rows")
        return df

    def get_latest_indicators(self, data: pd.DataFrame) -> MeanReversionIndicators:
        """Get latest indicator values.

        Args:
            data: DataFrame with calculated indicators

        Returns:
            MeanReversionIndicators with latest values
        """
        latest = data.iloc[-1]

        # pandas-ta bbands uses format: BB{L,M,U,B,P}_{length}_{lower_std}_{upper_std}
        suffix = f"{self.bb_period}_{self.bb_std}_{self.bb_std}"
        lower_col = f"BBL_{suffix}"
        middle_col = f"BBM_{suffix}"
        upper_col = f"BBU_{suffix}"
        width_col = f"BBB_{suffix}"
        percent_col = f"BBP_{suffix}"

        close = float(latest["Close"])
        bb_lower = float(latest[lower_col])
        bb_upper = float(latest[upper_col])

        return MeanReversionIndicators(
            close=close,
            bb_upper=bb_upper,
            bb_middle=float(latest[middle_col]),
            bb_lower=bb_lower,
            bb_width=float(latest[width_col]),
            bb_percent=float(latest[percent_col]),
            oversold=close < bb_lower,
            overbought=close > bb_upper,
        )

    def generate_signal(self, data: pd.DataFrame) -> tuple[Signal, MeanReversionIndicators]:
        """Generate trading signal based on Bollinger Bands.

        Args:
            data: OHLCV dataframe

        Returns:
            Tuple of (Signal, MeanReversionIndicators)
        """
        df = self.calculate_indicators(data)
        indicators = self.get_latest_indicators(df)

        if indicators.oversold:
            signal = Signal.BUY
        elif indicators.overbought:
            signal = Signal.SELL
        else:
            signal = Signal.HOLD

        logger.info(
            f"Signal: {signal.value} | Close={indicators.close:.2f} | "
            f"BB=[{indicators.bb_lower:.2f}, {indicators.bb_middle:.2f}, {indicators.bb_upper:.2f}]"
        )

        return signal, indicators

    def __repr__(self) -> str:
        """String representation."""
        return f"MeanReversionStrategy(bb_period={self.bb_period}, bb_std={self.bb_std})"
