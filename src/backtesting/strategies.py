"""Backtesting strategy implementations."""

import pandas as pd
import pandas_ta_classic as ta  # type: ignore[import-untyped]

from backtesting import Strategy


class MomentumBacktestStrategy(Strategy):
    """Momentum-based backtesting strategy using RSI and MACD.

    Tunable parameters:
        rsi_period: RSI calculation period (default: 14)
        rsi_oversold: Oversold threshold (default: 30)
        rsi_overbought: Overbought threshold (default: 70)
        macd_fast: MACD fast period (default: 12)
        macd_slow: MACD slow period (default: 26)
        macd_signal: MACD signal period (default: 9)
    """

    rsi_period = 14
    rsi_oversold = 30
    rsi_overbought = 70
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9

    def init(self) -> None:
        """Initialize indicators."""
        close_series = pd.Series(self.data.Close, name="Close")

        rsi_result = ta.rsi(close_series, length=self.rsi_period)
        self.rsi = self.I(lambda: rsi_result.values)

        macd = ta.macd(close_series, fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal)
        if macd is not None:
            self.macd_hist = self.I(
                lambda: macd[f"MACDh_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}"].values
            )
        else:
            self.macd_hist = self.I(lambda: pd.Series([0.0] * len(close_series)).values)

    def next(self) -> None:
        """Execute trading logic on each bar."""
        if len(self.data) < max(self.rsi_period, self.macd_slow):
            return

        rsi_val = self.rsi[-1]
        macd_hist_val = self.macd_hist[-1]

        if rsi_val < self.rsi_oversold and macd_hist_val > 0 and not self.position:
            self.buy()
        elif rsi_val > self.rsi_overbought and macd_hist_val < 0 and self.position:
            self.position.close()
