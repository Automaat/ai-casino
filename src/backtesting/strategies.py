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
        if rsi_result is not None:
            self.rsi = self.I(lambda: rsi_result.values)
        else:
            self.rsi = self.I(lambda: pd.Series([50.0] * len(close_series)).values)

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


class TrendFollowingBacktestStrategy(Strategy):
    """Trend following strategy using SMA crossover and ADX.

    Tunable parameters:
        sma_fast: Fast SMA period (default: 50)
        sma_slow: Slow SMA period (default: 200)
        adx_period: ADX calculation period (default: 14)
        adx_threshold: ADX threshold for strong trend (default: 25)
    """

    sma_fast = 50
    sma_slow = 200
    adx_period = 14
    adx_threshold = 25.0

    def init(self) -> None:
        """Initialize indicators."""
        close_series = pd.Series(self.data.Close, name="Close")
        high_series = pd.Series(self.data.High, name="High")
        low_series = pd.Series(self.data.Low, name="Low")

        sma_fast_result = ta.sma(close_series, length=self.sma_fast)
        sma_slow_result = ta.sma(close_series, length=self.sma_slow)
        if sma_fast_result is not None:
            self.sma_fast_line = self.I(lambda: sma_fast_result.values)
        else:
            self.sma_fast_line = self.I(lambda: close_series.values)
        if sma_slow_result is not None:
            self.sma_slow_line = self.I(lambda: sma_slow_result.values)
        else:
            self.sma_slow_line = self.I(lambda: close_series.values)

        adx_result = ta.adx(high_series, low_series, close_series, length=self.adx_period)
        if adx_result is not None:
            self.adx = self.I(lambda: adx_result[f"ADX_{self.adx_period}"].values)
            self.plus_di = self.I(lambda: adx_result[f"DMP_{self.adx_period}"].values)
            self.minus_di = self.I(lambda: adx_result[f"DMN_{self.adx_period}"].values)
        else:
            zeros = pd.Series([0.0] * len(close_series)).values
            self.adx = self.I(lambda: zeros)
            self.plus_di = self.I(lambda: zeros)
            self.minus_di = self.I(lambda: zeros)

    def next(self) -> None:
        """Execute trading logic on each bar."""
        if len(self.data) < self.sma_slow:
            return

        sma_fast_val = self.sma_fast_line[-1]
        sma_slow_val = self.sma_slow_line[-1]
        adx_val = self.adx[-1]
        plus_di_val = self.plus_di[-1]
        minus_di_val = self.minus_di[-1]

        # Golden cross with strong trend
        bullish = sma_fast_val > sma_slow_val and adx_val > self.adx_threshold and plus_di_val > minus_di_val
        # Death cross with strong trend
        bearish = sma_fast_val < sma_slow_val and adx_val > self.adx_threshold and minus_di_val > plus_di_val

        if bullish and not self.position:
            self.buy()
        elif bearish and self.position:
            self.position.close()


class MeanReversionBacktestStrategy(Strategy):
    """Mean reversion strategy using Bollinger Bands.

    Tunable parameters:
        bb_period: Bollinger Bands period (default: 20)
        bb_std: Number of standard deviations (default: 2.0)
    """

    bb_period = 20
    bb_std = 2.0

    def init(self) -> None:
        """Initialize indicators."""
        close_series = pd.Series(self.data.Close, name="Close")

        bbands = ta.bbands(close_series, length=self.bb_period, std=self.bb_std)
        if bbands is not None:
            # Find actual columns (pandas-ta may format differently)
            lower_cols = [c for c in bbands.columns if c.startswith(f"BBL_{self.bb_period}")]
            upper_cols = [c for c in bbands.columns if c.startswith(f"BBU_{self.bb_period}")]

            if lower_cols and upper_cols:
                self.bb_lower = self.I(lambda: bbands[lower_cols[0]].values)
                self.bb_upper = self.I(lambda: bbands[upper_cols[0]].values)
            else:
                zeros = pd.Series([0.0] * len(close_series)).values
                self.bb_lower = self.I(lambda: zeros)
                self.bb_upper = self.I(lambda: zeros)
        else:
            zeros = pd.Series([0.0] * len(close_series)).values
            self.bb_lower = self.I(lambda: zeros)
            self.bb_upper = self.I(lambda: zeros)

    def next(self) -> None:
        """Execute trading logic on each bar."""
        if len(self.data) < self.bb_period:
            return

        close = self.data.Close[-1]
        bb_lower_val = self.bb_lower[-1]
        bb_upper_val = self.bb_upper[-1]

        # Buy when price touches lower band (oversold)
        if close < bb_lower_val and not self.position:
            self.buy()
        # Sell when price touches upper band (overbought)
        elif close > bb_upper_val and self.position:
            self.position.close()


class EnsembleBacktestStrategy(Strategy):
    """Ensemble strategy combining momentum, trend following, and mean reversion.

    Tunable parameters:
        momentum_weight: Weight for momentum signals (default: 0.4)
        mean_reversion_weight: Weight for mean reversion signals (default: 0.25)
        trend_following_weight: Weight for trend following signals (default: 0.35)
    """

    momentum_weight = 0.4
    mean_reversion_weight = 0.25
    trend_following_weight = 0.35
    ensemble_threshold = 0.3

    # Momentum params
    rsi_period = 14
    rsi_oversold = 30
    rsi_overbought = 70
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9

    # Trend following params (shorter periods for ensemble)
    sma_fast = 20
    sma_slow = 50
    adx_period = 14
    adx_threshold = 25.0

    # Mean reversion params
    bb_period = 20
    bb_std = 2.0

    def _init_momentum_indicators(self, close_series: pd.Series) -> None:
        """Initialize RSI and MACD indicators."""
        rsi_result = ta.rsi(close_series, length=self.rsi_period)
        if rsi_result is not None:
            self.rsi = self.I(lambda: rsi_result.values)
        else:
            self.rsi = self.I(lambda: pd.Series([50.0] * len(close_series)).values)

        macd = ta.macd(close_series, fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal)
        if macd is not None:
            self.macd_hist = self.I(
                lambda: macd[f"MACDh_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}"].values
            )
        else:
            self.macd_hist = self.I(lambda: pd.Series([0.0] * len(close_series)).values)

    def _init_trend_indicators(
        self, close_series: pd.Series, high_series: pd.Series, low_series: pd.Series
    ) -> None:
        """Initialize SMA and ADX indicators."""
        sma_fast_result = ta.sma(close_series, length=self.sma_fast)
        sma_slow_result = ta.sma(close_series, length=self.sma_slow)
        if sma_fast_result is not None:
            self.sma_fast_line = self.I(lambda: sma_fast_result.values)
        else:
            self.sma_fast_line = self.I(lambda: close_series.values)
        if sma_slow_result is not None:
            self.sma_slow_line = self.I(lambda: sma_slow_result.values)
        else:
            self.sma_slow_line = self.I(lambda: close_series.values)

        adx_result = ta.adx(high_series, low_series, close_series, length=self.adx_period)
        if adx_result is not None:
            self.adx = self.I(lambda: adx_result[f"ADX_{self.adx_period}"].values)
            self.plus_di = self.I(lambda: adx_result[f"DMP_{self.adx_period}"].values)
            self.minus_di = self.I(lambda: adx_result[f"DMN_{self.adx_period}"].values)
        else:
            zeros = pd.Series([0.0] * len(close_series)).values
            self.adx = self.I(lambda: zeros)
            self.plus_di = self.I(lambda: zeros)
            self.minus_di = self.I(lambda: zeros)

    def _init_mean_reversion_indicators(self, close_series: pd.Series) -> None:
        """Initialize Bollinger Bands indicators."""
        bbands = ta.bbands(close_series, length=self.bb_period, std=self.bb_std)
        if bbands is not None:
            lower_cols = [c for c in bbands.columns if c.startswith(f"BBL_{self.bb_period}")]
            upper_cols = [c for c in bbands.columns if c.startswith(f"BBU_{self.bb_period}")]

            if lower_cols and upper_cols:
                self.bb_lower = self.I(lambda: bbands[lower_cols[0]].values)
                self.bb_upper = self.I(lambda: bbands[upper_cols[0]].values)
            else:
                zeros = pd.Series([0.0] * len(close_series)).values
                self.bb_lower = self.I(lambda: zeros)
                self.bb_upper = self.I(lambda: zeros)
        else:
            zeros = pd.Series([0.0] * len(close_series)).values
            self.bb_lower = self.I(lambda: zeros)
            self.bb_upper = self.I(lambda: zeros)

    def init(self) -> None:
        """Initialize all indicators."""
        close_series = pd.Series(self.data.Close, name="Close")
        high_series = pd.Series(self.data.High, name="High")
        low_series = pd.Series(self.data.Low, name="Low")

        self._init_momentum_indicators(close_series)
        self._init_trend_indicators(close_series, high_series, low_series)
        self._init_mean_reversion_indicators(close_series)

    def _get_momentum_signal(self) -> int:
        """Get momentum signal: 1 (buy), -1 (sell), 0 (hold)."""
        rsi_val = self.rsi[-1]
        macd_hist_val = self.macd_hist[-1]

        if rsi_val < self.rsi_oversold and macd_hist_val > 0:
            return 1
        if rsi_val > self.rsi_overbought and macd_hist_val < 0:
            return -1
        return 0

    def _get_trend_signal(self) -> int:
        """Get trend following signal: 1 (buy), -1 (sell), 0 (hold)."""
        sma_fast_val = self.sma_fast_line[-1]
        sma_slow_val = self.sma_slow_line[-1]
        adx_val = self.adx[-1]
        plus_di_val = self.plus_di[-1]
        minus_di_val = self.minus_di[-1]

        bullish = sma_fast_val > sma_slow_val and adx_val > self.adx_threshold and plus_di_val > minus_di_val
        bearish = sma_fast_val < sma_slow_val and adx_val > self.adx_threshold and minus_di_val > plus_di_val

        if bullish:
            return 1
        if bearish:
            return -1
        return 0

    def _get_mean_reversion_signal(self) -> int:
        """Get mean reversion signal: 1 (buy), -1 (sell), 0 (hold)."""
        close = self.data.Close[-1]
        bb_lower_val = self.bb_lower[-1]
        bb_upper_val = self.bb_upper[-1]

        if close < bb_lower_val:
            return 1
        if close > bb_upper_val:
            return -1
        return 0

    def next(self) -> None:
        """Execute trading logic based on weighted ensemble."""
        min_period = max(self.sma_slow, self.macd_slow, self.bb_period)
        if len(self.data) < min_period:
            return

        # Get individual signals
        momentum_sig = self._get_momentum_signal()
        trend_sig = self._get_trend_signal()
        mean_rev_sig = self._get_mean_reversion_signal()

        # Weighted voting
        total_weight = self.momentum_weight + self.trend_following_weight + self.mean_reversion_weight
        weighted_signal = (
            momentum_sig * self.momentum_weight
            + trend_sig * self.trend_following_weight
            + mean_rev_sig * self.mean_reversion_weight
        ) / total_weight

        if weighted_signal > self.ensemble_threshold and not self.position:
            self.buy()
        elif weighted_signal < -self.ensemble_threshold and self.position:
            self.position.close()
