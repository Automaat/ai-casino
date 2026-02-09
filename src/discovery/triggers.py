"""Real-time market triggers for stock discovery."""

import pandas as pd
from loguru import logger

from src.data.market import MarketDataFetcher

# Data quality thresholds (minimum data points required)
MIN_VOLUME_DATA_POINTS = 21
MIN_GAP_DATA_POINTS = 2
MIN_ATR_DATA_POINTS = 35


class TriggerDetector:
    """Detect real-time market triggers (volume spikes, gaps, ATR anomalies)."""

    def __init__(
        self,
        market_fetcher: MarketDataFetcher,
        volume_spike_threshold: float = 2.0,
        price_gap_threshold: float = 5.0,
        atr_spike_threshold: float = 1.5,
    ) -> None:
        """Initialize trigger detector with thresholds."""
        self.market_fetcher = market_fetcher
        self.volume_spike_threshold = volume_spike_threshold
        self.price_gap_threshold = price_gap_threshold
        self.atr_spike_threshold = atr_spike_threshold
        logger.info(
            f"Initialized TriggerDetector (volume={volume_spike_threshold}x, "
            f"gap={price_gap_threshold}%, atr={atr_spike_threshold}x)"
        )

    def detect_volume_spikes(self, universe: list[str]) -> list[str]:
        """Detect stocks with volume >threshold * average.

        Args:
            universe: List of symbols to scan

        Returns:
            Symbols with volume spikes
        """
        spikes: list[str] = []

        for symbol in universe:
            try:
                # Fetch recent daily data (30 days for baseline)
                market_data = self.market_fetcher.fetch_daily(symbol, period_days=30)
                df = market_data.data

                if len(df) < MIN_VOLUME_DATA_POINTS:
                    continue

                # Calculate 20-day average volume
                avg_volume = df["Volume"].iloc[-21:-1].mean()
                current_volume = df["Volume"].iloc[-1]

                if current_volume > avg_volume * self.volume_spike_threshold:
                    spikes.append(symbol)
                    logger.debug(
                        f"{symbol}: volume spike {current_volume:,.0f} vs avg {avg_volume:,.0f} "
                        f"({current_volume / avg_volume:.1f}x)"
                    )

            except Exception as e:
                logger.warning(f"Volume spike detection failed for {symbol}: {e}")
                continue

        logger.info(f"Volume spike detection: {len(spikes)} candidates from {len(universe)} symbols")
        return spikes

    def detect_price_gaps(self, universe: list[str]) -> list[str]:
        """Detect stocks with gap >threshold %.

        Args:
            universe: List of symbols to scan

        Returns:
            Symbols with significant price gaps
        """
        gaps: list[str] = []

        for symbol in universe:
            try:
                # Fetch recent data
                market_data = self.market_fetcher.fetch_daily(symbol, period_days=5)
                df = market_data.data

                if len(df) < MIN_GAP_DATA_POINTS:
                    continue

                # Compare previous close to current open/price
                prev_close = df["Close"].iloc[-2]
                current_price = df["Open"].iloc[-1]

                gap_pct = abs((current_price - prev_close) / prev_close) * 100

                if gap_pct > self.price_gap_threshold:
                    gaps.append(symbol)
                    logger.debug(
                        f"{symbol}: price gap {gap_pct:.1f}% (${prev_close:.2f} -> ${current_price:.2f})"
                    )

            except Exception as e:
                logger.warning(f"Price gap detection failed for {symbol}: {e}")
                continue

        logger.info(f"Price gap detection: {len(gaps)} candidates from {len(universe)} symbols")
        return gaps

    async def detect_atr_anomalies(self, universe: list[str]) -> list[str]:
        """Detect stocks with ATR spike >threshold * average.

        Args:
            universe: List of symbols to scan

        Returns:
            Symbols with ATR anomalies
        """
        anomalies: list[str] = []

        for symbol in universe:
            try:
                # Fetch data for ATR calculation (need ~35 bars)
                market_data = self.market_fetcher.fetch_daily(symbol, period_days=50)
                df = market_data.data

                if len(df) < MIN_ATR_DATA_POINTS:
                    continue

                # Calculate ATR(14)
                df.ta.atr(length=14, append=True)  # type: ignore[attr-defined]

                if "ATR_14" not in df.columns or df["ATR_14"].isna().all():
                    continue

                # Get 20-day average ATR vs current
                avg_atr = df["ATR_14"].iloc[-21:-1].mean()
                current_atr = df["ATR_14"].iloc[-1]

                if pd.isna(avg_atr) or pd.isna(current_atr):
                    continue

                if current_atr > avg_atr * self.atr_spike_threshold:
                    anomalies.append(symbol)
                    logger.debug(
                        f"{symbol}: ATR spike {current_atr:.2f} vs avg {avg_atr:.2f} "
                        f"({current_atr / avg_atr:.1f}x)"
                    )

            except Exception as e:
                logger.warning(f"ATR anomaly detection failed for {symbol}: {e}")
                continue

        logger.info(f"ATR anomaly detection: {len(anomalies)} candidates from {len(universe)} symbols")
        return anomalies

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"TriggerDetector(volume={self.volume_spike_threshold}x, "
            f"gap={self.price_gap_threshold}%, atr={self.atr_spike_threshold}x)"
        )
