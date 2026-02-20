"""Signal outcome tracker for accuracy metrics."""

from datetime import UTC, datetime

from loguru import logger
from pandas.tseries.offsets import BDay

from src.cache.historical import HistoricalCache
from src.data.market import MarketDataFetcher
from src.metrics.models import SignalUpdateRecord
from src.v1.trades.brokers import Broker


class SignalOutcomeTracker:
    """Track price outcomes after signals for accuracy metrics."""

    def __init__(
        self,
        historical_cache: HistoricalCache,
        market_fetcher: MarketDataFetcher,
        broker: Broker | None = None,
    ) -> None:
        """Initialize signal outcome tracker.

        Args:
            historical_cache: Historical cache for signal storage and OHLCV
            market_fetcher: MarketDataFetcher for price lookups
            broker: Optional broker for early exit detection
        """
        self._cache = historical_cache
        self._broker = broker
        self._market_fetcher = market_fetcher

    def update_outcomes(self) -> dict[str, int]:
        """Batch update outcomes for signals needing T+1d/5d/20d prices.

        Returns:
            Stats dict with updated counts per horizon
        """
        now = datetime.now(UTC)
        stats = {"updated_1d": 0, "updated_5d": 0, "updated_20d": 0}

        for horizon in ["1d", "5d", "20d"]:
            signals = self._cache.get_signals_needing_update(horizon)
            if not signals:
                logger.debug(f"No signals needing {horizon} update")
                continue

            logger.info(f"Updating {len(signals)} signals for {horizon} horizon")

            # Check early exits if broker available
            exit_prices = self._get_early_exits(signals) if self._broker else {}

            # Update outcomes
            for signal in signals:
                exit_price = exit_prices.get(signal.id)

                if exit_price is not None:
                    # Early exit: write both actual_exit_price and price_at_{horizon}
                    self._cache.update_signal_outcome(
                        signal.id,
                        actual_exit_price=exit_price,
                        **{f"price_at_{horizon}": exit_price},
                        outcome_updated_at=now.isoformat(),
                    )
                    stats[f"updated_{horizon}"] += 1
                else:
                    # Normal case: fetch price at target trading date
                    target_price = self._fetch_price_at_target_date(signal, horizon)

                    if target_price is not None:
                        self._cache.update_signal_outcome(
                            signal.id,
                            **{f"price_at_{horizon}": target_price},
                            outcome_updated_at=now.isoformat(),
                        )
                        stats[f"updated_{horizon}"] += 1
                    else:
                        logger.warning(f"No price available for {signal.symbol} at {horizon}")

        logger.info(f"Signal tracking complete: {stats}")
        return stats

    def _fetch_price_at_target_date(self, signal: SignalUpdateRecord, horizon: str) -> float | None:
        """Fetch close price at target trading date for a signal.

        Args:
            signal: Signal record with timestamp and symbol
            horizon: Time horizon (1d/5d/20d)

        Returns:
            Close price at target date or None if unavailable
        """
        trading_days = {"1d": 1, "5d": 5, "20d": 20}[horizon]
        signal_timestamp = datetime.fromisoformat(signal.timestamp)

        # Calculate target trading date (signal date + N business days)
        target_date = signal_timestamp + BDay(trading_days)

        try:
            # Fetch enough historical data to cover target date
            # Add buffer days to account for market closures
            lookback_days = trading_days + 10

            market_data = self._market_fetcher.fetch_daily(signal.symbol, period_days=lookback_days)

            if market_data.data.empty:
                return None

            # Find close at target date (or nearest prior trading day)
            import pandas as pd

            df = market_data.data
            if isinstance(df.index, pd.DatetimeIndex):
                df.index = df.index.tz_localize(None)  # Remove timezone for comparison
            target_ts = pd.Timestamp(target_date)
            target_date_normalized = target_ts.normalize()

            # Get closest date on or before target
            valid_dates = df.index[df.index <= target_date_normalized]

            if len(valid_dates) == 0:
                return None

            closest_date = valid_dates[-1]
            close_price = float(df.loc[closest_date, "Close"])

            logger.debug(
                f"Fetched {horizon} price for {signal.symbol}: {close_price:.2f} at {closest_date.date()}"
            )

            return close_price

        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to fetch {horizon} price for {signal.symbol}: {e}")
            return None

    def _get_early_exits(self, signals: list[SignalUpdateRecord]) -> dict[int, float]:
        """Get actual exit prices for signals with closed trades.

        Args:
            signals: List of signal records

        Returns:
            Dict mapping signal_id to exit_price for early closures
        """
        if not self._broker:
            return {}

        # Early exit detection not yet implemented
        symbol_count = len({s.symbol for s in signals})
        logger.debug(f"Early exit detection skipped (not implemented) for {symbol_count} symbols")
        return {}

    def __repr__(self) -> str:
        """Return string representation."""
        return f"SignalOutcomeTracker(broker={self._broker is not None})"
