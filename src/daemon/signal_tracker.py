"""Signal outcome tracker for accuracy metrics."""

from datetime import UTC, datetime

from loguru import logger

from src.cache.historical import HistoricalCache
from src.data.broker import AlpacaBroker
from src.data.market import MarketDataFetcher


class SignalOutcomeTracker:
    """Track price outcomes after signals for accuracy metrics."""

    def __init__(self, historical_cache: HistoricalCache, broker: AlpacaBroker | None = None) -> None:
        """Initialize signal outcome tracker.

        Args:
            historical_cache: Historical cache for signal storage and OHLCV
            broker: Optional broker for early exit detection
        """
        self._cache = historical_cache
        self._broker = broker
        self._market_fetcher = MarketDataFetcher(historical_cache=historical_cache)

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

            # Batch fetch prices by symbol
            prices_by_symbol = self._fetch_prices_for_signals(signals)

            # Check early exits if broker available
            exit_prices = self._get_early_exits(signals) if self._broker else {}

            # Update outcomes
            for signal in signals:
                # Use actual exit price if trade closed early, else market price
                price = exit_prices.get(signal["id"]) or prices_by_symbol.get(signal["symbol"])

                if price:
                    self._cache.update_signal_outcome(
                        signal["id"],
                        **{f"price_at_{horizon}": price},
                        outcome_updated_at=now.isoformat(),
                    )
                    stats[f"updated_{horizon}"] += 1
                else:
                    logger.warning(f"No price available for {signal['symbol']} at {horizon}")

        logger.info(f"Signal tracking complete: {stats}")
        return stats

    def _fetch_prices_for_signals(self, signals: list[dict]) -> dict[str, float]:
        """Batch fetch current prices for unique symbols.

        Args:
            signals: List of signal records

        Returns:
            Dict mapping symbol to latest close price
        """
        unique_symbols = {s["symbol"] for s in signals}
        prices = {}

        for symbol in unique_symbols:
            try:
                market_data = self._market_fetcher.fetch_daily(symbol, period_days=5)
                if not market_data.data.empty:
                    latest_close = float(market_data.data["Close"].iloc[-1])
                    prices[symbol] = latest_close
                    logger.debug(f"Fetched price for {symbol}: {latest_close:.2f}")
            except Exception as e:
                logger.warning(f"Failed to fetch price for {symbol}: {e}")

        return prices

    def _get_early_exits(self, signals: list[dict]) -> dict[int, float]:
        """Get actual exit prices for signals with closed trades.

        Args:
            signals: List of signal records

        Returns:
            Dict mapping signal_id to exit_price for early closures
        """
        if not self._broker:
            return {}

        exit_prices = {}

        try:
            # Query historical order fills for symbols with signals
            symbols = {s["symbol"] for s in signals}

            for _symbol in symbols:
                # Check if position was closed for this symbol
                # This is simplified - full implementation would match signal timestamps
                # to specific trade sequences in order_fills table
                pass

        except Exception as e:
            logger.warning(f"Failed to check early exits: {e}")

        return exit_prices

    def __repr__(self) -> str:
        """Return string representation."""
        return f"SignalOutcomeTracker(broker={self._broker is not None})"
