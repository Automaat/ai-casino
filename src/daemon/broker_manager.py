"""Broker lifecycle and watchlist management for daemon."""

from datetime import UTC, datetime

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from src.cache.historical import HistoricalCache
from src.daemon.config import DaemonConfig
from src.daemon.state import DaemonState
from src.data.broker import AlpacaBroker


class BrokerManager:
    """Manage broker lifecycle and watchlist composition."""

    def __init__(
        self,
        config: DaemonConfig,
        state: DaemonState,
        historical_cache: HistoricalCache,
    ) -> None:
        """Initialize broker manager.

        Args:
            config: Daemon configuration
            state: Daemon state
            historical_cache: Historical data cache
        """
        self.config = config
        self.state = state
        self._historical_cache = historical_cache
        self.broker: AlpacaBroker | None = None
        logger.info("BrokerManager initialized (broker not yet configured)")

    def __repr__(self) -> str:
        """Return string representation."""
        status = "configured" if self.broker else "not_configured"
        return f"BrokerManager(broker={status})"

    async def initialize_broker(self) -> None:
        """Initialize Alpaca broker based on config.

        Handles both auto_trade mode and watchlist-only mode.
        Sets paper trading start dates and validates credentials.

        Raises:
            ValueError: If credentials missing or invalid
        """
        from src.daemon.config import TradingMode

        if self.config.auto_trade:
            # Get credentials from config only
            if self.config.trading_mode == TradingMode.PAPER:
                api_key = self.config.api_keys.alpaca_paper_api_key or self.config.api_keys.alpaca_api_key
                secret_key = (
                    self.config.api_keys.alpaca_paper_secret_key or self.config.api_keys.alpaca_secret_key
                )
            else:
                api_key = self.config.api_keys.alpaca_api_key
                secret_key = self.config.api_keys.alpaca_secret_key

            if not api_key or not secret_key:
                if self.config.trading_mode == TradingMode.LIVE:
                    msg = "auto_trade with live mode requires alpaca_api_key/alpaca_secret_key in config"
                else:
                    msg = (
                        "auto_trade with paper mode requires "
                        "alpaca_paper_api_key/alpaca_paper_secret_key "
                        "or alpaca_api_key/alpaca_secret_key as a fallback in config"
                    )
                raise ValueError(msg)

            is_paper = self.config.trading_mode.value == "paper"
            self.broker = AlpacaBroker(
                api_key=api_key,
                secret_key=secret_key,
                paper=is_paper,
                historical_cache=self._historical_cache,
            )
            logger.info(f"Alpaca broker initialized (mode={self.config.trading_mode.value})")

            # Initialize paper trading start date
            if self.config.trading_mode.value == "paper":
                paper_start = await self.state.get_paper_trading_start_date()
                if paper_start is None:
                    await self.state.set_paper_trading_start_date(datetime.now(UTC))
                trading_mode = await self.state.get_current_trading_mode()
                if trading_mode != "paper":
                    await self.state.set_current_trading_mode("paper")
                    await self.state.set_paper_trading_start_date(datetime.now(UTC))
                    logger.warning("Switched to paper mode, reset start date")
        else:
            # Watchlist-only mode: Try to init broker for position merging if credentials present
            api_key = self.config.api_keys.alpaca_paper_api_key or self.config.api_keys.alpaca_api_key
            secret_key = (
                self.config.api_keys.alpaca_paper_secret_key or self.config.api_keys.alpaca_secret_key
            )

            if api_key and secret_key:
                try:
                    self.broker = AlpacaBroker(
                        api_key=api_key,
                        secret_key=secret_key,
                        paper=True,
                        historical_cache=self._historical_cache,
                    )
                    logger.info("Alpaca broker initialized for watchlist merging (read-only)")
                except Exception as e:
                    logger.warning(f"Failed to init broker for watchlist merging: {e}")
                    self.broker = None

    async def get_merged_watchlist(self, session: AsyncSession | None = None) -> list[str]:
        """Get watchlist merged with broker positions and screening candidates.

        Args:
            session: Optional database session for API requests

        Returns:
            Deduplicated list combining config watchlist, broker positions,
            and latest screening candidates. Config order is preserved,
            broker positions are appended in alphabetical order, and screening
            candidates are appended in the order of ``latest.top_symbols``
            (typically ordered by screening score/rank).
        """
        # Source 1: config watchlist (preserve order)
        merged_watchlist: list[str] = []
        seen: set[str] = set()

        for symbol in self.config.watchlist:
            if symbol not in seen:
                merged_watchlist.append(symbol)
                seen.add(symbol)

        # Source 2: broker positions
        self._merge_broker_positions(merged_watchlist, seen)

        # Source 3: pre-market candidates (7:00-9:30 AM ET)
        await self._merge_pre_market_candidates(merged_watchlist, seen, session=session)

        # Source 4: active discovery candidates (ordered by score)
        await self._merge_discovery_candidates(merged_watchlist, seen, session=session)

        return merged_watchlist

    def _merge_broker_positions(self, merged_watchlist: list[str], seen: set[str]) -> None:
        """Merge broker positions into watchlist."""
        if not self.broker:
            logger.debug("No broker configured, skipping position merge")
            return

        try:
            account_info = self.broker.get_account_info()
            position_symbols = set(account_info.positions.keys())

            if not position_symbols:
                logger.debug("No positions to merge")
                return

            added = position_symbols - seen
            if added:
                logger.info(f"Merged {len(added)} positions into watchlist: {sorted(added)}")
                merged_watchlist.extend(sorted(added))
                seen.update(added)
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to fetch positions for watchlist merge: {e}")

    async def _merge_pre_market_candidates(
        self, merged_watchlist: list[str], seen: set[str], session: AsyncSession | None = None
    ) -> None:
        """Merge pre-market screening candidates into watchlist (7:00-9:30 AM ET)."""
        if not self.config.pre_market.enabled:
            return

        from src.discovery.models import DiscoverySource

        active_candidates = await self.state.get_active_discovery_candidates(session=session)
        pre_market_symbols = [
            c.symbol
            for c in active_candidates
            if c.symbol not in seen and DiscoverySource.PRE_MARKET in c.sources
        ]

        if pre_market_symbols:
            logger.info(f"Merged {len(pre_market_symbols)} pre-market candidates: {pre_market_symbols}")
            merged_watchlist.extend(pre_market_symbols)
            seen.update(pre_market_symbols)

    async def _merge_discovery_candidates(
        self, merged_watchlist: list[str], seen: set[str], session: AsyncSession | None = None
    ) -> None:
        """Merge discovery candidates or screening results into watchlist."""
        active_candidates = await self.state.get_active_discovery_candidates(session=session)
        if self.config.discovery.enabled and active_candidates:
            # Expire stale candidates first
            expired = await self.state.expire_stale_candidates()
            if expired:
                logger.info(f"Expired {len(expired)} discovery candidates: {expired}")

            # Add active candidates (already sorted by score in discovery engine)
            # Refresh active candidates after expiration
            active_candidates = await self.state.get_active_discovery_candidates(session=session)
            discovery_symbols = [c.symbol for c in active_candidates if c.symbol not in seen]
            if discovery_symbols:
                logger.info(f"Merged {len(discovery_symbols)} discovery candidates: {discovery_symbols}")
                merged_watchlist.extend(discovery_symbols)
                seen.update(discovery_symbols)
        elif self.config.screening.enabled:
            screening_history = await self.state.get_screening_history(limit=1, session=session)
            if screening_history:
                # Fallback to old screening (backward compatible)
                latest = screening_history[-1]
                new_symbols = [s for s in latest.top_symbols if s not in seen]
                if new_symbols:
                    logger.info(f"Merged {len(new_symbols)} screening candidates: {new_symbols}")
                    merged_watchlist.extend(new_symbols)
                    seen.update(new_symbols)

    def is_available(self) -> bool:
        """Check if broker available.

        Returns:
            True if broker initialized
        """
        return self.broker is not None
