"""Broker lifecycle and watchlist management for daemon."""

import os
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from loguru import logger

from src.data.broker import AlpacaBroker

if TYPE_CHECKING:
    from src.cache.historical import HistoricalCache
    from src.daemon.config import DaemonConfig
    from src.daemon.state import DaemonState


class BrokerManager:
    """Manage broker lifecycle and watchlist composition."""

    def __init__(
        self,
        config: "DaemonConfig",
        state: "DaemonState",
        historical_cache: "HistoricalCache",
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

    def initialize_broker(self) -> None:
        """Initialize Alpaca broker based on config.

        Handles both auto_trade mode and watchlist-only mode.
        Sets paper trading start dates and validates credentials.

        Raises:
            ValueError: If credentials missing or invalid
        """
        from src.daemon.config import TradingMode

        if self.config.auto_trade:
            # Resolve credentials with config priority
            if self.config.trading_mode == TradingMode.PAPER:
                api_key = (
                    self.config.api_keys.alpaca_paper_api_key
                    or os.getenv("ALPACA_PAPER_API_KEY")
                    or self.config.api_keys.alpaca_api_key
                    or os.getenv("ALPACA_API_KEY")
                )
                secret_key = (
                    self.config.api_keys.alpaca_paper_secret_key
                    or os.getenv("ALPACA_PAPER_SECRET_KEY")
                    or self.config.api_keys.alpaca_secret_key
                    or os.getenv("ALPACA_SECRET_KEY")
                )
            else:
                api_key = self._resolve_config_or_env(self.config.api_keys.alpaca_api_key, "ALPACA_API_KEY")
                secret_key = self._resolve_config_or_env(
                    self.config.api_keys.alpaca_secret_key, "ALPACA_SECRET_KEY"
                )

            if not api_key or not secret_key:
                if self.config.trading_mode == TradingMode.LIVE:
                    msg = "auto_trade with live mode requires ALPACA_API_KEY/ALPACA_SECRET_KEY"
                else:
                    msg = (
                        "auto_trade with paper mode requires "
                        "ALPACA_PAPER_API_KEY/ALPACA_PAPER_SECRET_KEY "
                        "or ALPACA_API_KEY/ALPACA_SECRET_KEY as a fallback"
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
                if self.state.paper_trading_start_date is None:
                    self.state.paper_trading_start_date = datetime.now(UTC)
                if self.state.current_trading_mode != "paper":
                    self.state.current_trading_mode = "paper"
                    self.state.paper_trading_start_date = datetime.now(UTC)
                    logger.warning("Switched to paper mode, reset start date")
        elif os.getenv("ALPACA_API_KEY") and os.getenv("ALPACA_SECRET_KEY"):
            try:
                self.broker = AlpacaBroker(paper=True, historical_cache=self._historical_cache)
                logger.info("Alpaca broker initialized for watchlist merging")
            except Exception as e:
                logger.exception(f"Failed to initialize broker: {e}")
                self.broker = None

    def _resolve_config_or_env(self, config_value: str | None, env_var: str) -> str | None:
        """Resolve config value from daemon config or env var.

        Config takes priority over environment variable.

        Args:
            config_value: Value from daemon config (priority)
            env_var: Environment variable name (fallback)

        Returns:
            Resolved config value or None
        """
        return config_value or os.getenv(env_var)

    def get_merged_watchlist(self) -> list[str]:
        """Get watchlist merged with broker positions and screening candidates.

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
        if self.broker:
            try:
                account_info = self.broker.get_account_info()
                position_symbols = set(account_info.positions.keys())

                if position_symbols:
                    added = position_symbols - seen
                    if added:
                        logger.info(f"Merged {len(added)} positions into watchlist: {sorted(added)}")
                        merged_watchlist.extend(sorted(added))
                        seen.update(added)
                else:
                    logger.debug("No positions to merge")
            except Exception as e:
                logger.warning(f"Failed to fetch positions for watchlist merge: {e}")
        else:
            logger.debug("No broker configured, skipping position merge")

        # Source 3: active discovery candidates (ordered by score)
        if self.config.discovery.enabled and self.state.active_discovery_candidates:
            # Expire stale candidates first
            expired = self.state.expire_stale_candidates(self.config.discovery.candidate_ttl_days)
            if expired:
                logger.info(f"Expired {len(expired)} discovery candidates: {expired}")

            # Add active candidates (already sorted by score in discovery engine)
            discovery_symbols = [
                c.symbol for c in self.state.active_discovery_candidates if c.symbol not in seen
            ]
            if discovery_symbols:
                logger.info(f"Merged {len(discovery_symbols)} discovery candidates: {discovery_symbols}")
                merged_watchlist.extend(discovery_symbols)
                seen.update(discovery_symbols)
        elif self.config.screening.enabled and self.state.screening_history:
            # Fallback to old screening (backward compatible)
            latest = self.state.screening_history[-1]
            new_symbols = [s for s in latest.top_symbols if s not in seen]
            if new_symbols:
                logger.info(f"Merged {len(new_symbols)} screening candidates: {new_symbols}")
                merged_watchlist.extend(new_symbols)
                seen.update(new_symbols)

        return merged_watchlist

    def is_available(self) -> bool:
        """Check if broker available.

        Returns:
            True if broker initialized
        """
        return self.broker is not None
