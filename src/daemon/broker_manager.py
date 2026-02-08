"""Broker lifecycle and watchlist management for daemon."""

from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from src.daemon.config import DaemonConfig
    from src.daemon.state import DaemonState
    from src.data.broker import AlpacaBroker


class BrokerManager:
    """Manage broker lifecycle and watchlist composition."""

    def __init__(
        self,
        config: "DaemonConfig",
        state: "DaemonState",
    ) -> None:
        """Initialize broker manager.

        Args:
            config: Daemon configuration
            state: Daemon state
        """
        self.config = config
        self.state = state
        self.broker: AlpacaBroker | None = None
        logger.info("BrokerManager initialized (broker not yet configured)")

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

        # Source 3: latest screening candidates (ordered by score)
        if self.config.screening.enabled and self.state.screening_history:
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
