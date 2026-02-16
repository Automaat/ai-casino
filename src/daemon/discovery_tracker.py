"""Discovery outcome tracking and source performance metrics calculation."""

from __future__ import annotations

import asyncio
from datetime import date, timedelta
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from src.daemon.state.models import DiscoverySourceMetrics
    from src.data.market import MarketDataFetcher
    from src.database.repositories.discovery import DiscoveryHistoryRepository
    from src.database.repositories.discovery_source_metrics import DiscoverySourceMetricsRepository


class DiscoveryOutcomeTracker:
    """Tracks discovery outcomes and calculates source performance metrics."""

    def __init__(
        self,
        market_fetcher: MarketDataFetcher | None,
        discovery_repo: DiscoveryHistoryRepository,
        metrics_repo: DiscoverySourceMetricsRepository,
    ) -> None:
        """Initialize discovery outcome tracker.

        Args:
            market_fetcher: Market data fetcher for price lookups (optional)
            discovery_repo: Discovery history repository
            metrics_repo: Discovery source metrics repository
        """
        self.market_fetcher = market_fetcher
        self.discovery_repo = discovery_repo
        self.metrics_repo = metrics_repo

    async def update_all_outcomes(self) -> dict[str, int]:
        """Update outcome prices for pending discoveries.

        Returns:
            Dict with counts: {updated_7d, updated_30d, failed}
        """
        if not self.market_fetcher:
            logger.warning("Market fetcher not available, skipping outcome updates")
            return {"updated_7d": 0, "updated_30d": 0, "failed": 0}

        logger.info("Starting discovery outcome updates")

        discoveries_7d = await self.discovery_repo.get_discoveries_needing_outcome(horizon_days=7)
        discoveries_30d = await self.discovery_repo.get_discoveries_needing_outcome(horizon_days=30)

        updated_7d = 0
        updated_30d = 0
        failed = 0

        for discovery in discoveries_7d:
            try:
                target_date = discovery.discovered_at + timedelta(days=7)
                price_7d = await self._fetch_price_at_date(discovery.symbol, target_date.date())

                if price_7d and discovery.price_at_discovery:
                    outcome_7d = (
                        (price_7d - discovery.price_at_discovery) / discovery.price_at_discovery
                    ) * 100
                    await self.discovery_repo.update_outcome_prices(
                        symbol=discovery.symbol,
                        discovered_at=discovery.discovered_at,
                        outcome_7d=outcome_7d,
                        outcome_30d=None,
                        price_at_discovery=discovery.price_at_discovery,
                    )
                    updated_7d += 1
            except Exception as e:
                logger.opt(exception=True).warning(f"Failed to update 7d outcome for {discovery.symbol}: {e}")
                failed += 1

        for discovery in discoveries_30d:
            try:
                target_date = discovery.discovered_at + timedelta(days=30)
                price_30d = await self._fetch_price_at_date(discovery.symbol, target_date.date())

                if price_30d and discovery.price_at_discovery:
                    outcome_30d = (
                        (price_30d - discovery.price_at_discovery) / discovery.price_at_discovery
                    ) * 100
                    await self.discovery_repo.update_outcome_prices(
                        symbol=discovery.symbol,
                        discovered_at=discovery.discovered_at,
                        outcome_7d=None,
                        outcome_30d=outcome_30d,
                        price_at_discovery=discovery.price_at_discovery,
                    )
                    updated_30d += 1
            except Exception as e:
                logger.opt(exception=True).warning(
                    f"Failed to update 30d outcome for {discovery.symbol}: {e}"
                )
                failed += 1

        logger.info(f"Outcome updates complete: {updated_7d} 7d, {updated_30d} 30d, {failed} failed")

        return {"updated_7d": updated_7d, "updated_30d": updated_30d, "failed": failed}

    async def calculate_daily_source_metrics(self, measurement_date: date) -> list[DiscoverySourceMetrics]:
        """Calculate performance metrics for all sources on a specific date.

        Args:
            measurement_date: Date to calculate metrics for

        Returns:
            List of calculated metrics per source
        """
        logger.info(f"Calculating source metrics for {measurement_date}")

        metrics_list = await self.metrics_repo.calculate_metrics_for_date(
            measurement_date=measurement_date,
            window_days=30,
        )

        for metrics in metrics_list:
            await self.metrics_repo.create_or_update_daily_metrics(
                source_type=metrics.source_type,
                measurement_date=measurement_date,
                metrics=metrics,
            )

        logger.info(f"Calculated metrics for {len(metrics_list)} sources on {measurement_date}")

        return metrics_list

    async def _fetch_price_at_date(self, symbol: str, target_date: date) -> float | None:
        """Fetch closing price on target date (handle weekends).

        Args:
            symbol: Stock ticker symbol
            target_date: Target date for price

        Returns:
            Closing price or None if unavailable
        """
        if not self.market_fetcher:
            return None

        try:
            market_data = await asyncio.to_thread(
                self.market_fetcher.fetch_daily,
                symbol,
                period_days=10,
            )

            df = market_data.data

            price_at_date = None
            search_date = target_date
            max_attempts = 5

            for _ in range(max_attempts):
                if search_date in df.index:
<<<<<<< HEAD
                    close_value = df.loc[search_date]["Close"]
                    # Pandas Series scalar extraction - cast to Any for type checker
                    scalar_value: Any = close_value.item() if hasattr(close_value, "item") else close_value
                    price_at_date = float(scalar_value)
=======
                    import numpy as np

                    close_value = df.loc[search_date]["Close"]
                    if hasattr(close_value, "item"):
                        price_at_date = float(close_value.item())
                    else:
                        price_at_date = float(np.asarray(close_value).item())
>>>>>>> c6a30cfbbf9252d562c45e1af14dd40a6efb19ad
                    break
                search_date -= timedelta(days=1)

            if price_at_date:
                logger.debug(f"Found price for {symbol} on {search_date}: ${price_at_date:.2f}")
                return price_at_date

            logger.warning(f"No price data found for {symbol} near {target_date}")
            return None

        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to fetch price for {symbol} on {target_date}: {e}")
            return None

    def __repr__(self) -> str:
        """Return string representation."""
        return "DiscoveryOutcomeTracker()"
