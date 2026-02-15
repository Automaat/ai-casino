"""Daemon integration for sector attribution analysis."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from loguru import logger

from src.daemon.state.models import SectorAttributionRecord
from src.metrics.sector_attribution import SectorAttributionAnalyzer

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.daemon.state.facade import DaemonState
    from src.data.broker import AlpacaBroker


class DaemonSectorAttribution:
    """Daemon wrapper for sector attribution analysis."""

    def __init__(self, state: DaemonState, broker: AlpacaBroker) -> None:
        """Initialize daemon sector attribution.

        Args:
            state: Daemon state facade
            broker: Broker for fetching account info
        """
        self._state = state
        self._broker = broker
        self._analyzer = SectorAttributionAnalyzer()
        logger.info("Initialized DaemonSectorAttribution")

    async def run(self, session: AsyncSession) -> None:
        """Run sector attribution analysis and store results.

        Args:
            session: Database session for storing results
        """
        logger.info("Running sector attribution analysis")

        positions = await self._state.get_all_positions()

        if not positions:
            logger.info("No positions to analyze")
            return

        account_info = await asyncio.to_thread(self._broker.get_account_info)
        broker_positions = account_info.positions

        if not broker_positions:
            logger.warning("No broker positions available")
            return

        analysis = await self._analyzer.analyze_attribution(positions, broker_positions)

        contributions_data = [
            {
                "sector": c.sector,
                "sector_etf": c.sector_etf,
                "total_value": c.total_value,
                "portfolio_weight": c.portfolio_weight,
                "benchmark_weight": c.benchmark_weight,
                "over_under_weight": c.over_under_weight,
                "pnl": c.pnl,
                "return_pct": c.return_pct,
                "position_count": c.position_count,
            }
            for c in analysis.contributions
        ]

        record = SectorAttributionRecord(
            timestamp=analysis.timestamp,
            total_portfolio_value=analysis.total_portfolio_value,
            benchmark_name=analysis.benchmark_name,
            contributions=contributions_data,
        )

        await self._state.store_sector_attribution(record, session=session)

        logger.info(
            f"Sector attribution complete: {len(analysis.contributions)} sectors, "
            f"value=${analysis.total_portfolio_value:,.2f}"
        )

    def __repr__(self) -> str:
        """String representation."""
        return "DaemonSectorAttribution()"
