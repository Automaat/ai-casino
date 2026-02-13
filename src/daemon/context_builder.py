"""Daemon context builder service."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from src.agents.game_plan import GamePlan

if TYPE_CHECKING:
    from src.daemon.factory import DaemonComponents
    from src.daemon.state import SectorRotationRecord
    from src.di.container import AppContainer


class DaemonContextBuilder:
    """Build analysis contexts for agents."""

    def __init__(
        self,
        components: DaemonComponents,
        container: AppContainer,
    ) -> None:
        """Initialize context builder.

        Args:
            components: Daemon components
            container: DI container for fetcher access
        """
        self.components = components
        self.container = container

    def __repr__(self) -> str:
        """Return string representation."""
        return "DaemonContextBuilder()"

    def build_analysis_contexts(
        self,
        symbol: str,
    ) -> tuple[str | None, str | None, str | None, str | None]:
        """Build all analysis contexts for symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Tuple of (sector_context, earnings_context, peer_context, game_plan_context)
        """
        sector_ctx = self._build_sector_context()
        earnings_ctx = self._build_earnings_context(symbol)
        peer_ctx = self._build_peer_context(symbol)
        game_plan_ctx = self._load_game_plan_context()
        return sector_ctx, earnings_ctx, peer_ctx, game_plan_ctx

    def build_earnings_context_for_watchlist(self, watchlist: list[str]) -> str | None:
        """Build earnings context for all watchlist symbols.

        Args:
            watchlist: List of symbols

        Returns:
            Combined earnings context or None
        """
        if not watchlist:
            return None

        contexts = []
        for symbol in watchlist:
            ctx = self._build_earnings_context(symbol)
            if ctx:
                contexts.append(ctx)

        return "\n".join(contexts) if contexts else None

    def _format_sector_context(self, record: SectorRotationRecord) -> str:
        """Format sector rotation record as text for trader prompt.

        Args:
            record: Sector rotation state record

        Returns:
            Formatted context string
        """
        lines = [
            f"Leading Sectors: {', '.join(record.leading_sectors)}",
            f"Lagging Sectors: {', '.join(record.lagging_sectors)}",
            "",
        ]

        # Sort by strength descending
        sorted_sectors = sorted(record.sector_strengths.items(), key=lambda x: x[1], reverse=True)

        for rank, (sector, strength) in enumerate(sorted_sectors, 1):
            momentum = record.sector_momenta.get(sector, "NEUTRAL")
            lines.append(f"  {rank}. {sector}: strength={strength:+.2f} [{momentum}]")

        return "\n".join(lines)

    def _build_sector_context(self) -> str | None:
        """Build sector rotation context from latest record.

        Returns:
            Formatted sector context string or None if not available
        """
        # NOTE: Requires async state access after JSON elimination
        # TODO: Implement using await self.components.state.get_sector_rotation_history()
        return None

    def _load_game_plan_context(self) -> str | None:
        """Load today's game plan and format as context string.

        Returns:
            Formatted game plan context or None
        """
        plan_dir = Path(self.components.config.game_plan.plan_dir).expanduser()
        today = datetime.now(self.components.scheduler.timezone).date()
        plan_file = plan_dir / f"{today}.json"

        if not plan_file.exists():
            return None

        try:
            with plan_file.open() as f:
                data = json.load(f)
                plan = GamePlan.model_validate(data)

            key_levels_str = ", ".join(f"{sym}: ${lvl:.2f}" for sym, lvl in plan.key_levels.items())
            return (
                f"Risk Stance: {plan.risk_stance}\n"
                f"Priority Symbols: {', '.join(plan.priority_symbols)}\n"
                f"Sector Focus: {', '.join(plan.sector_focus)}\n"
                f"Key Levels: {key_levels_str}\n"
                f"Reasoning: {plan.reasoning}"
            )
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to load game plan context: {e}")
            return None

    def _build_earnings_context(self, _symbol: str) -> str | None:
        """Build earnings context string from latest calendar state.

        Args:
            symbol: Stock ticker to build context for

        Returns:
            Formatted earnings context or None
        """
        # NOTE: Requires async state access after JSON elimination
        # TODO: Implement using await self.components.state.get_earnings_calendar_history()
        return None

    def _build_peer_context(self, symbol: str) -> str | None:
        """Build peer analysis context string from persisted data.

        Args:
            symbol: Stock ticker to build context for

        Returns:
            Formatted peer analysis context or None
        """
        try:
            from src.daemon.peer_analysis import DeepPeerAnalyzer, PeerAnalyzerConfig

            config = PeerAnalyzerConfig(output_dir=self.components.config.peer_analysis.output_dir)
            analyzer = DeepPeerAnalyzer(config=config)
            return analyzer.format_context(symbol)
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to build peer context for {symbol}: {e}")
            return None
