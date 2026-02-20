"""Options flow watcher - monitors institutional positioning via options chains."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass
from typing import Final

from loguru import logger

from src.daemon.events import BlockTrade, OptionsFlowDirection, OptionsFlowSignal
from src.data.options_flow import OptionContract, OptionsChainSnapshot, OptionsFlowFetcher
from src.v1.watchers.base import PeriodicWatcher

_VOLUME_HISTORY_WINDOW: Final[int] = 6
_MIN_HISTORY_ENTRIES: Final[int] = 2
_BULLISH_CALL_RATIO: Final[float] = 0.6
_BEARISH_CALL_RATIO: Final[float] = 0.4
_PCR_ELEVATED_PUTS: Final[float] = 1.5
_PCR_HEAVY_CALLS: Final[float] = 0.5


@dataclass
class OptionsFlowWatcherConfig:
    """Configuration for options flow watcher."""

    poll_interval_minutes: int = 15
    volume_spike_threshold: float = 2.0
    block_trade_threshold: float = 100_000
    symbols: list[str] | None = None


class OptionsFlowWatcher(PeriodicWatcher):
    """Background service that polls yfinance options chains and computes flow signals."""

    def __init__(
        self,
        fetcher: OptionsFlowFetcher,
        config: OptionsFlowWatcherConfig,
    ) -> None:
        """Initialize options flow watcher.

        Args:
            fetcher: Options chain data fetcher
            config: Watcher configuration
        """
        super().__init__(poll_interval=config.poll_interval_minutes * 60)
        self._fetcher = fetcher
        self._config = config
        self._signals: dict[str, OptionsFlowSignal] = {}
        self._volume_history: dict[str, list[int]] = defaultdict(list)

    @property
    def name(self) -> str:
        """Watcher display name."""
        return "OptionsFlowWatcher"

    def get_signal(self, symbol: str) -> OptionsFlowSignal | None:
        """Return current flow signal for a symbol (sync, no await).

        Args:
            symbol: Stock ticker

        Returns:
            OptionsFlowSignal if available, None otherwise
        """
        return self._signals.get(symbol)

    def _detect_block_trades(self, snapshot: OptionsChainSnapshot) -> list[BlockTrade]:
        """Detect high-premium contracts from options chain.

        Args:
            snapshot: Options chain snapshot

        Returns:
            List of detected block trades
        """
        blocks: list[BlockTrade] = []
        threshold = self._config.block_trade_threshold

        for contract in snapshot.calls + snapshot.puts:
            premium = self._compute_contract_premium(contract)
            if premium >= threshold:
                blocks.append(
                    BlockTrade(
                        strike=contract.strike,
                        expiry=str(contract.expiry),
                        premium=premium,
                        volume=contract.volume,
                        option_type=contract.option_type,
                        is_itm=contract.in_the_money,
                    )
                )

        blocks.sort(key=lambda b: b.premium, reverse=True)
        return blocks

    @staticmethod
    def _compute_contract_premium(contract: OptionContract) -> float:
        """Compute premium for a contract.

        Uses midpoint if available, falls back to lastPrice.

        Args:
            contract: Option contract

        Returns:
            Estimated premium in dollars
        """
        midpoint = (contract.bid + contract.ask) / 2
        price = midpoint if midpoint > 0 else contract.last_price
        return contract.volume * price * 100

    def _compute_put_call_ratio(self, snapshot: OptionsChainSnapshot) -> float:
        """Compute put/call volume ratio.

        Args:
            snapshot: Options chain snapshot

        Returns:
            Put/call ratio (>1 = bearish, <1 = bullish)
        """
        if snapshot.total_call_volume == 0:
            return 0.0 if snapshot.total_put_volume == 0 else 10.0
        return snapshot.total_put_volume / snapshot.total_call_volume

    def _compute_volume_spike(self, symbol: str, total_volume: int) -> float:
        """Compute volume spike vs rolling 5-session average.

        Args:
            symbol: Stock ticker
            total_volume: Current total options volume

        Returns:
            Volume as multiple of average (1.0 = normal)
        """
        history = self._volume_history[symbol]
        history.append(total_volume)

        # Keep rolling window of 6 (current + 5 history)
        if len(history) > _VOLUME_HISTORY_WINDOW:
            self._volume_history[symbol] = history[-_VOLUME_HISTORY_WINDOW:]
            history = self._volume_history[symbol]

        if len(history) < _MIN_HISTORY_ENTRIES:
            return 1.0

        # Average of previous sessions (exclude current)
        prev_avg = sum(history[:-1]) / len(history[:-1])
        if prev_avg == 0:
            return 1.0
        return total_volume / prev_avg

    def _determine_direction(self, snapshot: OptionsChainSnapshot) -> OptionsFlowDirection:
        """Determine net premium direction from call vs put premium.

        Args:
            snapshot: Options chain snapshot

        Returns:
            OptionsFlowDirection
        """
        call_premium = sum(self._compute_contract_premium(c) for c in snapshot.calls)
        put_premium = sum(self._compute_contract_premium(c) for c in snapshot.puts)

        total = call_premium + put_premium
        if total == 0:
            return OptionsFlowDirection.NEUTRAL

        ratio = call_premium / total
        if ratio > _BULLISH_CALL_RATIO:
            return OptionsFlowDirection.BULLISH
        if ratio < _BEARISH_CALL_RATIO:
            return OptionsFlowDirection.BEARISH
        return OptionsFlowDirection.NEUTRAL

    def _compute_significance(
        self,
        spike: float,
        blocks: list[BlockTrade],
        pcr: float,
    ) -> float:
        """Compute composite significance score (0.0-1.0).

        Weights: 40% volume spike, 30% block trades, 30% PCR extremity.

        Args:
            spike: Volume spike multiple
            blocks: Detected block trades
            pcr: Put/call ratio

        Returns:
            Significance score between 0.0 and 1.0
        """
        # Volume spike component (0-1): spike of 2x = 0.5, 4x+ = 1.0
        spike_score = min(1.0, max(0.0, (spike - 1.0) / 3.0))

        # Block trades component (0-1): 1 block = 0.33, 3+ blocks = 1.0
        block_score = min(1.0, len(blocks) / 3.0)

        # PCR extremity component (0-1): distance from neutral (1.0)
        pcr_deviation = abs(pcr - 1.0)
        pcr_score = min(1.0, pcr_deviation / 1.5)

        return 0.4 * spike_score + 0.3 * block_score + 0.3 * pcr_score

    def _build_reason(
        self,
        pcr: float,
        spike: float,
        blocks: list[BlockTrade],
        direction: OptionsFlowDirection,
    ) -> str:
        """Build human-readable reason string.

        Args:
            pcr: Put/call ratio
            spike: Volume spike multiple
            blocks: Detected block trades
            direction: Net premium direction

        Returns:
            Reason string
        """
        parts = []
        if spike >= self._config.volume_spike_threshold:
            parts.append(f"Vol {spike:.1f}x avg")
        if blocks:
            total_premium = sum(b.premium for b in blocks)
            parts.append(f"{len(blocks)} large trades (${total_premium:,.0f})")
        if pcr > _PCR_ELEVATED_PUTS:
            parts.append(f"Elevated puts P/C={pcr:.2f}")
        elif pcr < _PCR_HEAVY_CALLS:
            parts.append(f"Heavy calls P/C={pcr:.2f}")

        if not parts:
            return f"{direction} flow, normal activity"
        return f"{direction} flow: {'; '.join(parts)}"

    async def _fetch_and_assess_symbol(self, symbol: str) -> None:
        """Fetch and assess options flow for a single symbol.

        Args:
            symbol: Stock ticker
        """
        snapshot = await asyncio.to_thread(self._fetcher.fetch_options_chain, symbol)

        if snapshot.is_empty:
            return

        pcr = self._compute_put_call_ratio(snapshot)
        spike = self._compute_volume_spike(symbol, snapshot.total_volume)
        blocks = self._detect_block_trades(snapshot)
        direction = self._determine_direction(snapshot)
        significance = self._compute_significance(spike, blocks, pcr)
        has_unusual = spike >= self._config.volume_spike_threshold or len(blocks) > 0
        reason = self._build_reason(pcr, spike, blocks, direction)

        signal = OptionsFlowSignal(
            symbol=symbol,
            put_call_ratio=pcr,
            volume_vs_avg=spike,
            has_unusual_activity=has_unusual,
            block_trades=blocks,
            net_premium_direction=direction,
            significance_score=significance,
            reason=reason,
        )
        self._signals[symbol] = signal

    async def _tick(self) -> None:
        """Fetch and assess all configured symbols with concurrency limit."""
        symbols = self._config.symbols or []
        if not symbols:
            return

        sem = asyncio.Semaphore(3)

        async def _limited(sym: str) -> None:
            async with sem:
                try:
                    await self._fetch_and_assess_symbol(sym)
                except Exception as e:
                    logger.opt(exception=True).warning(f"Options flow assessment failed for {sym}: {e}")

        await asyncio.gather(*[_limited(s) for s in symbols])

        active = [s for s in self._signals.values() if s.has_unusual_activity]
        logger.info(f"Options flow assessed: {len(symbols)} symbols, {len(active)} with unusual activity")

    def __repr__(self) -> str:
        """String representation."""
        return f"OptionsFlowWatcher(running={self.running}, signals={len(self._signals)})"
