"""Persistent storage for optimized strategy parameters."""

import json
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger
from pydantic import BaseModel


class SymbolStrategyParams(BaseModel):
    """Optimized parameters for a symbol-strategy combination."""

    symbol: str
    strategy_name: str
    params: dict[str, float | int]
    metrics: dict[str, float]
    optimized_at: datetime
    trials_count: int
    validation_trades: int


class OptimizedParamStore:
    """Persistent store for optimized strategy parameters.

    JSON schema: {symbol: {strategy_name: SymbolStrategyParams}}
    """

    def __init__(self, path: str = "~/.ai-casino/optimized-params.json") -> None:
        """Initialize param store.

        Args:
            path: Path to JSON file (supports ~ expansion)
        """
        self._path = Path(path).expanduser()
        self._data: dict[str, dict[str, SymbolStrategyParams]] = {}
        self._load_from_disk()
        logger.info(f"OptimizedParamStore initialized: {self._path}")

    def _load_from_disk(self) -> None:
        """Load params from disk."""
        if not self._path.exists():
            self._data = {}
            return

        try:
            with self._path.open() as f:
                raw = json.load(f)

            self._data = {}
            for symbol, strategies in raw.items():
                self._data[symbol] = {}
                for strategy_name, params_data in strategies.items():
                    self._data[symbol][strategy_name] = SymbolStrategyParams.model_validate(params_data)

            logger.info(f"Loaded optimized params for {len(self._data)} symbols")
        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to load param store: {e}, starting fresh")
            self._data = {}

    def _save_to_disk(self) -> None:
        """Persist params to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)

        serialized: dict[str, dict[str, dict]] = {}
        for symbol, strategies in self._data.items():
            serialized[symbol] = {}
            for strategy_name, params in strategies.items():
                serialized[symbol][strategy_name] = params.model_dump(mode="json")

        with self._path.open("w") as f:
            json.dump(serialized, f, indent=2, default=str)

        logger.debug(f"Saved optimized params to {self._path}")

    def load_all(self) -> dict[str, dict[str, SymbolStrategyParams]]:
        """Load all optimized parameters.

        Returns:
            Nested dict: {symbol: {strategy_name: SymbolStrategyParams}}
        """
        return self._data

    def get(self, symbol: str, strategy_name: str) -> SymbolStrategyParams | None:
        """Get optimized params for a symbol-strategy pair.

        Args:
            symbol: Stock ticker
            strategy_name: Strategy name

        Returns:
            SymbolStrategyParams or None if not found
        """
        return self._data.get(symbol, {}).get(strategy_name)

    def save(self, params: SymbolStrategyParams) -> None:
        """Save optimized params, merging into existing data.

        Args:
            params: Optimized parameters to save
        """
        if params.symbol not in self._data:
            self._data[params.symbol] = {}

        self._data[params.symbol][params.strategy_name] = params
        self._save_to_disk()

        logger.info(f"Saved optimized params: {params.symbol}/{params.strategy_name}")

    def is_stale(self, symbol: str, strategy_name: str, max_age_days: int = 30) -> bool:
        """Check if optimized params are stale.

        Args:
            symbol: Stock ticker
            strategy_name: Strategy name
            max_age_days: Maximum age in days before params are considered stale

        Returns:
            True if params don't exist or are older than max_age_days
        """
        params = self.get(symbol, strategy_name)
        if params is None:
            return True

        age = datetime.now(UTC) - params.optimized_at
        return age.days >= max_age_days

    def __repr__(self) -> str:
        """Return string representation."""
        total = sum(len(strategies) for strategies in self._data.values())
        return f"OptimizedParamStore(symbols={len(self._data)}, entries={total})"
