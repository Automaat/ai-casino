"""Daemon state persistence."""

import json
from datetime import datetime
from pathlib import Path

from loguru import logger
from pydantic import BaseModel, Field

from src.strategies.session import TradingSession


class AnalysisRecord(BaseModel):
    """Record of a single analysis run."""

    symbol: str
    timestamp: datetime
    signal: str
    confidence: float
    executed_trade: bool = False
    trading_session: TradingSession = TradingSession.REGULAR


class DaemonState(BaseModel):
    """Persistent state for the trading daemon."""

    last_run: datetime | None = None
    analyses: list[AnalysisRecord] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    total_analyses: int = 0
    total_trades: int = 0
    last_journal_date: str | None = None

    @classmethod
    def load(cls, path: str) -> "DaemonState":
        """Load state from JSON file.

        Args:
            path: Path to state file (supports ~ expansion)

        Returns:
            DaemonState instance
        """
        expanded_path = Path(path).expanduser()

        if not expanded_path.exists():
            logger.info(f"No existing state at {expanded_path}, starting fresh")
            return cls()

        try:
            with expanded_path.open() as f:
                data = json.load(f)
            logger.info(f"Loaded daemon state from {expanded_path}")
            return cls.model_validate(data)
        except Exception as e:
            logger.warning(f"Failed to load state: {e}, starting fresh")
            return cls()

    def save(self, path: str) -> None:
        """Save state to JSON file.

        Args:
            path: Path to state file (supports ~ expansion)
        """
        expanded_path = Path(path).expanduser()
        expanded_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with expanded_path.open("w") as f:
                json.dump(self.model_dump(mode="json"), f, indent=2, default=str)
            logger.debug(f"Saved daemon state to {expanded_path}")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def record_analysis(
        self,
        symbol: str,
        signal: str,
        confidence: float,
        executed: bool = False,
        trading_session: TradingSession = TradingSession.REGULAR,
    ) -> None:
        """Record an analysis result.

        Args:
            symbol: Stock ticker
            signal: Trading signal (BUY/SELL/HOLD)
            confidence: Signal confidence
            executed: Whether trade was executed
            trading_session: Trading session type (REGULAR/PRE_MARKET)
        """
        self.analyses.append(
            AnalysisRecord(
                symbol=symbol,
                timestamp=datetime.now(),  # noqa: DTZ005
                signal=signal,
                confidence=confidence,
                executed_trade=executed,
                trading_session=trading_session,
            )
        )
        self.total_analyses += 1
        if executed:
            self.total_trades += 1
        self.last_run = datetime.now()  # noqa: DTZ005

        if len(self.analyses) > 1000:
            self.analyses = self.analyses[-500:]

    def record_error(self, error: str) -> None:
        """Record an error.

        Args:
            error: Error message
        """
        timestamp = datetime.now().isoformat()  # noqa: DTZ005
        self.errors.append(f"{timestamp}: {error}")

        if len(self.errors) > 100:
            self.errors = self.errors[-50:]

    def __repr__(self) -> str:
        """Return string representation."""
        return f"DaemonState(analyses={self.total_analyses}, trades={self.total_trades})"
