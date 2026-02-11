"""Coordinator memory for persistent learning observations."""

import asyncio
from collections import deque
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Final

from loguru import logger
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from src.daemon.state import DaemonState
    from src.data.broker import AlpacaBroker
    from src.database.repositories.analysis import AnalysisRecordRepository
    from src.database.repositories.trade import TradeRepository

# Constants for memory limits
_MAX_IN_MEMORY_RECORDS: Final[int] = 20


class ObservationRecord(BaseModel):
    """Single observation record."""

    timestamp: datetime
    observation: str
    category: str = Field(description="Observation category")


class CoordinatorMemory:
    """Append-only memory for coordinator learning observations with multi-tier context."""

    def __init__(
        self,
        memory_file: Path | None = None,
        daemon_state: DaemonState | None = None,
        analysis_repo: AnalysisRecordRepository | None = None,
        trade_repo: TradeRepository | None = None,
        broker: AlpacaBroker | None = None,
    ) -> None:
        """Initialize coordinator memory.

        Args:
            memory_file: Path to JSONL memory file (default: ~/.ai-casino/coordinator-memory.jsonl)
            daemon_state: Optional daemon state for today's data access
            analysis_repo: Optional analysis repository for historical queries
            trade_repo: Optional trade repository for historical queries
            broker: Optional broker for portfolio data access
        """
        self._memory_file = memory_file or Path("~/.ai-casino/coordinator-memory.jsonl").expanduser()
        self._memory_file.parent.mkdir(parents=True, exist_ok=True)

        # Dependencies for multi-tier memory
        self._daemon_state = daemon_state
        self._analysis_repo = analysis_repo
        self._trade_repo = trade_repo
        self._broker = broker

        # Create file if it doesn't exist
        if not self._memory_file.exists():
            self._memory_file.touch()
            logger.info(f"Created coordinator memory at {self._memory_file}")

    async def save(self, observation: str, category: str = "general") -> None:
        """Save observation to memory file.

        Args:
            observation: Observation text
            category: Category (market/pattern/error/success/general)
        """
        record = ObservationRecord(
            timestamp=datetime.now(UTC),
            observation=observation,
            category=category,
        )

        # Append to JSONL file (offload to thread)
        await asyncio.to_thread(self._append_record, record)
        logger.debug(f"Saved observation: {category}")

    def _append_record(self, record: ObservationRecord) -> None:
        """Append record to JSONL file.

        Args:
            record: Observation record to append
        """
        with self._memory_file.open("a") as f:
            f.write(record.model_dump_json() + "\n")

    async def retrieve_recent(
        self,
        limit: int = 50,
        category: str | None = None,
    ) -> list[ObservationRecord]:
        """Retrieve recent observations from memory.

        Args:
            limit: Maximum number of records to retrieve
            category: Optional category filter

        Returns:
            List of observation records (most recent first)
        """
        return await asyncio.to_thread(self._read_records, limit, category)

    def _read_records(self, limit: int, category: str | None) -> list[ObservationRecord]:
        """Read records from JSONL file.

        Args:
            limit: Maximum number of records
            category: Optional category filter

        Returns:
            List of observation records (most recent first)
        """
        if not self._memory_file.exists():
            return []

        try:
            # Use deque to keep only last N matching records, bounded memory
            records: deque[ObservationRecord] = deque(maxlen=limit)

            with self._memory_file.open() as f:
                for line in f:
                    if not line.strip():
                        continue

                    try:
                        record = ObservationRecord.model_validate_json(line)
                        if category is None or record.category == category:
                            records.append(record)
                    except Exception as e:
                        logger.warning(f"Failed to parse observation record: {e}")
                        continue

            # Return in reverse order (most recent first)
            return list(reversed(records))

        except Exception as e:
            logger.error(f"Failed to read observations: {e}")
            return []

    async def get_today_summary(self, max_tokens: int = 500) -> str:
        """Get summary of today's analyses grouped by signal.

        Args:
            max_tokens: Maximum token budget for output

        Returns:
            Formatted markdown summary of today's analyses
        """
        if not self._daemon_state:
            return "No analyses today"

        try:
            # Filter today's analyses
            today = datetime.now(UTC).date()
            today_analyses = [
                record for record in self._daemon_state.analyses if record.timestamp.date() == today
            ]

            if not today_analyses:
                return "No analyses today"

            # Group by signal
            by_signal: dict[str, list] = {"BUY": [], "SELL": [], "HOLD": []}
            for record in today_analyses:
                signal = record.signal.upper()
                if signal in by_signal:
                    by_signal[signal].append(record)

            # Build summary
            lines = ["## Today's Analyses"]
            for signal in ["BUY", "SELL", "HOLD"]:
                records = by_signal[signal]
                if records:
                    lines.append(f"\n**{signal}** ({len(records)} symbols):")
                    for record in records:
                        session = (
                            f" ({record.trading_session.value})"
                            if record.trading_session.value == "PRE_MARKET"
                            else ""
                        )
                        executed = "✓" if record.executed_trade else "✗"
                        time_str = record.timestamp.strftime("%H:%M")
                        lines.append(
                            f"- **{record.symbol}** @ {time_str}{session} - "
                            f"Confidence: {record.confidence:.0%} | Executed: {executed}"
                        )

            text = "\n".join(lines)
            return self._truncate_to_budget(text, max_tokens)

        except Exception as e:
            logger.opt(exception=True).error(f"Failed to generate today summary: {e}")
            return f"Error generating today summary: {e}"

    async def get_today_game_plan(self, max_tokens: int = 300) -> str:
        """Get today's game plan from DaemonState.

        Args:
            max_tokens: Maximum token budget for output

        Returns:
            Formatted markdown game plan or fallback message
        """
        if not self._daemon_state:
            return "Game plan unavailable"

        try:
            # Filter today's game plans
            today = datetime.now(UTC).date()
            today_plans = [
                plan for plan in self._daemon_state.game_plan_history if plan.timestamp.date() == today
            ]

            if not today_plans:
                return "No game plan generated today"

            # Get most recent plan
            latest_plan = today_plans[-1]

            lines = [
                "## Today's Game Plan",
                f"**Priority Symbols:** {', '.join(latest_plan.priority_symbols)}",
                f"**Risk Stance:** {latest_plan.risk_stance}",
                f"**Sector Focus:** {', '.join(latest_plan.sector_focus)}",
            ]

            text = "\n".join(lines)
            return self._truncate_to_budget(text, max_tokens)

        except Exception as e:
            logger.opt(exception=True).error(f"Failed to get game plan: {e}")
            return f"Error loading game plan: {e}"

    async def get_portfolio_summary(self, max_tokens: int = 400) -> str:
        """Get current portfolio summary from broker.

        Args:
            max_tokens: Maximum token budget for output

        Returns:
            Formatted markdown portfolio summary
        """
        if not self._broker:
            return "Portfolio data unavailable"

        try:
            account_info = await asyncio.to_thread(self._broker.get_account_info)

            if getattr(account_info, "portfolio_value", None) and account_info.portfolio_value > 0:
                exposure_percent_str = (
                    f"{account_info.total_exposure / account_info.portfolio_value * 100:.1f}%"
                )
            else:
                exposure_percent_str = "N/A"

            lines = [
                "## Current Portfolio",
                f"- **Balance**: ${account_info.balance:,.2f}",
                f"- **Portfolio Value**: ${account_info.portfolio_value:,.2f}",
                f"- **Available Cash**: ${account_info.available_cash:,.2f}",
                f"- **Total Exposure**: ${account_info.total_exposure:,.2f} ({exposure_percent_str})",
            ]

            if account_info.positions:
                lines.append(f"\n**Positions ({len(account_info.positions)}):**")
                for symbol, pos in account_info.positions.items():
                    lines.append(
                        f"- {symbol}: {pos.qty} shares @ ${pos.avg_entry_price:.2f} "
                        f"(P&L: ${pos.unrealized_pnl:,.2f} / {pos.unrealized_pnl_percent:+.1f}%)"
                    )

            text = "\n".join(lines)
            return self._truncate_to_budget(text, max_tokens)

        except Exception as e:
            logger.opt(exception=True).error(f"Failed to get portfolio summary: {e}")
            return f"Portfolio data unavailable: {e}"

    async def get_analysis_history(
        self,
        symbol: str,
        days: int = 7,
        max_tokens: int = 500,
    ) -> str:
        """Get historical analysis for a symbol from database.

        Args:
            symbol: Stock ticker symbol
            days: Number of days to look back
            max_tokens: Maximum token budget for output

        Returns:
            Formatted markdown analysis history
        """
        if not self._analysis_repo:
            # Fallback to in-memory if no repository
            if self._daemon_state:
                records = [r for r in self._daemon_state.analyses if r.symbol.upper() == symbol.upper()]
                if not records:
                    return f"No analysis history found for {symbol}"

                # Get most recent records (limit to prevent unbounded growth)
                recent = (
                    records[-_MAX_IN_MEMORY_RECORDS:] if len(records) > _MAX_IN_MEMORY_RECORDS else records
                )
                recent.reverse()  # Most recent first

                lines = [f"# Analysis History - {symbol} (in-memory only)", ""]
                for i, record in enumerate(recent, 1):
                    executed = "✓" if record.executed_trade else "✗"
                    rsi_text = f"RSI: {record.rsi:.1f}" if record.rsi is not None else "RSI: N/A"
                    macd_text = (
                        f"MACD: {record.macd_hist:.4f}" if record.macd_hist is not None else "MACD: N/A"
                    )
                    lines.extend(
                        [
                            f"## {i}. {record.signal}",
                            f"- **Timestamp:** {record.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
                            f"- **Confidence:** {record.confidence:.0%}",
                            f"- **Executed:** {executed}",
                            f"- **Indicators:** {rsi_text} | {macd_text}",
                            "",
                        ]
                    )

                text = "\n".join(lines)
                return self._truncate_to_budget(text, max_tokens)

            return f"No analysis history available for {symbol}"

        try:
            # Query database for historical records
            start_date = datetime.now(UTC) - timedelta(days=days)
            records = await self._analysis_repo.get_by_date_range(
                start=start_date,
                end=datetime.now(UTC),
                symbol=symbol,
            )

            if not records:
                return f"No analysis history found for {symbol} in last {days} days"

            # Sort by timestamp descending (most recent first)
            records.sort(key=lambda r: r.timestamp, reverse=True)

            lines = [f"# Analysis History - {symbol} (last {days} days)", ""]
            for i, record in enumerate(records, 1):
                executed = "✓" if record.executed_trade else "✗"
                rsi_text = f"RSI: {record.rsi:.1f}" if record.rsi is not None else "RSI: N/A"
                macd_text = f"MACD: {record.macd_hist:.4f}" if record.macd_hist is not None else "MACD: N/A"
                lines.extend(
                    [
                        f"## {i}. {record.signal}",
                        f"- **Timestamp:** {record.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
                        f"- **Confidence:** {record.confidence:.0%}",
                        f"- **Executed:** {executed}",
                        f"- **Session:** {record.trading_session.value}",
                        f"- **Indicators:** {rsi_text} | {macd_text}",
                        "",
                    ]
                )

            text = "\n".join(lines)
            return self._truncate_to_budget(text, max_tokens)

        except Exception as e:
            logger.opt(exception=True).error(f"Analysis history query failed: {e}")
            return f"Failed to retrieve analysis history for {symbol}: {e}"

    def _truncate_to_budget(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token budget.

        Uses character-based approximation: 4 chars ≈ 1 token.
        Truncates at last complete line before limit.

        Args:
            text: Text to truncate
            max_tokens: Maximum token budget

        Returns:
            Truncated text with suffix if truncated
        """
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return text

        # Truncate at last complete line before limit
        truncated = text[:max_chars]
        last_newline = truncated.rfind("\n")
        if last_newline > 0:
            truncated = truncated[:last_newline]

        return f"{truncated}\n\n[Truncated for length]"

    def __repr__(self) -> str:
        """String representation."""
        deps = []
        if self._daemon_state:
            deps.append("daemon_state")
        if self._analysis_repo:
            deps.append("analysis_repo")
        if self._trade_repo:
            deps.append("trade_repo")
        if self._broker:
            deps.append("broker")
        deps_str = f", deps={','.join(deps)}" if deps else ""
        return f"CoordinatorMemory(file={self._memory_file}{deps_str})"
