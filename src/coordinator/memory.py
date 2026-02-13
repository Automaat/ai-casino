"""Coordinator memory for persistent learning observations."""

import asyncio
from collections import deque
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Final

from loguru import logger
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from src.coordinator.decision_models import DecisionQueryResult
    from src.daemon.state import DaemonState
    from src.daemon.state.models import SignalOutcome
    from src.data.broker import AlpacaBroker
    from src.database.repositories.analysis import AnalysisRecordRepository
    from src.database.repositories.signal_outcome import SignalOutcomeRepository

# Constants for memory limits
_MAX_IN_MEMORY_RECORDS: Final[int] = 20


class ObservationRecord(BaseModel):
    """Single observation record."""

    timestamp: datetime
    observation: str
    category: str = Field(description="Observation category")


class DecisionQueryParams(BaseModel):
    """Parameters for querying past trading decisions."""

    symbol: str | None = Field(default=None, description="Optional symbol filter")
    signal: str | None = Field(default=None, description="Optional signal filter (BUY/SELL/HOLD)")
    lookback_days: int = Field(default=90, description="Days to look back")
    min_confidence: float | None = Field(default=None, ge=0.0, le=1.0, description="Min confidence filter")
    limit: int = Field(default=50, description="Max results to return")
    horizon: str = Field(default="5d", description="Outcome horizon - 1d, 5d, or 20d")


class CoordinatorMemory:
    """Append-only memory for coordinator learning observations with multi-tier context."""

    def __init__(
        self,
        memory_file: Path | None = None,
        daemon_state: DaemonState | None = None,
        analysis_repo: AnalysisRecordRepository | None = None,
        signal_outcome_repo: SignalOutcomeRepository | None = None,
        broker: AlpacaBroker | None = None,
    ) -> None:
        """Initialize coordinator memory.

        Args:
            memory_file: Path to JSONL memory file (default: ~/.ai-casino/coordinator-memory.jsonl)
            daemon_state: Optional daemon state for today's data access
            analysis_repo: Optional analysis repository for historical queries
            signal_outcome_repo: Optional signal outcome repository for learning queries
            broker: Optional broker for portfolio data access
        """
        self._memory_file = memory_file or Path("~/.ai-casino/coordinator-memory.jsonl").expanduser()
        self._memory_file.parent.mkdir(parents=True, exist_ok=True)

        # Dependencies for multi-tier memory
        self._daemon_state = daemon_state
        self._analysis_repo = analysis_repo
        self._signal_outcome_repo = signal_outcome_repo
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
                        logger.opt(exception=True).warning(f"Failed to parse observation record: {e}")
                        continue

            # Return in reverse order (most recent first)
            return list(reversed(records))

        except Exception as e:
            logger.opt(exception=True).error(f"Failed to read observations: {e}")
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
            all_analyses = await self._daemon_state.get_analyses(limit=1000)
            today_analyses = [record for record in all_analyses if record.timestamp.date() == today]

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
            all_plans = await self._daemon_state.get_game_plan_history(limit=100)
            today_plans = [plan for plan in all_plans if plan.timestamp.date() == today]

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
                all_analyses = await self._daemon_state.get_analyses(limit=1000)
                records = [r for r in all_analyses if r.symbol.upper() == symbol.upper()]
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

    async def query_decisions(self, params: DecisionQueryParams | None = None) -> list:
        """Query past trading decisions with outcomes for learning.

        Args:
            params: Query parameters (uses defaults if None)

        Returns:
            List of DecisionQueryResult instances
        """
        if params is None:
            params = DecisionQueryParams()

        if not self._signal_outcome_repo:
            logger.warning("Signal outcome repository not available")
            return []

        try:
            outcomes = await self._fetch_outcomes(
                params.symbol, params.signal, params.lookback_days, params.min_confidence, params.limit
            )
            return [self._convert_outcome_to_result(outcome, params.horizon) for outcome in outcomes]
        except Exception as e:
            logger.opt(exception=True).error(f"Decision query failed: {e}")
            return []

    async def _fetch_outcomes(
        self,
        symbol: str | None,
        signal: str | None,
        lookback_days: int,
        min_confidence: float | None,
        limit: int,
    ) -> list:
        """Fetch signal outcomes from repository with filters.

        Args:
            symbol: Optional symbol filter
            signal: Optional signal filter
            lookback_days: Days to look back
            min_confidence: Min confidence filter
            limit: Max results to return

        Returns:
            List of SignalOutcome instances
        """
        if not self._signal_outcome_repo:
            return []

        start_date = datetime.now(UTC) - timedelta(days=lookback_days)

        if symbol:
            outcomes = await self._signal_outcome_repo.get_by_symbol(
                symbol=symbol,
                limit=limit,
                start_date=start_date,
            )
        else:
            outcomes = await self._signal_outcome_repo.get_recent_outcomes(
                window=lookback_days,
                signal_type=signal,
                min_confidence=min_confidence,
            )
            outcomes = outcomes[:limit]

        return outcomes

    def _convert_outcome_to_result(self, outcome: SignalOutcome, horizon: str) -> DecisionQueryResult:
        """Convert SignalOutcome to DecisionQueryResult with HIT/MISS classification.

        Args:
            outcome: SignalOutcome instance
            horizon: Outcome horizon - "1d", "5d", or "20d"

        Returns:
            DecisionQueryResult instance
        """
        from src.coordinator.decision_models import DecisionQueryResult

        price_at_outcome = self._get_price_at_horizon(outcome, horizon)
        return_pct = self._calculate_return_pct(outcome.price_at_signal, price_at_outcome)
        hit_miss = self._classify_outcome(outcome.signal, outcome.price_at_signal, price_at_outcome)

        return DecisionQueryResult(
            symbol=outcome.symbol,
            timestamp=outcome.timestamp,
            signal=outcome.signal,
            confidence=outcome.confidence,
            price_at_signal=outcome.price_at_signal,
            price_at_outcome=price_at_outcome,
            return_pct=return_pct,
            hit_miss=hit_miss,
            regime=outcome.regime,
            strategy_used=outcome.strategy_used,
            trading_session=outcome.trading_session,
        )

    def _get_price_at_horizon(self, outcome: SignalOutcome, horizon: str) -> float | None:
        """Get price at specified horizon from outcome.

        Args:
            outcome: SignalOutcome instance
            horizon: Outcome horizon - "1d", "5d", or "20d"

        Returns:
            Price at horizon or None
        """
        if horizon == "1d":
            return outcome.price_at_1d
        if horizon == "5d":
            return outcome.price_at_5d
        if horizon == "20d":
            return outcome.price_at_20d
        return None

    def _calculate_return_pct(self, price_at_signal: float, price_at_outcome: float | None) -> float | None:
        """Calculate return percentage.

        Args:
            price_at_signal: Entry price
            price_at_outcome: Exit price or None

        Returns:
            Return percentage or None
        """
        if price_at_outcome is None:
            return None
        return ((price_at_outcome - price_at_signal) / price_at_signal) * 100

    def _classify_outcome(self, signal: str, price_at_signal: float, price_at_outcome: float | None) -> str:
        """Classify outcome as HIT/MISS/PENDING.

        Args:
            signal: Trading signal (BUY/SELL/HOLD)
            price_at_signal: Entry price
            price_at_outcome: Exit price or None

        Returns:
            Classification string - "HIT", "MISS", or "PENDING"
        """
        if price_at_outcome is None:
            return "PENDING"

        if signal == "BUY":
            return "HIT" if price_at_outcome > price_at_signal else "MISS"
        if signal == "SELL":
            return "HIT" if price_at_outcome < price_at_signal else "MISS"
        return "PENDING"  # HOLD

    async def get_success_rate(
        self,
        signal: str | None = None,
        regime: str | None = None,
        lookback_days: int = 90,
        horizon: str = "5d",
    ) -> dict[str, object]:
        """Calculate success rate statistics for past decisions.

        Args:
            signal: Optional signal filter (BUY/SELL)
            regime: Optional regime filter
            lookback_days: Days to look back (default: 90)
            horizon: Outcome horizon - "1d", "5d", or "20d"

        Returns:
            SuccessRateStats instance
        """
        from src.coordinator.decision_models import SuccessRateStats

        if not self._signal_outcome_repo:
            logger.warning("Signal outcome repository not available")
            return SuccessRateStats(
                total_decisions=0,
                hit_count=0,
                miss_count=0,
                success_rate=0.0,
                avg_return=None,
                avg_confidence=0.0,
            ).model_dump()

        try:
            # Query decisions with outcomes
            params = DecisionQueryParams(
                signal=signal,
                lookback_days=lookback_days,
                limit=1000,
                horizon=horizon,
            )
            decisions = await self.query_decisions(params)

            if not decisions:
                return SuccessRateStats(
                    total_decisions=0,
                    hit_count=0,
                    miss_count=0,
                    success_rate=0.0,
                    avg_return=None,
                    avg_confidence=0.0,
                ).model_dump()

            # Filter by regime if specified
            if regime:
                decisions = [d for d in decisions if d.regime == regime]

            # Calculate stats
            hit_count = sum(1 for d in decisions if d.hit_miss == "HIT")
            miss_count = sum(1 for d in decisions if d.hit_miss == "MISS")
            pending_count = sum(1 for d in decisions if d.hit_miss == "PENDING")

            total_decided = hit_count + miss_count
            success_rate = hit_count / total_decided if total_decided > 0 else 0.0

            # Calculate average return for completed decisions
            completed_returns = [d.return_pct for d in decisions if d.return_pct is not None]
            avg_return = sum(completed_returns) / len(completed_returns) if completed_returns else None

            # Calculate average confidence
            avg_confidence = sum(d.confidence for d in decisions) / len(decisions) if decisions else 0.0

            return SuccessRateStats(
                total_decisions=len(decisions),
                hit_count=hit_count,
                miss_count=miss_count,
                pending_count=pending_count,
                success_rate=success_rate,
                avg_return=avg_return,
                avg_confidence=avg_confidence,
            ).model_dump()

        except Exception as e:
            logger.opt(exception=True).error(f"Success rate calculation failed: {e}")
            return SuccessRateStats(
                total_decisions=0,
                hit_count=0,
                miss_count=0,
                success_rate=0.0,
                avg_return=None,
                avg_confidence=0.0,
            ).model_dump()

    def __repr__(self) -> str:
        """String representation."""
        deps = []
        if self._daemon_state:
            deps.append("daemon_state")
        if self._analysis_repo:
            deps.append("analysis_repo")
        if self._signal_outcome_repo:
            deps.append("signal_outcome_repo")
        if self._broker:
            deps.append("broker")
        deps_str = f", deps={','.join(deps)}" if deps else ""
        return f"CoordinatorMemory(file={self._memory_file}{deps_str})"
