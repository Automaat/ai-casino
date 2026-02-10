"""Daemon state persistence."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from src.discovery.models import DiscoveryCandidate, DiscoverySource
from src.screening.screener import ScreeningResult
from src.strategies.session import TradingSession

if TYPE_CHECKING:
    from src.daemon.degradation import DegradationContext
    from src.daemon.positions import PositionManagementAction, PositionRecord
    from src.database.repositories.analysis import AnalysisRecordRepository
    from src.database.repositories.discovery import DiscoveryHistoryRepository
    from src.database.repositories.snapshot import PortfolioSnapshotRepository


class AnalysisRecord(BaseModel):
    """Record of a single analysis run."""

    symbol: str
    timestamp: datetime
    signal: str
    confidence: float
    executed_trade: bool = False
    trading_session: TradingSession = TradingSession.REGULAR
    is_paper_trade: bool = True
    rsi: float | None = None
    macd_hist: float | None = None
    reasoning: list[str] = Field(default_factory=list)


class ScreeningRecord(BaseModel):
    """Record of an after-hours screening run."""

    timestamp: datetime
    criteria: str
    universe: str
    top_symbols: list[str]
    candidates: list[ScreeningResult]
    screened_at: datetime


class PortfolioSnapshot(BaseModel):
    """Snapshot of portfolio state at a point in time."""

    balance: float
    available_cash: float
    total_exposure: float
    portfolio_value: float
    positions: dict
    trigger: str


class DiscoveryHistoryRecord(BaseModel):
    """Record of stock discovery outcome for learning."""

    symbol: str
    discovered_at: datetime
    composite_score: float
    sources: list[DiscoverySource]
    added_to_watchlist: bool
    ttl_expires_at: datetime
    first_signal: str | None = None
    first_signal_date: datetime | None = None
    outcome_7d: float | None = None
    outcome_30d: float | None = None


class PortfolioAllocationRecord(BaseModel):
    """Single asset allocation in rebalancing record."""

    symbol: str
    weight: float
    action: str
    delta: float


class PortfolioRebalancingRecord(BaseModel):
    """Record of portfolio rebalancing analysis."""

    timestamp: datetime
    method: str
    allocations: list[PortfolioAllocationRecord]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    rebalances_executed: int
    rebalances_pending: int


class OptimizationRecord(BaseModel):
    """Record of a parameter optimization run."""

    timestamp: datetime
    symbols_optimized: list[str]
    symbols_skipped: list[str]
    total_time_seconds: float


class PrefetchRecord(BaseModel):
    """Record of a data prefetch run."""

    timestamp: datetime
    symbols_prefetched: int
    symbols_failed: int
    finbert_ready: bool
    total_duration_seconds: float


class SectorRotationRecord(BaseModel):
    """Record of a sector rotation analysis run."""

    timestamp: datetime
    leading_sectors: list[str]
    lagging_sectors: list[str]
    sector_strengths: dict[str, float]
    sector_momenta: dict[str, str]
    flagged_positions: list[str]


class PeerAnalysisRecord(BaseModel):
    """Record of a deep peer benchmarking analysis run."""

    timestamp: datetime
    symbols_analyzed: list[str]
    rankings: dict[str, int]
    swap_recommendations: list[str]
    total_peers: int
    total_duration_seconds: float


class CorrelationAuditRecord(BaseModel):
    """Record of a portfolio correlation audit run."""

    timestamp: datetime
    num_positions: int
    num_correlated_pairs: int
    max_correlation: float
    avg_correlation: float
    diversification_ratio: float
    num_substitutions: int
    total_duration_seconds: float


class GamePlanRecord(BaseModel):
    """Record of game plan execution."""

    timestamp: datetime
    priority_symbols: list[str]
    risk_stance: str
    sector_focus: list[str]


class RiskReportRecord(BaseModel):
    """Record of a portfolio risk report."""

    timestamp: datetime
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    cdar_95: float
    max_drawdown: float
    risk_status: str


class MonteCarloRecord(BaseModel):
    """Record of Monte Carlo stress test execution."""

    timestamp: datetime
    simulation_method: str
    num_simulations: int
    horizon_days: int

    # Key results
    prob_loss_gt_threshold: float
    expected_worst_drawdown: float
    var_95: float
    cvar_95: float
    median_recovery_days: float | None

    # Alert status
    exceeds_risk_tolerance: bool
    alert_message: str | None

    # Portfolio snapshot
    portfolio_symbols: list[str]
    total_market_value: float


class EarningsEventRecord(BaseModel):
    """Record of a single earnings event."""

    symbol: str
    earnings_date: str
    estimate_eps: float | None = None


class EarningsCalendarRecord(BaseModel):
    """Record of an earnings calendar fetch run."""

    timestamp: datetime
    events: list[EarningsEventRecord]
    symbols_fetched: int
    symbols_failed: int


class DegradationRecord(BaseModel):
    """Record of degradation event."""

    timestamp: datetime
    tier: str
    unavailable_services: list[str]
    confidence_adjustment: float
    halt_reason: str | None = None


class DaemonState(BaseModel):
    """Persistent state for the trading daemon."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    last_run: datetime | None = None
    analyses: list[AnalysisRecord] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    total_analyses: int = 0
    total_trades: int = 0
    paper_trading_start_date: datetime | None = None
    current_trading_mode: str = "paper"
    last_journal_date: str | None = None
    last_after_hours_screening: datetime | None = None
    last_health_check: datetime | None = None
    screening_history: list[ScreeningRecord] = Field(default_factory=list)
    last_optimization: datetime | None = None
    optimization_history: list[OptimizationRecord] = Field(default_factory=list)
    last_prefetch: datetime | None = None
    last_pre_market_refresh: datetime | None = None
    prefetch_history: list[PrefetchRecord] = Field(default_factory=list)
    last_sector_rotation: datetime | None = None
    sector_rotation_history: list[SectorRotationRecord] = Field(default_factory=list)
    last_earnings_fetch: datetime | None = None
    earnings_calendar_history: list[EarningsCalendarRecord] = Field(default_factory=list)
    last_peer_analysis: datetime | None = None
    peer_analysis_history: list[PeerAnalysisRecord] = Field(default_factory=list)
    last_correlation_audit: datetime | None = None
    correlation_audit_history: list[CorrelationAuditRecord] = Field(default_factory=list)
    last_tearsheet: datetime | None = None
    last_risk_report: datetime | None = None
    risk_report_history: list[RiskReportRecord] = Field(default_factory=list)
    last_portfolio_rebalancing: datetime | None = None
    portfolio_rebalancing_history: list[PortfolioRebalancingRecord] = Field(default_factory=list)
    active_target_allocations: dict[str, float] | None = None
    last_signal_tracking: datetime | None = None
    last_game_plan: datetime | None = None
    game_plan_history: list[GamePlanRecord] = Field(default_factory=list)
    active_positions: dict[str, dict] = Field(default_factory=dict)
    position_management_history: list[dict] = Field(default_factory=list)
    monte_carlo_tests: list[MonteCarloRecord] = Field(default_factory=list)
    degradation_history: list[DegradationRecord] = Field(default_factory=list)
    last_degradation: datetime | None = None
    market_events: list[dict] = Field(default_factory=list)
    last_discovery: datetime | None = None
    discovery_history: list[DiscoveryHistoryRecord] = Field(default_factory=list)
    active_discovery_candidates: list[DiscoveryCandidate] = Field(default_factory=list)

    # Database repositories (private attributes - not serialized)
    _analysis_repository: "AnalysisRecordRepository | None" = PrivateAttr(default=None)
    _discovery_repository: "DiscoveryHistoryRepository | None" = PrivateAttr(default=None)
    _snapshot_repository: "PortfolioSnapshotRepository | None" = PrivateAttr(default=None)

    def set_repositories(
        self,
        analysis_repository: "AnalysisRecordRepository | None" = None,
        discovery_repository: "DiscoveryHistoryRepository | None" = None,
        snapshot_repository: "PortfolioSnapshotRepository | None" = None,
    ) -> None:
        """Inject database repositories after loading state.

        Args:
            analysis_repository: Analysis record repository
            discovery_repository: Discovery history repository
            snapshot_repository: Portfolio snapshot repository
        """
        self._analysis_repository = analysis_repository
        self._discovery_repository = discovery_repository
        self._snapshot_repository = snapshot_repository
        logger.debug("Repositories injected into DaemonState")

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

    def record_analysis(  # noqa: PLR0913
        self,
        symbol: str,
        signal: str,
        confidence: float,
        executed: bool = False,
        trading_session: TradingSession = TradingSession.REGULAR,
        is_paper_trade: bool = True,
        rsi: float | None = None,
        macd_hist: float | None = None,
        reasoning: list[str] | None = None,
    ) -> None:
        """Record an analysis result.

        Args:
            symbol: Stock ticker
            signal: Trading signal (BUY/SELL/HOLD)
            confidence: Signal confidence
            executed: Whether trade was executed
            trading_session: Trading session type (REGULAR/PRE_MARKET)
            is_paper_trade: Whether trade was paper or live
            rsi: RSI indicator value
            macd_hist: MACD histogram value
            reasoning: LLM decision reasoning
        """
        record = AnalysisRecord(
            symbol=symbol,
            timestamp=datetime.now(UTC),
            signal=signal,
            confidence=confidence,
            executed_trade=executed,
            trading_session=trading_session,
            is_paper_trade=is_paper_trade,
            rsi=rsi,
            macd_hist=macd_hist,
            reasoning=reasoning or [],
        )

        # Persist to database if repository available
        if self._analysis_repository:
            try:
                import asyncio

                task = asyncio.create_task(self._analysis_repository.create(record))
                task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
                logger.debug(f"Persisted analysis record to database: {symbol} {signal}")
            except Exception as e:
                logger.error(f"Failed to persist analysis record to database: {e}")
                raise  # Fail fast per user requirement

        # Keep in-memory list (capped for transition period)
        self.analyses.append(record)
        self.total_analyses += 1
        if executed:
            self.total_trades += 1
        self.last_run = datetime.now(UTC)

        if len(self.analyses) > 1000:
            self.analyses = self.analyses[-500:]

    def record_error(self, error: str) -> None:
        """Record an error.

        Args:
            error: Error message
        """
        timestamp = datetime.now(tz=UTC).isoformat()
        self.errors.append(f"{timestamp}: {error}")

        if len(self.errors) > 100:
            self.errors = self.errors[-50:]

    def record_degradation(self, context: "DegradationContext") -> None:
        """Record degradation event.

        Args:
            context: Degradation context
        """
        now = datetime.now(UTC)
        self.degradation_history.append(
            DegradationRecord(
                timestamp=now,
                tier=context.tier.value,
                unavailable_services=context.unavailable_services,
                confidence_adjustment=context.confidence_adjustment,
                halt_reason=context.halt_reason,
            )
        )
        self.last_degradation = now

        # Keep last 100 records
        if len(self.degradation_history) > 100:
            self.degradation_history = self.degradation_history[-100:]

    def record_after_hours_screening(
        self,
        criteria: str,
        universe: str,
        candidates: list[ScreeningResult],
        top_n: int = 10,
        screened_at: datetime | None = None,
    ) -> None:
        """Record after-hours screening results.

        Args:
            criteria: Screening criteria
            universe: Universe screened
            candidates: Candidate list (typically top-N from screening)
            top_n: Number of top symbols to track
            screened_at: Timestamp when screening was performed (defaults to now)
        """
        now = datetime.now(UTC)
        top_symbols = [c.symbol for c in candidates[:top_n]]

        self.screening_history.append(
            ScreeningRecord(
                timestamp=now,
                criteria=criteria,
                universe=universe,
                top_symbols=top_symbols,
                candidates=candidates[:top_n],
                screened_at=screened_at or now,
            )
        )
        self.last_after_hours_screening = now

        # Keep last 30 days (assume max 1 screening per day)
        if len(self.screening_history) > 30:
            self.screening_history = self.screening_history[-30:]

    def record_optimization(
        self,
        symbols_optimized: list[str],
        symbols_skipped: list[str],
        total_time_seconds: float,
    ) -> None:
        """Record a parameter optimization run.

        Args:
            symbols_optimized: Symbols that were optimized
            symbols_skipped: Symbols skipped (non-stale)
            total_time_seconds: Total optimization duration
        """
        now = datetime.now(UTC)

        self.optimization_history.append(
            OptimizationRecord(
                timestamp=now,
                symbols_optimized=symbols_optimized,
                symbols_skipped=symbols_skipped,
                total_time_seconds=total_time_seconds,
            )
        )
        self.last_optimization = now

        if len(self.optimization_history) > 10:
            self.optimization_history = self.optimization_history[-10:]

    def record_portfolio_rebalancing(  # noqa: PLR0913
        self,
        method: str,
        allocations: list[PortfolioAllocationRecord],
        expected_return: float,
        expected_volatility: float,
        sharpe_ratio: float,
        rebalances_executed: int,
        rebalances_pending: int,
    ) -> None:
        """Record portfolio rebalancing run.

        Args:
            method: Optimization method used
            allocations: Asset allocation records
            expected_return: Expected portfolio return
            expected_volatility: Expected portfolio volatility
            sharpe_ratio: Portfolio Sharpe ratio
            rebalances_executed: Number of rebalances executed
            rebalances_pending: Number of rebalances pending
        """
        now = datetime.now(UTC)

        self.portfolio_rebalancing_history.append(
            PortfolioRebalancingRecord(
                timestamp=now,
                method=method,
                allocations=allocations,
                expected_return=expected_return,
                expected_volatility=expected_volatility,
                sharpe_ratio=sharpe_ratio,
                rebalances_executed=rebalances_executed,
                rebalances_pending=rebalances_pending,
            )
        )
        self.last_portfolio_rebalancing = now

        if len(self.portfolio_rebalancing_history) > 30:
            self.portfolio_rebalancing_history = self.portfolio_rebalancing_history[-30:]

        self.active_target_allocations = {a.symbol: a.weight for a in allocations}

    def record_prefetch(
        self,
        symbols_prefetched: int,
        symbols_failed: int,
        finbert_ready: bool,
        total_duration_seconds: float,
    ) -> None:
        """Record a data prefetch run.

        Args:
            symbols_prefetched: Number of symbols successfully prefetched
            symbols_failed: Number of symbols that failed
            finbert_ready: Whether FinBERT was warmed up
            total_duration_seconds: Total prefetch duration
        """
        now = datetime.now(UTC)

        self.prefetch_history.append(
            PrefetchRecord(
                timestamp=now,
                symbols_prefetched=symbols_prefetched,
                symbols_failed=symbols_failed,
                finbert_ready=finbert_ready,
                total_duration_seconds=total_duration_seconds,
            )
        )
        self.last_prefetch = now

        if len(self.prefetch_history) > 30:
            self.prefetch_history = self.prefetch_history[-30:]

    def record_sector_rotation(
        self,
        leading_sectors: list[str],
        lagging_sectors: list[str],
        sector_strengths: dict[str, float],
        sector_momenta: dict[str, str],
        flagged_positions: list[str] | None = None,
    ) -> None:
        """Record a sector rotation analysis run.

        Args:
            leading_sectors: Top 3 sectors by relative strength
            lagging_sectors: Bottom 3 sectors by relative strength
            sector_strengths: Composite strength by sector name
            sector_momenta: Momentum classification by sector name
            flagged_positions: Symbols flagged in weak sectors
        """
        now = datetime.now(UTC)

        self.sector_rotation_history.append(
            SectorRotationRecord(
                timestamp=now,
                leading_sectors=leading_sectors,
                lagging_sectors=lagging_sectors,
                sector_strengths=sector_strengths,
                sector_momenta=sector_momenta,
                flagged_positions=flagged_positions or [],
            )
        )
        self.last_sector_rotation = now

        if len(self.sector_rotation_history) > 30:
            self.sector_rotation_history = self.sector_rotation_history[-30:]

    def record_earnings_fetch(
        self,
        events: list[EarningsEventRecord],
        symbols_fetched: int,
        symbols_failed: int,
    ) -> None:
        """Record an earnings calendar fetch run.

        Args:
            events: Earnings event records
            symbols_fetched: Number of symbols with earnings data
            symbols_failed: Number of symbols that failed to fetch
        """
        now = datetime.now(UTC)

        self.earnings_calendar_history.append(
            EarningsCalendarRecord(
                timestamp=now,
                events=events,
                symbols_fetched=symbols_fetched,
                symbols_failed=symbols_failed,
            )
        )
        self.last_earnings_fetch = now

        if len(self.earnings_calendar_history) > 10:
            self.earnings_calendar_history = self.earnings_calendar_history[-10:]

    def record_peer_analysis(
        self,
        symbols_analyzed: list[str],
        rankings: dict[str, int],
        swap_recommendations: list[str],
        total_peers: int,
        total_duration_seconds: float,
    ) -> None:
        """Record a deep peer benchmarking analysis run.

        Args:
            symbols_analyzed: Symbols that were analyzed
            rankings: Symbol to rank mapping
            swap_recommendations: Generated swap recommendations
            total_peers: Total number of peers analyzed
            total_duration_seconds: Total analysis duration
        """
        now = datetime.now(UTC)

        self.peer_analysis_history.append(
            PeerAnalysisRecord(
                timestamp=now,
                symbols_analyzed=symbols_analyzed,
                rankings=rankings,
                swap_recommendations=swap_recommendations,
                total_peers=total_peers,
                total_duration_seconds=total_duration_seconds,
            )
        )
        self.last_peer_analysis = now

        if len(self.peer_analysis_history) > 10:
            self.peer_analysis_history = self.peer_analysis_history[-10:]

    def record_correlation_audit(  # noqa: PLR0913
        self,
        num_positions: int,
        num_correlated_pairs: int,
        max_correlation: float,
        avg_correlation: float,
        diversification_ratio: float,
        num_substitutions: int,
        total_duration_seconds: float,
    ) -> None:
        """Record a correlation audit run.

        Args:
            num_positions: Number of positions analyzed
            num_correlated_pairs: Number of highly correlated pairs found
            max_correlation: Maximum correlation found
            avg_correlation: Average portfolio correlation
            diversification_ratio: Portfolio diversification ratio
            num_substitutions: Number of substitution suggestions
            total_duration_seconds: Total audit duration
        """
        now = datetime.now(UTC)

        self.correlation_audit_history.append(
            CorrelationAuditRecord(
                timestamp=now,
                num_positions=num_positions,
                num_correlated_pairs=num_correlated_pairs,
                max_correlation=max_correlation,
                avg_correlation=avg_correlation,
                diversification_ratio=diversification_ratio,
                num_substitutions=num_substitutions,
                total_duration_seconds=total_duration_seconds,
            )
        )
        self.last_correlation_audit = now

        if len(self.correlation_audit_history) > 10:
            self.correlation_audit_history = self.correlation_audit_history[-10:]

    def record_tearsheet(self, symbol: str, html_path: str) -> None:
        """Record a tearsheet generation run.

        Args:
            symbol: Stock ticker symbol
            html_path: Path to generated HTML tearsheet
        """
        now = datetime.now(UTC)
        self.last_tearsheet = now
        logger.info(f"Recorded tearsheet generation for {symbol} at {html_path}")

    def record_risk_report(self, report: RiskReportRecord) -> None:
        """Record a portfolio risk report.

        Args:
            report: Risk report record to store
        """
        self.risk_report_history.append(report)
        self.last_risk_report = report.timestamp

        if len(self.risk_report_history) > 30:
            self.risk_report_history = self.risk_report_history[-30:]

    def record_game_plan(
        self,
        priority_symbols: list[str],
        risk_stance: str,
        sector_focus: list[str],
    ) -> None:
        """Record game plan generation.

        Args:
            priority_symbols: Priority symbols for the day
            risk_stance: Risk stance (AGGRESSIVE/DEFENSIVE/NEUTRAL)
            sector_focus: Sector focus list
        """
        now = datetime.now(UTC)

        self.game_plan_history.append(
            GamePlanRecord(
                timestamp=now,
                priority_symbols=priority_symbols,
                risk_stance=risk_stance,
                sector_focus=sector_focus,
            )
        )
        self.last_game_plan = now

        if len(self.game_plan_history) > 30:
            self.game_plan_history = self.game_plan_history[-30:]

    def add_position(self, position: "PositionRecord") -> None:
        """Add or update position in state.

        Args:
            position: Position record to add
        """
        self.active_positions[position.symbol] = position.model_dump(mode="json")
        logger.debug(f"Added position: {position.symbol}")

    def remove_position(self, symbol: str) -> None:
        """Remove position from state.

        Args:
            symbol: Stock ticker to remove
        """
        if symbol in self.active_positions:
            self.active_positions.pop(symbol)
            logger.debug(f"Removed position: {symbol}")

    def update_position(self, position: "PositionRecord") -> None:
        """Update existing position in state.

        Args:
            position: Position record to update
        """
        self.add_position(position)

    def record_position_action(self, action: "PositionManagementAction") -> None:
        """Record position management action.

        Args:
            action: Action to record
        """
        self.position_management_history.append(action.model_dump(mode="json"))

        if len(self.position_management_history) > 100:
            self.position_management_history = self.position_management_history[-100:]

    def get_position(self, symbol: str) -> "PositionRecord | None":
        """Get position record by symbol.

        Args:
            symbol: Stock ticker

        Returns:
            PositionRecord or None
        """
        from src.daemon.positions import PositionRecord

        if symbol not in self.active_positions:
            return None
        return PositionRecord.model_validate(self.active_positions[symbol])

    def record_monte_carlo_test(self, record: MonteCarloRecord, max_records: int = 52) -> None:
        """Add Monte Carlo test record.

        Args:
            record: Monte Carlo test record
            max_records: Maximum records to retain (default 52)
        """
        self.monte_carlo_tests.append(record)
        if len(self.monte_carlo_tests) > max_records:
            self.monte_carlo_tests = self.monte_carlo_tests[-max_records:]

    def record_discovery(self, candidates: list[DiscoveryCandidate], added_symbols: list[str]) -> None:
        """Record discovery run and update active candidates.

        Args:
            candidates: Discovery candidates to record
            added_symbols: Symbols actually added to watchlist
        """
        # Add new history records
        for candidate in candidates:
            history_record = DiscoveryHistoryRecord(
                symbol=candidate.symbol,
                discovered_at=candidate.discovery_timestamp,
                composite_score=candidate.composite_score,
                sources=candidate.sources,
                added_to_watchlist=candidate.symbol in added_symbols,
                ttl_expires_at=candidate.ttl_expires_at,
            )

            # Persist to database if repository available
            if self._discovery_repository:
                try:
                    import asyncio

                    task = asyncio.create_task(self._discovery_repository.create(history_record))
                    task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
                    logger.debug(f"Persisted discovery history to database: {candidate.symbol}")
                except Exception as e:
                    logger.error(f"Failed to persist discovery history to database: {e}")
                    raise  # Fail fast per user requirement

            # Keep in-memory list (capped for transition period)
            self.discovery_history.append(history_record)

        # Update active candidates (replace old with new)
        self.active_discovery_candidates = candidates

        # Limit history to last 100 records
        if len(self.discovery_history) > 100:
            self.discovery_history = self.discovery_history[-100:]

        logger.info(f"Recorded discovery: {len(candidates)} candidates, {len(added_symbols)} added")

    def snapshot_portfolio(self, snapshot: PortfolioSnapshot) -> None:
        """Create portfolio snapshot and persist to database.

        Args:
            snapshot: Portfolio snapshot with balance, positions, and trigger info
        """
        if self._snapshot_repository:
            try:
                import asyncio

                from src.database.repositories.snapshot import PortfolioSnapshot as DBSnapshot

                db_snapshot = DBSnapshot(
                    timestamp=datetime.now(UTC),
                    balance=snapshot.balance,
                    available_cash=snapshot.available_cash,
                    total_exposure=snapshot.total_exposure,
                    portfolio_value=snapshot.portfolio_value,
                    positions=snapshot.positions,
                    trigger=snapshot.trigger,
                )
                task = asyncio.create_task(self._snapshot_repository.create(db_snapshot))
                task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
                logger.info(
                    f"Created portfolio snapshot: {snapshot.trigger} value={snapshot.portfolio_value}"
                )
            except Exception as e:
                logger.error(f"Failed to persist portfolio snapshot to database: {e}")
                raise

    def expire_stale_candidates(self) -> list[str]:
        """Remove candidates past TTL, return expired symbols.

        Returns:
            List of expired symbols
        """
        now = datetime.now(UTC)
        expired_symbols: list[str] = []

        # Filter out expired candidates
        active_candidates: list[DiscoveryCandidate] = []
        for candidate in self.active_discovery_candidates:
            ttl_expires_at = candidate.ttl_expires_at
            if ttl_expires_at.tzinfo is None:
                ttl_expires_at = ttl_expires_at.replace(tzinfo=UTC)
            if ttl_expires_at > now:
                active_candidates.append(candidate)
            else:
                expired_symbols.append(candidate.symbol)

        self.active_discovery_candidates = active_candidates

        if expired_symbols:
            logger.info(f"Expired {len(expired_symbols)} discovery candidates")

        return expired_symbols

    def get_active_discovery_symbols(self) -> list[str]:
        """Get symbols from active discovery candidates.

        Returns:
            List of active discovery symbols
        """
        return [c.symbol for c in self.active_discovery_candidates]

    def __repr__(self) -> str:
        """Return string representation."""
        return f"DaemonState(analyses={self.total_analyses}, trades={self.total_trades})"
