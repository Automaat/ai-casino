"""Daemon state persistence."""

import json
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger
from pydantic import BaseModel, Field

from src.screening.screener import ScreeningResult
from src.strategies.session import TradingSession

if False:
    from src.daemon.positions import PositionManagementAction, PositionRecord


class AnalysisRecord(BaseModel):
    """Record of a single analysis run."""

    symbol: str
    timestamp: datetime
    signal: str
    confidence: float
    executed_trade: bool = False
    trading_session: TradingSession = TradingSession.REGULAR


class ScreeningRecord(BaseModel):
    """Record of an after-hours screening run."""

    timestamp: datetime
    criteria: str
    universe: str
    top_symbols: list[str]
    candidates: list[ScreeningResult]
    screened_at: datetime


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


class DaemonState(BaseModel):
    """Persistent state for the trading daemon."""

    last_run: datetime | None = None
    analyses: list[AnalysisRecord] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    total_analyses: int = 0
    total_trades: int = 0
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
                timestamp=datetime.now(UTC),
                signal=signal,
                confidence=confidence,
                executed_trade=executed,
                trading_session=trading_session,
            )
        )
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
        timestamp = datetime.now().isoformat()  # noqa: DTZ005
        self.errors.append(f"{timestamp}: {error}")

        if len(self.errors) > 100:
            self.errors = self.errors[-50:]

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

    def __repr__(self) -> str:
        """Return string representation."""
        return f"DaemonState(analyses={self.total_analyses}, trades={self.total_trades})"
