"""Paper trading validation framework for live promotion readiness."""

import json
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger
from pydantic import BaseModel

from src.daemon.config import PaperTradingConfig
from src.daemon.state import DaemonState
from src.metrics.tracker import BaseMetricsTracker, PerformanceMetrics


class ValidationCriterion(BaseModel):
    """Single validation criterion result."""

    name: str
    passed: bool
    current_value: float
    threshold: float
    message: str


class SimulatedLiveComparison(BaseModel):
    """Comparison between paper and simulated live trading."""

    paper_metrics: PerformanceMetrics
    live_metrics: PerformanceMetrics
    sharpe_delta: float
    total_pnl_delta: float
    win_rate_delta: float


class ReadinessReport(BaseModel):
    """Complete paper trading readiness assessment."""

    ready_for_live: bool
    assessment_date: datetime
    paper_trading_duration_days: int
    total_paper_trades: int
    criteria: list[ValidationCriterion]
    metrics: PerformanceMetrics
    simulated_live: SimulatedLiveComparison | None
    recommendations: list[str]


class PaperTradingValidator:
    """Validates paper trading readiness for live promotion."""

    ALPACA_FEE_PER_SHARE = 0.0003  # $0.0003/share
    SLIPPAGE_PERCENT = 0.05  # 0.05% slippage

    def __init__(
        self,
        config: PaperTradingConfig,
        state: DaemonState,
        metrics_tracker: BaseMetricsTracker,
    ) -> None:
        """Initialize validator.

        Args:
            config: Paper trading validation configuration
            state: Daemon state with paper trading history
            metrics_tracker: Metrics tracker for performance calculation
        """
        self.config = config
        self.state = state
        self.metrics_tracker = metrics_tracker
        logger.info("Initialized PaperTradingValidator")

    def assess_readiness(self) -> ReadinessReport:
        """Evaluate all validation criteria and generate readiness report.

        Returns:
            ReadinessReport with complete assessment
        """
        logger.info("Assessing paper trading readiness")

        # Calculate paper metrics
        paper_metrics = self._calculate_paper_metrics()

        # Run validation checks
        criteria = []
        criteria.append(self._check_duration())
        criteria.append(self._check_min_trades())
        criteria.append(self._check_sharpe(paper_metrics.sharpe_ratio))
        criteria.append(self._check_drawdown(paper_metrics.max_drawdown_percent))
        criteria.append(self._check_win_rate(paper_metrics.win_rate))

        # All criteria must pass
        ready = all(c.passed for c in criteria)

        # Calculate simulated live comparison
        simulated_live = None
        if paper_metrics.closed_trades > 0:
            try:
                simulated_live = self._simulate_live_comparison(paper_metrics)
            except Exception as e:
                logger.opt(exception=True).warning(f"Failed to simulate live comparison: {e}")

        # Generate recommendations
        recommendations = self._generate_recommendations(criteria, simulated_live)

        # Calculate duration
        if self.state.paper_trading_start_date:
            duration_days = (datetime.now(UTC) - self.state.paper_trading_start_date).days
        else:
            duration_days = 0

        report = ReadinessReport(
            ready_for_live=ready,
            assessment_date=datetime.now(UTC),
            paper_trading_duration_days=duration_days,
            total_paper_trades=paper_metrics.closed_trades,
            criteria=criteria,
            metrics=paper_metrics,
            simulated_live=simulated_live,
            recommendations=recommendations,
        )

        logger.info(f"Readiness assessment complete: ready={ready}")
        return report

    def _calculate_paper_metrics(self) -> PerformanceMetrics:
        """Calculate metrics for paper trades only, scoped to start date.

        Returns:
            PerformanceMetrics for paper trading period
        """
        # Filter to paper trades scoped by start date
        start_date = self.state.paper_trading_start_date
        paper_trades = []
        for t in self.metrics_tracker.trades:
            if not t.is_paper_trade:
                continue
            if start_date is not None and t.timestamp < start_date:
                continue
            paper_trades.append(t)

        if not paper_trades:
            return PerformanceMetrics(
                window="paper",
                total_decisions=0,
                approved_trades=0,
                rejected_trades=0,
                open_trades=0,
                closed_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                total_pnl=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                profit_factor=0.0,
                max_drawdown=0.0,
                max_drawdown_percent=0.0,
                sharpe_ratio=0.0,
                risk_adjusted_return=0.0,
                start_date=datetime.now(UTC),
                end_date=datetime.now(UTC),
            )

        # Restore full trades list after calculation
        original_trades = self.metrics_tracker.trades
        try:
            self.metrics_tracker.trades = paper_trades
            return self.metrics_tracker.calculate_metrics("all")
        finally:
            self.metrics_tracker.trades = original_trades

    def _simulate_live_trading(self) -> PerformanceMetrics:
        """Simulate live trading with fees and slippage applied to paper trades.

        Returns:
            PerformanceMetrics for simulated live trading
        """
        # Filter to paper trades scoped by start date
        start_date = self.state.paper_trading_start_date
        paper_trades = []
        for t in self.metrics_tracker.trades:
            if not t.is_paper_trade or not t.is_closed():
                continue
            if start_date is not None and t.timestamp < start_date:
                continue
            paper_trades.append(t)

        # Apply fees and slippage to each trade
        adjusted_trades = []
        for trade in paper_trades:
            # Calculate fees: entry + exit
            fee = trade.shares * self.ALPACA_FEE_PER_SHARE * 2

            # Calculate slippage on entry
            slippage = trade.shares * trade.entry_price * (self.SLIPPAGE_PERCENT / 100)

            # Adjust PnL
            adjusted_pnl = (trade.pnl or 0.0) - fee - slippage
            adjusted_pnl_percent = (adjusted_pnl / (trade.shares * trade.entry_price)) * 100

            adjusted_trade = trade.model_copy(
                update={"pnl": adjusted_pnl, "pnl_percent": adjusted_pnl_percent}
            )
            adjusted_trades.append(adjusted_trade)

        # Calculate metrics on adjusted trades
        original_trades = self.metrics_tracker.trades
        try:
            self.metrics_tracker.trades = adjusted_trades
            return self.metrics_tracker.calculate_metrics("all")
        finally:
            self.metrics_tracker.trades = original_trades

    def _simulate_live_comparison(self, paper_metrics: PerformanceMetrics) -> SimulatedLiveComparison:
        """Generate side-by-side comparison of paper vs simulated live.

        Args:
            paper_metrics: Paper trading metrics

        Returns:
            SimulatedLiveComparison with delta analysis
        """
        live_metrics = self._simulate_live_trading()

        return SimulatedLiveComparison(
            paper_metrics=paper_metrics,
            live_metrics=live_metrics,
            sharpe_delta=live_metrics.sharpe_ratio - paper_metrics.sharpe_ratio,
            total_pnl_delta=live_metrics.total_pnl - paper_metrics.total_pnl,
            win_rate_delta=live_metrics.win_rate - paper_metrics.win_rate,
        )

    def _check_duration(self) -> ValidationCriterion:
        """Check minimum paper trading duration."""
        if not self.state.paper_trading_start_date:
            return ValidationCriterion(
                name="Duration",
                passed=False,
                current_value=0.0,
                threshold=float(self.config.min_duration_days),
                message="Paper trading start date not set",
            )

        duration_days = (datetime.now(UTC) - self.state.paper_trading_start_date).days

        return ValidationCriterion(
            name="Duration",
            passed=duration_days >= self.config.min_duration_days,
            current_value=float(duration_days),
            threshold=float(self.config.min_duration_days),
            message=f"{duration_days}/{self.config.min_duration_days} days",
        )

    def _check_min_trades(self) -> ValidationCriterion:
        """Check minimum number of executed trades."""
        # Filter to paper trades scoped by start date
        start_date = self.state.paper_trading_start_date
        paper_trades = []
        for t in self.metrics_tracker.trades:
            if not t.is_paper_trade or t.status not in ("OPEN", "CLOSED"):
                continue
            if start_date is not None and t.timestamp < start_date:
                continue
            paper_trades.append(t)
        trade_count = len(paper_trades)

        return ValidationCriterion(
            name="Min Trades",
            passed=trade_count >= self.config.min_trades,
            current_value=float(trade_count),
            threshold=float(self.config.min_trades),
            message=f"{trade_count}/{self.config.min_trades} trades executed",
        )

    def _check_sharpe(self, sharpe: float) -> ValidationCriterion:
        """Check Sharpe ratio threshold."""
        return ValidationCriterion(
            name="Sharpe Ratio",
            passed=sharpe >= self.config.min_sharpe,
            current_value=sharpe,
            threshold=self.config.min_sharpe,
            message=f"Sharpe {sharpe:.2f} (min {self.config.min_sharpe:.2f})",
        )

    def _check_drawdown(self, drawdown_pct: float) -> ValidationCriterion:
        """Check maximum drawdown threshold."""
        return ValidationCriterion(
            name="Max Drawdown",
            passed=drawdown_pct <= self.config.max_drawdown_percent,
            current_value=drawdown_pct,
            threshold=self.config.max_drawdown_percent,
            message=f"Max DD {drawdown_pct:.1f}% (max {self.config.max_drawdown_percent:.1f}%)",
        )

    def _check_win_rate(self, win_rate: float) -> ValidationCriterion:
        """Check win rate threshold."""
        return ValidationCriterion(
            name="Win Rate",
            passed=win_rate >= self.config.min_win_rate,
            current_value=win_rate,
            threshold=self.config.min_win_rate,
            message=f"Win rate {win_rate:.1%} (min {self.config.min_win_rate:.1%})",
        )

    def _generate_recommendations(
        self,
        criteria: list[ValidationCriterion],
        simulated_live: SimulatedLiveComparison | None,
    ) -> list[str]:
        """Generate context-aware recommendations based on validation results.

        Args:
            criteria: Validation criteria results
            simulated_live: Simulated live comparison (optional)

        Returns:
            List of recommendation strings
        """
        recommendations = []
        failed_criteria = [c for c in criteria if not c.passed]

        if not failed_criteria:
            recommendations.append("All validation criteria met - ready for live promotion")
            if simulated_live:
                pnl_impact = abs(simulated_live.total_pnl_delta)
                recommendations.append(
                    f"Expected fees/slippage impact: ${pnl_impact:.2f} "
                    f"({simulated_live.sharpe_delta:+.2f} Sharpe)"
                )
            return recommendations

        # Duration recommendations
        duration_failed = next((c for c in failed_criteria if c.name == "Duration"), None)
        if duration_failed:
            days_remaining = int(duration_failed.threshold - duration_failed.current_value)
            recommendations.append(f"Continue paper trading for {days_remaining} more days")

        # Trade count recommendations
        trades_failed = next((c for c in failed_criteria if c.name == "Min Trades"), None)
        if trades_failed:
            trades_remaining = int(trades_failed.threshold - trades_failed.current_value)
            recommendations.append(f"Execute {trades_remaining} more trades before promotion")

        # Performance recommendations
        sharpe_failed = next((c for c in failed_criteria if c.name == "Sharpe Ratio"), None)
        if sharpe_failed:
            recommendations.append(
                f"Improve risk-adjusted returns (Sharpe {sharpe_failed.current_value:.2f} "
                f"< {sharpe_failed.threshold:.2f})"
            )

        drawdown_failed = next((c for c in failed_criteria if c.name == "Max Drawdown"), None)
        if drawdown_failed:
            recommendations.append(
                f"Reduce drawdown exposure (current {drawdown_failed.current_value:.1f}% "
                f"> {drawdown_failed.threshold:.1f}%)"
            )

        win_rate_failed = next((c for c in failed_criteria if c.name == "Win Rate"), None)
        if win_rate_failed:
            recommendations.append(
                f"Improve trade selection (win rate {win_rate_failed.current_value:.1%} "
                f"< {win_rate_failed.threshold:.1%})"
            )

        return recommendations

    def save_report(
        self, report: ReadinessReport, path: str = "~/.ai-casino/paper-trading-report.json"
    ) -> None:
        """Save readiness report to JSON file.

        Args:
            report: Readiness report to save
            path: Path to save report (supports ~ expansion)
        """
        expanded_path = Path(path).expanduser()
        expanded_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with expanded_path.open("w") as f:
                json.dump(report.model_dump(mode="json"), f, indent=2, default=str)
            logger.info(f"Saved paper trading report to {expanded_path}")
        except Exception as e:
            logger.opt(exception=True).error(f"Failed to save report: {e}")
            raise

    def __repr__(self) -> str:
        """String representation."""
        return f"PaperTradingValidator(min_duration={self.config.min_duration_days}d)"
