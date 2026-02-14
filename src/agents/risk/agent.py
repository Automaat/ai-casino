"""Risk Management Agent."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from loguru import logger

if TYPE_CHECKING:
    from src.daemon.config import PositionSizingConfig
    from src.daemon.degradation import DegradationContext
    from src.database.repositories.risk_audit import RiskAuditRepository
    from src.workflows.types import BacktestValidation

from src.agents.risk.context import RiskContext
from src.agents.risk.models import (
    AccountInfo,
    PortfolioRiskReport,
    PortfolioVaRConfig,
    PositionSizeCalculation,
    RiskAssessment,
    RiskAuditRecord,
    RiskValidation,
    StopLossCalculation,
)
from src.agents.risk.position_sizer import PositionSizer
from src.agents.risk.stop_loss_calculator import StopLossCalculator
from src.data.broker import BrokerPosition
from src.metrics.portfolio_var import PortfolioVaRCalculator
from src.models.llm import LLMClient
from src.strategies.signal import Signal


class RiskManagementAgent:
    """Agent for position sizing, stop-loss, and risk validation."""

    MAX_POSITION_RISK_PERCENT = 2.0
    MAX_TOTAL_EXPOSURE_PERCENT = 80.0
    MAX_SINGLE_POSITION_PERCENT = 20.0
    DEFAULT_STOP_LOSS_PERCENT = 2.0
    ATR_MULTIPLIER = 2.0
    TRAILING_STOP_PERCENT = 3.0
    TRAILING_ACTIVATION_PERCENT = 5.0
    MIN_DECISION_CONFIDENCE = 0.6
    RISK_LEVEL_LOW_THRESHOLD = 0.75
    RISK_LEVEL_MEDIUM_THRESHOLD = 0.5
    REJECTED_CONFIDENCE_PENALTY = 0.3
    RISK_SCORE_WEIGHT = 0.6
    DECISION_CONFIDENCE_WEIGHT = 0.4

    def __init__(
        self,
        llm_client: LLMClient,
        max_position_risk: float | None = None,
        max_exposure: float | None = None,
        max_single_position: float | None = None,
        enable_trailing_stop: bool = True,
        portfolio_var_calculator: PortfolioVaRCalculator | None = None,
        portfolio_var_config: PortfolioVaRConfig | None = None,
        position_sizing_config: PositionSizingConfig | None = None,
        audit_repository: RiskAuditRepository | None = None,
    ) -> None:
        """Initialize risk management agent.

        Args:
            llm_client: LLM client for risk interpretation
            max_position_risk: Override max risk per trade (%)
            max_exposure: Override max total exposure (%)
            max_single_position: Override max single position size (%)
            enable_trailing_stop: Enable trailing stop-loss
            portfolio_var_calculator: Optional VaR calculator for portfolio-level limits
            portfolio_var_config: Optional VaR limit configuration
            position_sizing_config: Optional position sizing config (takes priority over individual params)
            audit_repository: Optional repository for database audit logging
        """
        self.llm = llm_client
        self._audit_repository = audit_repository
        self.max_position_risk = (
            position_sizing_config.max_risk_per_trade_pct
            if position_sizing_config
            else (max_position_risk or self.MAX_POSITION_RISK_PERCENT)
        )
        self.max_exposure = (
            position_sizing_config.max_total_exposure_pct
            if position_sizing_config
            else (max_exposure or self.MAX_TOTAL_EXPOSURE_PERCENT)
        )
        self.max_single_position = (
            position_sizing_config.max_single_position_pct
            if position_sizing_config
            else (max_single_position or self.MAX_SINGLE_POSITION_PERCENT)
        )
        self.enable_trailing_stop = enable_trailing_stop
        self._var_calculator = portfolio_var_calculator
        self._var_config = portfolio_var_config or PortfolioVaRConfig()

        # Create position sizer component with resolved config
        self._position_sizer = PositionSizer(
            max_position_risk=self.max_position_risk,
            max_exposure=self.max_exposure,
            max_single_position=self.max_single_position,
        )

        # Create stop-loss calculator component with resolved config
        self._stop_loss_calculator = StopLossCalculator(
            enable_trailing_stop=self.enable_trailing_stop,
            var_config=self._var_config,
            atr_multiplier=self.ATR_MULTIPLIER,
            default_stop_percent=self.DEFAULT_STOP_LOSS_PERCENT,
            trailing_stop_percent=self.TRAILING_STOP_PERCENT,
            trailing_activation_percent=self.TRAILING_ACTIVATION_PERCENT,
        )

        self.audit_log_path = Path("logs/risk_audit.jsonl")
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)

        var_str = f", var_limits={'ON' if self._var_config.enabled else 'OFF'}"
        logger.info(
            f"Initialized RiskManagementAgent "
            f"(max_risk={self.max_position_risk}%, max_exposure={self.max_exposure}%, "
            f"max_single={self.max_single_position}%, trailing={enable_trailing_stop}{var_str})"
        )

    def assess(
        self,
        symbol: str,
        action: Signal,
        current_price: float,
        account_info: AccountInfo,
        market_data: pd.DataFrame,
        decision_confidence: float,
        broker_positions: dict[str, BrokerPosition] | None = None,
        portfolio_value: float | None = None,
        target_portfolio_weight: float | None = None,
        backtest_validation: BacktestValidation | None = None,
        degradation_context: DegradationContext | None = None,
        broker_api_failed: bool = False,
    ) -> RiskAssessment:
        """Perform complete risk assessment.

        Args:
            symbol: Stock ticker
            action: Proposed trading action
            current_price: Current stock price
            account_info: Account balance and positions
            market_data: OHLCV data for volatility analysis
            decision_confidence: Trading decision confidence
            broker_positions: Optional broker positions for VaR calculation
            portfolio_value: Optional portfolio value for VaR calculation
            target_portfolio_weight: Optional target portfolio weight for allocation-based sizing
            backtest_validation: Optional pre-trade backtest validation result
            degradation_context: Optional degradation context
            broker_api_failed: True if broker API failed during account fetch

        Returns:
            RiskAssessment with sizing, stop-loss, validation
        """
        logger.info(f"Assessing risk for {action.value} {symbol} @ ${current_price:.2f}")

        context = RiskContext()

        if action == Signal.HOLD:
            assessment = self._hold_assessment(symbol, current_price, account_info)
        else:
            stop_loss = self._stop_loss_calculator.calculate(current_price, market_data, action, context)

            position_sizing = self._position_sizer.calculate(
                current_price, stop_loss, account_info, target_portfolio_weight
            )

            stop_loss.max_loss_amount = position_sizing.risk_amount

            validation = self._validate_risk(
                symbol,
                action,
                position_sizing,
                account_info,
                decision_confidence,
                context,
                broker_positions=broker_positions,
                portfolio_value=portfolio_value,
                backtest_validation=backtest_validation,
                broker_api_failed=broker_api_failed,
            )

            confidence = self._calculate_risk_confidence(validation, decision_confidence)

            # Apply degradation adjustment to confidence if provided
            if degradation_context and degradation_context.confidence_adjustment < 1.0:
                confidence = confidence * degradation_context.confidence_adjustment
                degradation_penalty_pct = (1 - degradation_context.confidence_adjustment) * 100
                logger.info(
                    f"Applied degradation penalty: -{degradation_penalty_pct:.0f}% "
                    f"(adjusted confidence: {confidence:.2f})"
                )

            logger.info(
                f"Risk assessment: {validation.risk_level} risk, "
                f"approved={validation.approved}, confidence={confidence:.2f}"
            )

            assessment = RiskAssessment(
                symbol=symbol,
                action=action,
                current_price=current_price,
                account_info=account_info,
                position_sizing=position_sizing,
                stop_loss=stop_loss,
                validation=validation,
                confidence=confidence,
                portfolio_var=context.latest_portfolio_var,
            )

        return assessment

    def _validate_exposure(
        self,
        action: Signal,
        position_sizing: PositionSizeCalculation,
        account_info: AccountInfo,
        warnings: list[str],
    ) -> tuple[bool, float]:
        """Validate exposure constraint."""
        new_exposure = (
            account_info.total_exposure + position_sizing.position_value
            if action == Signal.BUY
            else account_info.total_exposure - position_sizing.position_value
        )
        exposure_percent = (new_exposure / account_info.balance) * 100 if account_info.balance > 0 else 0.0
        met = exposure_percent <= self.max_exposure
        if not met:
            warnings.append(f"Total exposure {exposure_percent:.1f}% exceeds max {self.max_exposure}%")
        return met, exposure_percent

    def _validate_cash(
        self,
        action: Signal,
        position_sizing: PositionSizeCalculation,
        account_info: AccountInfo,
        warnings: list[str],
    ) -> bool:
        """Validate cash availability for BUY actions."""
        if action == Signal.BUY:
            met = position_sizing.position_value <= account_info.available_cash
            if not met:
                warnings.append(
                    f"Insufficient cash: need ${position_sizing.position_value:.2f}, "
                    f"have ${account_info.available_cash:.2f}"
                )
            return met
        return True

    def _validate_position_ownership(
        self, action: Signal, symbol: str, account_info: AccountInfo, warnings: list[str]
    ) -> bool:
        """Validate position ownership constraints."""
        has_position = symbol in account_info.positions
        if action == Signal.BUY:
            met = not has_position
            if has_position:
                warnings.append(f"Already have position in {symbol}")
            return met
        met = has_position
        if not has_position:
            warnings.append(f"No position in {symbol} to sell")
        return met

    def _validate_risk(
        self,
        symbol: str,
        action: Signal,
        position_sizing: PositionSizeCalculation,
        account_info: AccountInfo,
        decision_confidence: float,
        context: RiskContext,
        broker_positions: dict[str, BrokerPosition] | None = None,
        portfolio_value: float | None = None,
        backtest_validation: BacktestValidation | None = None,
        broker_api_failed: bool = False,
    ) -> RiskValidation:
        """Validate risk constraints and generate approval (populates context).

        Args:
            symbol: Stock ticker
            action: Trading action
            position_sizing: Position sizing calculation
            account_info: Account information
            decision_confidence: Decision confidence score
            context: Risk context to populate with portfolio VaR data
            broker_positions: Optional broker positions for VaR check
            portfolio_value: Optional portfolio value for VaR check
            backtest_validation: Optional pre-trade backtest validation result
            broker_api_failed: True if broker API failed during account fetch

        Returns:
            RiskValidation with approval status
        """
        warnings = []
        constraints_met = {}

        # Broker API failure check (highest priority)
        constraints_met["broker_available"] = not broker_api_failed
        if broker_api_failed:
            warnings.append(
                "Broker API unavailable - cannot verify account balance or positions. "
                "Trade execution blocked to prevent incorrect sizing."
            )

        constraints_met["position_risk"] = position_sizing.risk_percent <= self.max_position_risk
        if not constraints_met["position_risk"]:
            warnings.append(
                f"Position risk {position_sizing.risk_percent:.2f}% exceeds max {self.max_position_risk}%"
            )

        constraints_met["total_exposure"], exposure_percent = self._validate_exposure(
            action, position_sizing, account_info, warnings
        )

        constraints_met["cash_available"] = self._validate_cash(
            action, position_sizing, account_info, warnings
        )

        constraints_met["confidence"] = decision_confidence >= self.MIN_DECISION_CONFIDENCE
        if not constraints_met["confidence"]:
            warnings.append(f"Low decision confidence: {decision_confidence:.2f}")

        if action == Signal.BUY:
            constraints_met["no_duplicate"] = self._validate_position_ownership(
                action, symbol, account_info, warnings
            )
        else:
            constraints_met["has_position_to_sell"] = self._validate_position_ownership(
                action, symbol, account_info, warnings
            )

        if self._var_config.enabled and self._var_calculator:
            constraints_met["portfolio_var"] = self._validate_portfolio_var(
                symbol,
                action,
                position_sizing.position_value,
                broker_positions or {},
                portfolio_value or account_info.balance,
                warnings,
                context,
            )

        approved = all(constraints_met.values())

        risk_score = self._calculate_risk_score(
            position_sizing.risk_percent,
            exposure_percent,
            decision_confidence,
            backtest_validation,
        )

        if risk_score >= self.RISK_LEVEL_LOW_THRESHOLD:
            risk_level = "LOW"
        elif risk_score >= self.RISK_LEVEL_MEDIUM_THRESHOLD:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"

        reasoning = (
            f"{'APPROVED' if approved else 'REJECTED'}: {len(warnings)} warnings, "
            f"risk_score={risk_score:.2f}. "
            f"Constraints: {sum(constraints_met.values())}/{len(constraints_met)} met."
        )

        return RiskValidation(
            approved=approved,
            risk_score=risk_score,
            risk_level=risk_level,
            warnings=warnings,
            constraints_met=constraints_met,
            reasoning=reasoning,
        )

    def _validate_portfolio_var(
        self,
        symbol: str,
        action: Signal,
        position_value: float,
        broker_positions: dict[str, BrokerPosition],
        portfolio_value: float,
        warnings: list[str],
        context: RiskContext,
    ) -> bool:
        """Validate portfolio-level VaR limits (populates context).

        Args:
            symbol: Stock ticker
            action: Trading action
            position_value: Proposed position value
            broker_positions: Current broker positions
            portfolio_value: Current portfolio value
            warnings: Warning list to append to
            context: Risk context to populate with portfolio_cdar and latest_portfolio_var

        Returns:
            True if within limits
        """
        if action == Signal.SELL:
            return True

        if not self._var_calculator:
            return True

        try:
            var_result = self._var_calculator.calculate_with_hypothetical(
                broker_positions, portfolio_value, symbol, position_value, self._var_config.lookback_days
            )
            context.latest_portfolio_var = var_result

            if not var_result.sufficient_data:
                warnings.append("Insufficient data for portfolio VaR check, approving trade")
                return True

            context.portfolio_cdar = var_result.cdar_95

            var_ok = var_result.var_95 <= self._var_config.max_var_95
            cvar_ok = var_result.cvar_99 <= self._var_config.max_cvar_99

            if not var_ok:
                warnings.append(
                    f"Portfolio VaR95 {var_result.var_95:.4f} exceeds limit {self._var_config.max_var_95:.4f}"
                )
            if not cvar_ok:
                warnings.append(
                    f"Portfolio CVaR99 {var_result.cvar_99:.4f} exceeds limit {self._var_config.max_cvar_99:.4f}"
                )

            return var_ok and cvar_ok
        except Exception as e:
            logger.opt(exception=True).error(f"Portfolio VaR validation failed: {e}")
            warnings.append(f"Portfolio VaR check failed: {e}")
            return True

    def generate_risk_report(
        self,
        broker_positions: dict[str, BrokerPosition],
        portfolio_value: float,
        total_exposure: float,
        lookback_days: int = 90,
    ) -> PortfolioRiskReport:
        """Generate daily portfolio risk report.

        Args:
            broker_positions: Current broker positions
            portfolio_value: Total portfolio value
            total_exposure: Current total exposure
            lookback_days: Historical lookback period

        Returns:
            PortfolioRiskReport with current risk state
        """
        exposure_percent = (total_exposure / portfolio_value * 100) if portfolio_value > 0 else 0.0

        if not self._var_calculator or not broker_positions:
            return PortfolioRiskReport(
                date=datetime.now(UTC).date().isoformat(),
                var_95=0.0,
                var_99=0.0,
                cvar_95=0.0,
                cvar_99=0.0,
                cdar_95=0.0,
                max_drawdown=0.0,
                portfolio_volatility=0.0,
                current_exposure_percent=exposure_percent,
                num_positions=len(broker_positions) if broker_positions else 0,
                var_limit_breached=False,
                cvar_limit_breached=False,
                risk_status="HEALTHY",
            )

        var_result = self._var_calculator.calculate(broker_positions, portfolio_value, lookback_days)

        var_breached = var_result.var_95 > self._var_config.max_var_95
        cvar_breached = var_result.cvar_99 > self._var_config.max_cvar_99

        if var_breached or cvar_breached:
            risk_status = "BREACH"
        elif var_result.var_95 > self._var_config.max_var_95 * 0.8:
            risk_status = "WARNING"
        else:
            risk_status = "HEALTHY"

        return PortfolioRiskReport(
            date=datetime.now(UTC).date().isoformat(),
            var_95=var_result.var_95,
            var_99=var_result.var_99,
            cvar_95=var_result.cvar_95,
            cvar_99=var_result.cvar_99,
            cdar_95=var_result.cdar_95,
            max_drawdown=var_result.max_drawdown,
            portfolio_volatility=var_result.portfolio_volatility,
            current_exposure_percent=exposure_percent,
            num_positions=var_result.num_positions,
            var_limit_breached=var_breached,
            cvar_limit_breached=cvar_breached,
            risk_status=risk_status,
        )

    def _calculate_risk_score(
        self,
        risk_percent: float,
        exposure_percent: float,
        confidence: float,
        backtest_validation: BacktestValidation | None = None,
    ) -> float:
        """Calculate overall risk score (0.0-1.0, higher = safer).

        Args:
            risk_percent: Position risk percentage
            exposure_percent: Total exposure percentage
            confidence: Decision confidence
            backtest_validation: Optional backtest validation result

        Returns:
            Risk score (0.0-1.0)
        """
        risk_component = 1.0 - (risk_percent / self.max_position_risk)
        exposure_component = 1.0 - (exposure_percent / self.max_exposure)

        risk_component = max(0.0, min(1.0, risk_component))
        exposure_component = max(0.0, min(1.0, exposure_component))

        backtest_penalty = 0.0
        if backtest_validation and not backtest_validation.passed:
            backtest_penalty = 0.3

        base_score = risk_component * 0.3 + exposure_component * 0.3 + confidence * 0.4
        score = base_score - backtest_penalty

        return max(0.0, min(1.0, score))

    def _hold_assessment(
        self,
        symbol: str,
        current_price: float,
        account_info: AccountInfo,
    ) -> RiskAssessment:
        """Return minimal assessment for HOLD action.

        Args:
            symbol: Stock ticker
            current_price: Current price
            account_info: Account information

        Returns:
            RiskAssessment for HOLD
        """
        return RiskAssessment(
            symbol=symbol,
            action=Signal.HOLD,
            current_price=current_price,
            account_info=account_info,
            position_sizing=PositionSizeCalculation(
                recommended_shares=0,
                position_value=0.0,
                risk_amount=0.0,
                risk_percent=0.0,
                reasoning="No position change - HOLD",
            ),
            stop_loss=StopLossCalculation(
                stop_loss_price=current_price,
                stop_loss_percent=0.0,
                risk_per_share=0.0,
                max_loss_amount=0.0,
                methodology="N/A (HOLD)",
                trailing_stop=None,
            ),
            validation=RiskValidation(
                approved=True,
                risk_score=1.0,
                risk_level="LOW",
                warnings=[],
                constraints_met={},
                reasoning="No risk - HOLD action",
            ),
            confidence=1.0,
        )

    def _calculate_risk_confidence(
        self,
        validation: RiskValidation,
        decision_confidence: float,
    ) -> float:
        """Calculate overall confidence in risk assessment.

        Args:
            validation: Risk validation result
            decision_confidence: Decision confidence

        Returns:
            Overall confidence (0.0-1.0)
        """
        if not validation.approved:
            return max(0.0, validation.risk_score - self.REJECTED_CONFIDENCE_PENALTY)

        return (
            validation.risk_score * self.RISK_SCORE_WEIGHT
            + decision_confidence * self.DECISION_CONFIDENCE_WEIGHT
        )

    async def _audit_log(self, assessment: RiskAssessment) -> None:
        """Log risk assessment to database or JSONL fallback.

        Logs to database if audit_repository configured, otherwise JSONL file.

        Args:
            assessment: Risk assessment to log
        """
        try:
            record = RiskAuditRecord(
                timestamp=datetime.now(UTC),
                symbol=assessment.symbol,
                action=assessment.action,
                current_price=assessment.current_price,
                approved=assessment.validation.approved,
                risk_level=assessment.validation.risk_level,
                risk_score=assessment.validation.risk_score,
                confidence=assessment.confidence,
                recommended_shares=assessment.position_sizing.recommended_shares,
                position_value=assessment.position_sizing.position_value,
                risk_amount=assessment.position_sizing.risk_amount,
                risk_percent=assessment.position_sizing.risk_percent,
                stop_loss_price=assessment.stop_loss.stop_loss_price,
                warnings=assessment.validation.warnings,
                portfolio_var_95=assessment.portfolio_var.var_95 if assessment.portfolio_var else None,
                portfolio_cvar_99=assessment.portfolio_var.cvar_99 if assessment.portfolio_var else None,
                portfolio_cdar_95=assessment.portfolio_var.cdar_95 if assessment.portfolio_var else None,
            )

            if self._audit_repository:
                try:
                    await self._audit_repository.create(record)
                    logger.debug(f"Audit logged to DB: {assessment.symbol} {assessment.action.value}")
                except Exception as e:
                    logger.opt(exception=True).error(f"DB audit failed, falling back to JSONL: {e}")
                    log_entry = record.model_dump(mode="json", exclude={"id", "created_at"})
                    with self.audit_log_path.open("a") as f:
                        f.write(json.dumps(log_entry) + "\n")
            else:
                log_entry = record.model_dump(mode="json", exclude={"id", "created_at"})

                with self.audit_log_path.open("a") as f:
                    f.write(json.dumps(log_entry) + "\n")
                logger.debug(f"Audit logged to JSONL: {assessment.symbol} {assessment.action.value}")

        except Exception as e:
            logger.opt(exception=True).error(f"Audit logging failed: {e}")

    # Delegation methods for backward compatibility with tests
    def _get_atr(self, market_data: pd.DataFrame, period: int = 14) -> float | None:
        """Delegate to stop-loss calculator."""
        return self._stop_loss_calculator._get_atr(market_data, period)  # noqa: SLF001

    def _calculate_stop_loss(
        self, current_price: float, market_data: pd.DataFrame, action: Signal
    ) -> StopLossCalculation:
        """Delegate to stop-loss calculator."""
        return self._stop_loss_calculator.calculate(current_price, market_data, action, None)

    def _get_adaptive_atr_multiplier(self) -> float:
        """Delegate to stop-loss calculator with current CDaR context."""
        context = RiskContext(portfolio_cdar=getattr(self, "_portfolio_cdar", None))
        return self._stop_loss_calculator._get_adaptive_multiplier(context)  # noqa: SLF001

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"RiskManagementAgent(max_risk={self.max_position_risk}%, "
            f"max_exposure={self.max_exposure}%, trailing={self.enable_trailing_stop})"
        )
