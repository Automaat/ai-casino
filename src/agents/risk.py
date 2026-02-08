"""Risk Management Agent."""

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from src.daemon.degradation import DegradationContext
    from src.workflows.types import BacktestValidation

from src.data.broker import BrokerPosition
from src.metrics.portfolio_var import PortfolioVaRCalculator, PortfolioVaRResult
from src.models.llm import LLMClient
from src.strategies.signal import Signal


class AccountInfo(BaseModel):
    """Account information for risk calculations."""

    balance: float
    available_cash: float
    positions: dict[str, float]
    total_exposure: float


class PositionSizeCalculation(BaseModel):
    """Position sizing result."""

    recommended_shares: int
    position_value: float
    risk_amount: float
    risk_percent: float
    reasoning: str


class TrailingStopConfig(BaseModel):
    """Trailing stop-loss configuration."""

    enabled: bool
    trail_percent: float
    activation_percent: float


class StopLossCalculation(BaseModel):
    """Stop-loss calculation result."""

    stop_loss_price: float
    stop_loss_percent: float
    risk_per_share: float
    max_loss_amount: float
    methodology: str
    trailing_stop: TrailingStopConfig | None = None


class RiskValidation(BaseModel):
    """Risk validation result."""

    approved: bool
    risk_score: float
    risk_level: str
    warnings: list[str]
    constraints_met: dict[str, bool]
    reasoning: str


class PortfolioVaRConfig(BaseModel):
    """Configuration for portfolio-level VaR limits."""

    enabled: bool = False
    max_var_95: float = Field(default=0.03, ge=0.001, le=0.20)
    max_cvar_99: float = Field(default=0.05, ge=0.001, le=0.30)
    lookback_days: int = Field(default=90, ge=20, le=365)
    adaptive_stop_loss: bool = True
    cdar_stop_threshold: float = Field(default=0.10, ge=0.01, le=0.50)
    atr_multiplier_min: float = Field(default=1.0, ge=0.5, le=2.0)


class PortfolioRiskReport(BaseModel):
    """Daily portfolio risk report."""

    date: str
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    cdar_95: float
    max_drawdown: float
    portfolio_volatility: float
    current_exposure_percent: float
    num_positions: int
    var_limit_breached: bool
    cvar_limit_breached: bool
    risk_status: str


class RiskAssessment(BaseModel):
    """Complete risk management assessment."""

    symbol: str
    action: Signal
    current_price: float
    account_info: AccountInfo
    position_sizing: PositionSizeCalculation
    stop_loss: StopLossCalculation
    validation: RiskValidation
    confidence: float
    portfolio_var: PortfolioVaRResult | None = None


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
        """
        self.llm = llm_client
        self.max_position_risk = max_position_risk or float(
            os.getenv("MAX_POSITION_RISK", str(self.MAX_POSITION_RISK_PERCENT))
        )
        self.max_exposure = max_exposure or float(
            os.getenv("MAX_EXPOSURE", str(self.MAX_TOTAL_EXPOSURE_PERCENT))
        )
        self.max_single_position = max_single_position or float(
            os.getenv("MAX_SINGLE_POSITION", str(self.MAX_SINGLE_POSITION_PERCENT))
        )
        self.enable_trailing_stop = enable_trailing_stop
        self._var_calculator = portfolio_var_calculator
        self._var_config = portfolio_var_config or PortfolioVaRConfig()
        self._portfolio_cdar: float | None = None
        self._latest_portfolio_var: PortfolioVaRResult | None = None

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
        backtest_validation: "BacktestValidation | None" = None,
        degradation_context: "DegradationContext | None" = None,  # noqa: ARG002
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

        self._portfolio_cdar = None
        self._latest_portfolio_var = None

        if action == Signal.HOLD:
            assessment = self._hold_assessment(symbol, current_price, account_info)
        else:
            stop_loss = self._calculate_stop_loss(current_price, market_data, action)

            position_sizing = self._calculate_position_size(
                current_price, stop_loss, account_info, target_portfolio_weight
            )

            stop_loss.max_loss_amount = position_sizing.risk_amount

            validation = self._validate_risk(
                symbol,
                action,
                position_sizing,
                account_info,
                decision_confidence,
                broker_positions=broker_positions,
                portfolio_value=portfolio_value,
                backtest_validation=backtest_validation,
                broker_api_failed=broker_api_failed,
            )

            confidence = self._calculate_risk_confidence(validation, decision_confidence)

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
                portfolio_var=self._latest_portfolio_var,
            )

        self._audit_log(assessment)

        return assessment

    def _calculate_stop_loss(
        self,
        current_price: float,
        market_data: pd.DataFrame,
        action: Signal,
    ) -> StopLossCalculation:
        """Calculate stop-loss price using ATR or fixed %.

        Args:
            current_price: Current price
            market_data: OHLCV data
            action: Trading action (BUY/SELL)

        Returns:
            StopLossCalculation with stop price and methodology
        """
        atr = self._get_atr(market_data)
        atr_multiplier = self._get_adaptive_atr_multiplier()

        if atr and atr > 0:
            stop_distance = atr * atr_multiplier
            if action == Signal.BUY:
                stop_loss_price = current_price - stop_distance
            else:
                stop_loss_price = current_price + stop_distance
            methodology = f"ATR-based ({atr_multiplier:.1f}x ATR)"
            stop_loss_percent = (stop_distance / current_price) * 100
        else:
            stop_loss_percent = self.DEFAULT_STOP_LOSS_PERCENT
            if action == Signal.BUY:
                stop_loss_price = current_price * (1 - stop_loss_percent / 100)
            else:
                stop_loss_price = current_price * (1 + stop_loss_percent / 100)
            methodology = f"Fixed {stop_loss_percent}%"

        risk_per_share = abs(current_price - stop_loss_price)

        trailing_stop = None
        if self.enable_trailing_stop and action == Signal.BUY:
            trailing_stop = TrailingStopConfig(
                enabled=True,
                trail_percent=self.TRAILING_STOP_PERCENT,
                activation_percent=self.TRAILING_ACTIVATION_PERCENT,
            )
            methodology = f"{methodology} + Trailing {self.TRAILING_STOP_PERCENT}%"

        return StopLossCalculation(
            stop_loss_price=stop_loss_price,
            stop_loss_percent=stop_loss_percent,
            risk_per_share=risk_per_share,
            max_loss_amount=0.0,
            methodology=methodology,
            trailing_stop=trailing_stop,
        )

    def _calculate_position_size(
        self,
        current_price: float,
        stop_loss: StopLossCalculation,
        account_info: AccountInfo,
        target_portfolio_weight: float | None = None,
    ) -> PositionSizeCalculation:
        """Calculate position size based on risk parameters.

        Args:
            current_price: Current price
            stop_loss: Stop-loss calculation
            account_info: Account information
            target_portfolio_weight: Optional target portfolio weight for allocation-based sizing

        Returns:
            PositionSizeCalculation with sizing details
        """
        if current_price <= 0:
            msg = f"Invalid current_price: {current_price}. Must be positive."
            raise ValueError(msg)

        # If target weight provided, use weight-based sizing
        if target_portfolio_weight is not None and target_portfolio_weight > 0:
            return self._calculate_weight_based_position(
                current_price, stop_loss, account_info, target_portfolio_weight
            )

        # Otherwise use risk-based sizing (existing logic continues...)
        max_risk_amount = account_info.balance * (self.max_position_risk / 100)

        risk_per_share = stop_loss.risk_per_share
        min_risk_per_share = 1e-6
        if -min_risk_per_share < risk_per_share < min_risk_per_share:
            reasoning = (
                "Risk per share is zero or too small to calculate a reliable position size. "
                "Returning zero-sized position to avoid division by zero."
            )
            return PositionSizeCalculation(
                recommended_shares=0,
                position_value=0.0,
                risk_amount=0.0,
                risk_percent=0.0,
                reasoning=reasoning,
            )

        recommended_shares = int(max_risk_amount / risk_per_share)

        position_value = recommended_shares * current_price
        if position_value > account_info.available_cash:
            recommended_shares = int(account_info.available_cash / current_price)
            position_value = recommended_shares * current_price

        max_position_value = account_info.balance * (self.max_single_position / 100)
        if position_value > max_position_value:
            recommended_shares = int(max_position_value / current_price)
            position_value = recommended_shares * current_price

        risk_amount = recommended_shares * risk_per_share
        risk_percent = (risk_amount / account_info.balance) * 100 if account_info.balance > 0 else 0.0

        reasoning = (
            f"Risk {risk_percent:.2f}% (${risk_amount:.2f}) on {recommended_shares} shares. "
            f"Stop @ ${stop_loss.stop_loss_price:.2f} ({stop_loss.stop_loss_percent:.1f}% from entry)."
        )

        return PositionSizeCalculation(
            recommended_shares=recommended_shares,
            position_value=position_value,
            risk_amount=risk_amount,
            risk_percent=risk_percent,
            reasoning=reasoning,
        )

    def _calculate_weight_based_position(
        self,
        current_price: float,
        stop_loss: StopLossCalculation,
        account_info: AccountInfo,
        target_weight: float,
    ) -> PositionSizeCalculation:
        """Calculate position size based on target portfolio weight.

        Args:
            current_price: Current price
            stop_loss: Stop-loss calculation
            account_info: Account information
            target_weight: Target portfolio weight (0-1)

        Returns:
            PositionSizeCalculation with sizing details
        """
        # Calculate target position value based on portfolio weight
        target_position_value = account_info.balance * target_weight

        # Constrain by available cash
        target_position_value = min(target_position_value, account_info.available_cash)

        # Constrain by max single position limit
        max_position_value = account_info.balance * (self.max_single_position / 100)
        target_position_value = min(target_position_value, max_position_value)

        # Calculate shares
        recommended_shares = int(target_position_value / current_price)
        position_value = recommended_shares * current_price

        # Treat zero shares as constraint violation
        if recommended_shares <= 0:
            risk_percent = 100.0
            risk_amount = account_info.balance
            reasoning = (
                f"Portfolio-weighted position: {target_weight:.1%} target would result in 0 shares "
                f"@ ${current_price:.2f}. Insufficient capital for minimum position."
            )
        else:
            # Calculate risk based on stop loss
            risk_amount = recommended_shares * stop_loss.risk_per_share
            risk_percent = (risk_amount / account_info.balance) * 100 if account_info.balance > 0 else 0.0

            reasoning = (
                f"Portfolio-weighted position: {target_weight:.1%} target, {recommended_shares} shares "
                f"(${position_value:.2f}). Risk {risk_percent:.2f}% (${risk_amount:.2f}) "
                f"with stop @ ${stop_loss.stop_loss_price:.2f}."
            )

        return PositionSizeCalculation(
            recommended_shares=recommended_shares,
            position_value=position_value,
            risk_amount=risk_amount,
            risk_percent=risk_percent,
            reasoning=reasoning,
        )

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
        broker_positions: dict[str, BrokerPosition] | None = None,
        portfolio_value: float | None = None,
        backtest_validation: "BacktestValidation | None" = None,
        broker_api_failed: bool = False,
    ) -> RiskValidation:
        """Validate risk constraints and generate approval.

        Args:
            symbol: Stock ticker
            action: Trading action
            position_sizing: Position sizing calculation
            account_info: Account information
            decision_confidence: Decision confidence score
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
    ) -> bool:
        """Validate portfolio-level VaR limits.

        Args:
            symbol: Stock ticker
            action: Trading action
            position_value: Proposed position value
            broker_positions: Current broker positions
            portfolio_value: Current portfolio value
            warnings: Warning list to append to

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
            self._latest_portfolio_var = var_result

            if not var_result.sufficient_data:
                warnings.append("Insufficient data for portfolio VaR check, approving trade")
                return True

            self._portfolio_cdar = var_result.cdar_95

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
            logger.error(f"Portfolio VaR validation failed: {e}")
            warnings.append(f"Portfolio VaR check failed: {e}")
            return True

    def _get_adaptive_atr_multiplier(self) -> float:
        """Get ATR multiplier, adjusted for CDaR if adaptive stops enabled.

        Returns:
            ATR multiplier (default 2.0, reduced when CDaR is high)
        """
        if (
            not self._var_config.adaptive_stop_loss
            or self._portfolio_cdar is None
            or self._portfolio_cdar <= self._var_config.cdar_stop_threshold
        ):
            return self.ATR_MULTIPLIER

        # Linear interpolation: as CDaR goes from threshold to 2x threshold,
        # multiplier goes from ATR_MULTIPLIER down to atr_multiplier_min
        cdar_ratio = min(self._portfolio_cdar / self._var_config.cdar_stop_threshold, 2.0)
        t = cdar_ratio - 1.0  # 0.0 at threshold, 1.0 at 2x threshold
        multiplier = self.ATR_MULTIPLIER - t * (self.ATR_MULTIPLIER - self._var_config.atr_multiplier_min)

        logger.debug(
            f"Adaptive stop: CDaR={self._portfolio_cdar:.4f}, "
            f"threshold={self._var_config.cdar_stop_threshold:.4f}, multiplier={multiplier:.2f}"
        )
        return multiplier

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

    def _get_atr(self, market_data: pd.DataFrame, period: int = 14) -> float | None:
        """Calculate ATR from market data.

        Args:
            market_data: OHLCV dataframe
            period: ATR period

        Returns:
            ATR value or None if calculation fails
        """
        try:
            df = market_data.copy()
            df.ta.atr(length=period, append=True)
            atr_col = f"ATRr_{period}"
            if atr_col in df.columns:
                return float(df[atr_col].iloc[-1])
        except Exception as e:
            logger.warning(f"ATR calculation failed: {e}")
        return None

    def _calculate_risk_score(
        self,
        risk_percent: float,
        exposure_percent: float,
        confidence: float,
        backtest_validation: "BacktestValidation | None" = None,
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

    def _audit_log(self, assessment: RiskAssessment) -> None:
        """Log risk assessment to audit file.

        Args:
            assessment: Risk assessment to log
        """
        try:
            log_entry = {
                "timestamp": datetime.now(UTC).isoformat(),
                "symbol": assessment.symbol,
                "action": assessment.action.value,
                "current_price": assessment.current_price,
                "approved": assessment.validation.approved,
                "risk_level": assessment.validation.risk_level,
                "risk_score": assessment.validation.risk_score,
                "confidence": assessment.confidence,
                "recommended_shares": assessment.position_sizing.recommended_shares,
                "position_value": assessment.position_sizing.position_value,
                "risk_amount": assessment.position_sizing.risk_amount,
                "risk_percent": assessment.position_sizing.risk_percent,
                "stop_loss_price": assessment.stop_loss.stop_loss_price,
                "warnings": assessment.validation.warnings,
            }

            if assessment.portfolio_var:
                log_entry["portfolio_var_95"] = assessment.portfolio_var.var_95
                log_entry["portfolio_cvar_99"] = assessment.portfolio_var.cvar_99
                log_entry["portfolio_cdar_95"] = assessment.portfolio_var.cdar_95

            with self.audit_log_path.open("a") as f:
                f.write(json.dumps(log_entry) + "\n")

            logger.debug(f"Audit logged: {assessment.symbol} {assessment.action.value}")
        except Exception as e:
            logger.error(f"Audit logging failed: {e}")

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"RiskManagementAgent(max_risk={self.max_position_risk}%, "
            f"max_exposure={self.max_exposure}%, trailing={self.enable_trailing_stop})"
        )
