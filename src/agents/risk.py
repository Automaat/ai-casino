"""Risk Management Agent."""

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from loguru import logger
from pydantic import BaseModel

from src.models.llm import LLMClient
from src.strategies.momentum import Signal


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
    ) -> None:
        """Initialize risk management agent.

        Args:
            llm_client: LLM client for risk interpretation
            max_position_risk: Override max risk per trade (%)
            max_exposure: Override max total exposure (%)
            max_single_position: Override max single position size (%)
            enable_trailing_stop: Enable trailing stop-loss
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

        self.audit_log_path = Path("logs/risk_audit.jsonl")
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Initialized RiskManagementAgent "
            f"(max_risk={self.max_position_risk}%, max_exposure={self.max_exposure}%, "
            f"max_single={self.max_single_position}%, trailing={enable_trailing_stop})"
        )

    def assess(
        self,
        symbol: str,
        action: Signal,
        current_price: float,
        account_info: AccountInfo,
        market_data: pd.DataFrame,
        decision_confidence: float,
    ) -> RiskAssessment:
        """Perform complete risk assessment.

        Args:
            symbol: Stock ticker
            action: Proposed trading action
            current_price: Current stock price
            account_info: Account balance and positions
            market_data: OHLCV data for volatility analysis
            decision_confidence: Trading decision confidence

        Returns:
            RiskAssessment with sizing, stop-loss, validation
        """
        logger.info(f"Assessing risk for {action.value} {symbol} @ ${current_price:.2f}")

        if action == Signal.HOLD:
            assessment = self._hold_assessment(symbol, current_price, account_info)
        else:
            stop_loss = self._calculate_stop_loss(current_price, market_data, action)

            position_sizing = self._calculate_position_size(current_price, stop_loss, account_info)

            stop_loss.max_loss_amount = position_sizing.risk_amount

            validation = self._validate_risk(
                symbol, action, position_sizing, account_info, decision_confidence
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

        if atr and atr > 0:
            stop_distance = atr * self.ATR_MULTIPLIER
            if action == Signal.BUY:
                stop_loss_price = current_price - stop_distance
            else:
                stop_loss_price = current_price + stop_distance
            methodology = f"ATR-based ({self.ATR_MULTIPLIER}x ATR)"
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
    ) -> PositionSizeCalculation:
        """Calculate position size based on risk parameters.

        Args:
            current_price: Current price
            stop_loss: Stop-loss calculation
            account_info: Account information

        Returns:
            PositionSizeCalculation with sizing details
        """
        if current_price <= 0:
            msg = f"Invalid current_price: {current_price}. Must be positive."
            raise ValueError(msg)

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
    ) -> RiskValidation:
        """Validate risk constraints and generate approval.

        Args:
            symbol: Stock ticker
            action: Trading action
            position_sizing: Position sizing calculation
            account_info: Account information
            decision_confidence: Decision confidence score

        Returns:
            RiskValidation with approval status
        """
        warnings = []
        constraints_met = {}

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

        approved = all(constraints_met.values())

        risk_score = self._calculate_risk_score(
            position_sizing.risk_percent,
            exposure_percent,
            decision_confidence,
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
    ) -> float:
        """Calculate overall risk score (0.0-1.0, higher = safer).

        Args:
            risk_percent: Position risk percentage
            exposure_percent: Total exposure percentage
            confidence: Decision confidence

        Returns:
            Risk score (0.0-1.0)
        """
        risk_component = 1.0 - (risk_percent / self.max_position_risk)
        exposure_component = 1.0 - (exposure_percent / self.max_exposure)

        risk_component = max(0.0, min(1.0, risk_component))
        exposure_component = max(0.0, min(1.0, exposure_component))

        score = risk_component * 0.3 + exposure_component * 0.3 + confidence * 0.4
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
