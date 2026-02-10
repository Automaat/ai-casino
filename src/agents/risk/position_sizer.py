"""Position sizing component."""

from src.agents.risk.models import AccountInfo, PositionSizeCalculation, StopLossCalculation


class PositionSizer:
    """Calculates position sizes based on risk parameters."""

    def __init__(
        self,
        max_position_risk: float,
        max_exposure: float,
        max_single_position: float,
    ) -> None:
        """Initialize position sizer with resolved config values.

        Args:
            max_position_risk: Max risk per trade (%)
            max_exposure: Max total exposure (%)
            max_single_position: Max single position size (%)
        """
        self.max_position_risk = max_position_risk
        self.max_exposure = max_exposure
        self.max_single_position = max_single_position

    def calculate(
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
            return self._calculate_weight_based(
                current_price, stop_loss, account_info, target_portfolio_weight
            )

        # Otherwise use risk-based sizing
        return self._calculate_risk_based(current_price, stop_loss, account_info)

    def _calculate_risk_based(
        self,
        current_price: float,
        stop_loss: StopLossCalculation,
        account_info: AccountInfo,
    ) -> PositionSizeCalculation:
        """Calculate position size based on risk per trade.

        Args:
            current_price: Current price
            stop_loss: Stop-loss calculation
            account_info: Account information

        Returns:
            PositionSizeCalculation with sizing details
        """
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

    def _calculate_weight_based(
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

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"PositionSizer(max_risk={self.max_position_risk}%, "
            f"max_exposure={self.max_exposure}%, max_single={self.max_single_position}%)"
        )
