"""Take-profit calculation component."""

from loguru import logger

from src.agents.risk.models import StopLossCalculation, TakeProfitCalculation
from src.strategies.signal import Signal


class TakeProfitCalculator:
    """Calculates take-profit prices using R:R ratio or fixed percentages."""

    def __init__(
        self,
        min_reward_risk_ratio: float = 2.0,
        default_take_profit_percent: float = 4.0,
    ) -> None:
        """Initialize take-profit calculator.

        Args:
            min_reward_risk_ratio: Minimum reward:risk ratio for target
            default_take_profit_percent: Fallback fixed take-profit %
        """
        self.min_reward_risk_ratio = min_reward_risk_ratio
        self.default_take_profit_percent = default_take_profit_percent

    def calculate(
        self,
        current_price: float,
        stop_loss: StopLossCalculation,
        action: Signal,
    ) -> TakeProfitCalculation:
        """Calculate take-profit price from stop-loss distance and R:R ratio.

        Args:
            current_price: Current stock price
            stop_loss: Stop-loss calculation with risk_per_share
            action: Trading action (BUY/SELL)

        Returns:
            TakeProfitCalculation with target price and R:R ratio
        """
        risk_per_share = stop_loss.risk_per_share

        if risk_per_share > 0:
            profit_target = risk_per_share * self.min_reward_risk_ratio

            if action == Signal.BUY:
                take_profit_price = current_price + profit_target
            else:
                take_profit_price = current_price - profit_target

            take_profit_price = round(take_profit_price, 2)
            potential_profit = abs(take_profit_price - current_price)
            reward_risk_ratio = round(potential_profit / risk_per_share, 2)
            take_profit_percent = round((potential_profit / current_price) * 100, 2)
            methodology = f"R:R-based ({self.min_reward_risk_ratio:.1f}:1)"
        else:
            take_profit_percent = self.default_take_profit_percent
            if action == Signal.BUY:
                take_profit_price = round(current_price * (1 + take_profit_percent / 100), 2)
            else:
                take_profit_price = round(current_price * (1 - take_profit_percent / 100), 2)

            potential_profit = abs(take_profit_price - current_price)
            reward_risk_ratio = 0.0
            methodology = f"Fixed {take_profit_percent}%"

        logger.debug(
            f"Take-profit: ${take_profit_price:.2f} (R:R={reward_risk_ratio:.1f}, method={methodology})"
        )

        return TakeProfitCalculation(
            take_profit_price=take_profit_price,
            take_profit_percent=take_profit_percent,
            potential_profit_per_share=round(potential_profit, 2),
            reward_risk_ratio=reward_risk_ratio,
            methodology=methodology,
        )

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"TakeProfitCalculator(min_rr={self.min_reward_risk_ratio}, "
            f"default_pct={self.default_take_profit_percent})"
        )
