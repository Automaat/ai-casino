"""Stop-loss calculation component."""

import pandas as pd
from loguru import logger

from src.agents.risk.context import RiskContext
from src.agents.risk.models import PortfolioVaRConfig, StopLossCalculation, TrailingStopConfig
from src.strategies.signal import Signal


class StopLossCalculator:
    """Calculates stop-loss prices using ATR or fixed percentages."""

    def __init__(
        self,
        enable_trailing_stop: bool,
        var_config: PortfolioVaRConfig,
        atr_multiplier: float = 2.0,
        default_stop_percent: float = 2.0,
        trailing_stop_percent: float = 3.0,
        trailing_activation_percent: float = 5.0,
    ) -> None:
        """Initialize stop-loss calculator with resolved config values.

        Args:
            enable_trailing_stop: Enable trailing stop-loss
            var_config: VaR configuration for adaptive stops
            atr_multiplier: Base ATR multiplier
            default_stop_percent: Default stop-loss percentage
            trailing_stop_percent: Trailing stop percentage
            trailing_activation_percent: Activation threshold for trailing stop
        """
        self.enable_trailing_stop = enable_trailing_stop
        self.var_config = var_config
        self.atr_multiplier = atr_multiplier
        self.default_stop_percent = default_stop_percent
        self.trailing_stop_percent = trailing_stop_percent
        self.trailing_activation_percent = trailing_activation_percent

    def calculate(
        self,
        current_price: float,
        market_data: pd.DataFrame,
        action: Signal,
        context: RiskContext | None = None,
    ) -> StopLossCalculation:
        """Calculate stop-loss price (adaptive if context provided).

        Args:
            current_price: Current price
            market_data: OHLCV data
            action: Trading action (BUY/SELL)
            context: Optional risk context for adaptive stops

        Returns:
            StopLossCalculation with stop price and methodology
        """
        atr = self._get_atr(market_data)
        atr_multiplier = self._get_adaptive_multiplier(context)

        if atr and atr > 0:
            stop_distance = atr * atr_multiplier
            if action == Signal.BUY:
                stop_loss_price = current_price - stop_distance
            else:
                stop_loss_price = current_price + stop_distance
            methodology = f"ATR-based ({atr_multiplier:.1f}x ATR)"
            stop_loss_percent = (stop_distance / current_price) * 100
        else:
            stop_loss_percent = self.default_stop_percent
            if action == Signal.BUY:
                stop_loss_price = current_price * (1 - stop_loss_percent / 100)
            else:
                stop_loss_price = current_price * (1 + stop_loss_percent / 100)
            methodology = f"Fixed {stop_loss_percent}%"

        # Round stop loss to 2 decimals for broker API compliance
        stop_loss_price = round(stop_loss_price, 2)
        risk_per_share = abs(current_price - stop_loss_price)

        trailing_stop = None
        if self.enable_trailing_stop and action == Signal.BUY:
            trailing_stop = TrailingStopConfig(
                enabled=True,
                trail_percent=self.trailing_stop_percent,
                activation_percent=self.trailing_activation_percent,
            )
            methodology = f"{methodology} + Trailing {self.trailing_stop_percent}%"

        return StopLossCalculation(
            stop_loss_price=stop_loss_price,
            stop_loss_percent=stop_loss_percent,
            risk_per_share=risk_per_share,
            max_loss_amount=0.0,
            methodology=methodology,
            trailing_stop=trailing_stop,
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

    def _get_adaptive_multiplier(self, context: RiskContext | None) -> float:
        """Get ATR multiplier, adjusted for CDaR if adaptive stops enabled.

        Args:
            context: Optional risk context with portfolio CDaR

        Returns:
            ATR multiplier (default or reduced when CDaR is high)
        """
        if not context or not self.var_config.adaptive_stop_loss:
            return self.atr_multiplier

        portfolio_cdar = context.portfolio_cdar
        if portfolio_cdar is None or portfolio_cdar <= self.var_config.cdar_stop_threshold:
            return self.atr_multiplier

        # Linear interpolation: as CDaR goes from threshold to 2x threshold,
        # multiplier goes from atr_multiplier down to atr_multiplier_min
        cdar_ratio = min(portfolio_cdar / self.var_config.cdar_stop_threshold, 2.0)
        t = cdar_ratio - 1.0  # 0.0 at threshold, 1.0 at 2x threshold
        multiplier = self.atr_multiplier - t * (self.atr_multiplier - self.var_config.atr_multiplier_min)

        logger.debug(
            f"Adaptive stop: CDaR={portfolio_cdar:.4f}, "
            f"threshold={self.var_config.cdar_stop_threshold:.4f}, multiplier={multiplier:.2f}"
        )
        return multiplier

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"StopLossCalculator(trailing={self.enable_trailing_stop}, "
            f"atr_multiplier={self.atr_multiplier}, adaptive={self.var_config.adaptive_stop_loss})"
        )
