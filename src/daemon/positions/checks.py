"""Position check methods for management decisions."""

from __future__ import annotations

from datetime import UTC, datetime

from src.daemon.config import PositionManagementConfig
from src.daemon.positions.models import PositionManagementAction, PositionRecord
from src.workflows.types import TradingWorkflowResult


class PositionCheckRunner:
    """Run position management checks."""

    def __init__(self, config: PositionManagementConfig) -> None:
        """Initialize check runner.

        Args:
            config: Position management configuration
        """
        self.config = config

    def check_profit_targets(
        self,
        position: PositionRecord,
        current_price: float,
    ) -> list[PositionManagementAction]:
        """Check if profit targets are hit.

        Args:
            position: Position to check
            current_price: Current market price

        Returns:
            List of profit-taking actions
        """
        actions: list[PositionManagementAction] = []

        if not position.profit_targets:
            return actions

        targets_hit = [t for t in position.profit_targets if current_price >= t]
        if not targets_hit:
            return actions

        remaining_qty = position.current_qty

        for target in targets_hit:
            target_idx = position.profit_targets.index(target)
            sell_pct = (
                self.config.profit_target_1_sell_pct
                if target_idx == 0
                else self.config.profit_target_2_sell_pct
            )

            qty_to_sell = remaining_qty * sell_pct
            if qty_to_sell < 1:
                continue

            gain_pct = ((current_price - position.entry_price) / position.entry_price) * 100

            action = PositionManagementAction(
                symbol=position.symbol,
                action_type="PARTIAL_PROFIT",
                timestamp=datetime.now(UTC),
                qty_sold=qty_to_sell,
                price=current_price,
                reason=f"Hit profit target {target:.2f} (+{gain_pct:.1f}%)",
                executed=False,
            )
            actions.append(action)
            remaining_qty -= qty_to_sell

            position.profit_targets.remove(target)

        return actions

    def check_breakeven_activation(
        self,
        position: PositionRecord,
        current_price: float,
    ) -> PositionManagementAction | None:
        """Check if break-even stop should be activated.

        Args:
            position: Position to check
            current_price: Current market price

        Returns:
            Break-even action or None
        """
        if position.breakeven_activated:
            return None

        gain_pct = ((current_price - position.entry_price) / position.entry_price) * 100
        if gain_pct < self.config.breakeven_activation_percent:
            return None

        new_stop = position.entry_price
        if new_stop <= position.current_stop_loss:
            return None

        action = PositionManagementAction(
            symbol=position.symbol,
            action_type="BREAKEVEN",
            timestamp=datetime.now(UTC),
            old_stop_loss=position.current_stop_loss,
            new_stop_loss=new_stop,
            price=current_price,
            reason=f"Break-even activated at +{gain_pct:.1f}%",
            executed=False,
        )

        position.breakeven_activated = True
        return action

    def check_trailing_stop(
        self,
        position: PositionRecord,
        current_price: float,
        atr_value: float | None = None,
    ) -> PositionManagementAction | None:
        """Check if trailing stop should be updated.

        Args:
            position: Position to check
            current_price: Current market price
            atr_value: ATR(14) value for dynamic stops (None falls back to fixed %)

        Returns:
            Trailing stop action or None
        """
        if position.high_water_mark is None or current_price > position.high_water_mark:
            position.high_water_mark = current_price

        if self.config.use_atr_trailing_stop and atr_value is not None:
            new_stop = self._calculate_atr_trailing_stop(position, atr_value)
        else:
            new_stop = position.high_water_mark * (1 - self.config.trailing_stop_percent / 100)

        if new_stop <= position.current_stop_loss:
            return None

        reason = f"Trailing stop update (HWM=${position.high_water_mark:.2f})"
        if self.config.use_atr_trailing_stop and atr_value is not None:
            reason = f"ATR trailing stop (HWM=${position.high_water_mark:.2f}, ATR={atr_value:.2f})"

        action = PositionManagementAction(
            symbol=position.symbol,
            action_type="TRAILING_STOP",
            timestamp=datetime.now(UTC),
            old_stop_loss=position.current_stop_loss,
            new_stop_loss=new_stop,
            price=current_price,
            reason=reason,
            executed=False,
        )

        position.trailing_stop_activated = True
        return action

    def _calculate_atr_trailing_stop(
        self,
        position: PositionRecord,
        atr_value: float,
    ) -> float:
        """Calculate ATR-based trailing stop with ratcheting by R-tier.

        Args:
            position: Position to check
            atr_value: ATR(14) value

        Returns:
            New stop-loss price
        """
        hwm = position.high_water_mark or position.entry_price
        profit_per_share = hwm - position.entry_price
        risk_per_share = position.entry_price - position.initial_stop_loss

        r_multiple = profit_per_share / risk_per_share if risk_per_share > 0 else 0.0

        if r_multiple >= 2.0:
            multiplier = self.config.atr_ratchet_2r
        elif r_multiple >= 1.0:
            multiplier = self.config.atr_ratchet_1r
        else:
            multiplier = self.config.atr_trailing_multiplier

        return hwm - (atr_value * multiplier)

    def check_time_exit(
        self,
        position: PositionRecord,
        current_price: float,
    ) -> PositionManagementAction | None:
        """Check if time-based exit should trigger.

        Args:
            position: Position to check
            current_price: Current market price

        Returns:
            Time exit action or None
        """
        days_held = (datetime.now(UTC) - position.entry_timestamp).days
        position.days_held = days_held

        if days_held < self.config.max_holding_days:
            return None

        return PositionManagementAction(
            symbol=position.symbol,
            action_type="TIME_EXIT",
            timestamp=datetime.now(UTC),
            qty_sold=position.current_qty,
            price=current_price,
            reason=f"Held for {days_held} days (max={self.config.max_holding_days})",
            executed=False,
        )

    def check_conviction_scaling(
        self,
        position: PositionRecord,
        latest_analysis: TradingWorkflowResult,
    ) -> PositionManagementAction | None:
        """Check if conviction-based scaling should trigger.

        Args:
            position: Position to check
            latest_analysis: Latest trading analysis

        Returns:
            Conviction scaling action or None
        """
        if latest_analysis.decision.action.value != "REDUCE":
            return None

        confidence_drop = position.entry_confidence - latest_analysis.decision.confidence
        if confidence_drop < self.config.conviction_decrease_threshold:
            return None

        qty_to_sell = position.current_qty * self.config.conviction_scale_out_percent
        if qty_to_sell < 1:
            return None

        entry_conf = position.entry_confidence
        current_conf = latest_analysis.decision.confidence
        reason = f"Conviction dropped {confidence_drop:.2f} (entry={entry_conf:.2f}, now={current_conf:.2f})"

        return PositionManagementAction(
            symbol=position.symbol,
            action_type="CONVICTION_SCALE",
            timestamp=datetime.now(UTC),
            qty_sold=qty_to_sell,
            price=latest_analysis.risk.current_price,
            reason=reason,
            executed=False,
        )

    def check_r_multiple_targets(
        self,
        position: PositionRecord,
        current_price: float,
    ) -> list[PositionManagementAction]:
        """Check R-multiple profit targets and scale out accordingly.

        Args:
            position: Position to check
            current_price: Current market price

        Returns:
            List of actions (partial sells, stop adjustments)
        """
        actions: list[PositionManagementAction] = []
        if position.initial_risk_per_share is None or position.initial_risk_per_share <= 0:
            return actions

        profit = current_price - position.entry_price
        r_multiple = profit / position.initial_risk_per_share

        # High conviction delays targets by N R-levels
        delay = 0
        if position.entry_confidence >= self.config.high_conviction_threshold:
            delay = self.config.r_high_conviction_delay

        r_targets = [1 + delay, 2 + delay]

        for r_level in r_targets:
            if r_level in position.r_multiple_targets_hit:
                continue
            if r_multiple < r_level:
                continue

            sell_pct = (
                self.config.r_target_1_sell_pct
                if r_level == r_targets[0]
                else self.config.r_target_2_sell_pct
            )
            qty_to_sell = position.current_qty * sell_pct
            if qty_to_sell < 1:
                continue

            actions.append(
                PositionManagementAction(
                    symbol=position.symbol,
                    action_type="PARTIAL_PROFIT",
                    timestamp=datetime.now(UTC),
                    qty_sold=qty_to_sell,
                    price=current_price,
                    reason=f"R-multiple target {r_level}R hit (R={r_multiple:.1f})",
                    executed=False,
                )
            )
            position.r_multiple_targets_hit.append(r_level)

            # At 1R: move stop to breakeven
            if r_level == r_targets[0] and not position.breakeven_activated:
                new_stop = position.entry_price
                if new_stop > position.current_stop_loss:
                    actions.append(
                        PositionManagementAction(
                            symbol=position.symbol,
                            action_type="BREAKEVEN",
                            timestamp=datetime.now(UTC),
                            old_stop_loss=position.current_stop_loss,
                            new_stop_loss=new_stop,
                            price=current_price,
                            reason=f"Breakeven at {r_level}R",
                            executed=False,
                        )
                    )
                    position.breakeven_activated = True

            # At 2R: trail at 1R level
            if r_level == r_targets[1]:
                trail_stop = position.entry_price + position.initial_risk_per_share
                if trail_stop > position.current_stop_loss:
                    actions.append(
                        PositionManagementAction(
                            symbol=position.symbol,
                            action_type="TRAILING_STOP",
                            timestamp=datetime.now(UTC),
                            old_stop_loss=position.current_stop_loss,
                            new_stop_loss=trail_stop,
                            price=current_price,
                            reason=f"Trail at 1R after {r_level}R hit",
                            executed=False,
                        )
                    )

        return actions

    def check_circuit_breaker(
        self,
        position: PositionRecord,
        current_price: float,
    ) -> PositionManagementAction | None:
        """Check if position drawdown exceeds hard circuit breaker limit.

        Args:
            position: Position to check
            current_price: Current market price

        Returns:
            Sell-all action if circuit breaker triggered, None otherwise
        """
        cb = self.config.circuit_breaker
        if not cb.enabled:
            return None

        drawdown_pct = ((current_price - position.entry_price) / position.entry_price) * 100
        if drawdown_pct >= -cb.position_max_drawdown_pct:
            return None

        return PositionManagementAction(
            symbol=position.symbol,
            action_type="CIRCUIT_BREAKER",
            timestamp=datetime.now(UTC),
            qty_sold=position.current_qty,
            price=current_price,
            reason=(
                f"Circuit breaker: drawdown {drawdown_pct:.1f}% "
                f"exceeds limit -{cb.position_max_drawdown_pct:.1f}%"
            ),
            executed=False,
        )

    def check_conviction_decay(
        self,
        position: PositionRecord,
        current_price: float,
    ) -> PositionManagementAction | None:
        """Check if conviction has decayed below exit threshold.

        Args:
            position: Position to check
            current_price: Current market price

        Returns:
            Sell action if conviction decayed below threshold, None otherwise
        """
        if position.current_conviction is None or position.last_analysis_timestamp is None:
            return None

        days_since = (datetime.now(UTC) - position.last_analysis_timestamp).total_seconds() / 86400
        if days_since < 0.1:
            return None

        decayed = position.current_conviction * (self.config.conviction_decay_rate**days_since)

        if decayed >= self.config.conviction_exit_threshold:
            return None

        return PositionManagementAction(
            symbol=position.symbol,
            action_type="CONVICTION_DECAY",
            timestamp=datetime.now(UTC),
            qty_sold=position.current_qty,
            price=current_price,
            reason=(
                f"Conviction decayed to {decayed:.2f} "
                f"(threshold={self.config.conviction_exit_threshold}, "
                f"days_since_analysis={days_since:.1f})"
            ),
            executed=False,
        )

    def should_suppress_exit(self, adx_value: float | None) -> bool:
        """Check if soft exits should be suppressed due to low trend strength.

        Args:
            adx_value: ADX indicator value (None if unavailable)

        Returns:
            True if exits should be suppressed (low ADX = choppy market)
        """
        if not self.config.whipsaw_prevention_enabled:
            return False
        if adx_value is None:
            return False
        return adx_value < self.config.adx_exit_filter_threshold

    def check_sell_confirmation(self, position: PositionRecord, has_sell_signal: bool) -> bool:
        """Track consecutive sell signals and confirm when threshold met.

        Args:
            position: Position to track
            has_sell_signal: Whether current cycle produced a sell signal

        Returns:
            True if sell is confirmed (count >= N), False to suppress
        """
        if not self.config.whipsaw_prevention_enabled:
            return True
        if has_sell_signal:
            position.pending_sell_signal_count += 1
        else:
            position.pending_sell_signal_count = 0
        return position.pending_sell_signal_count >= self.config.sell_confirmation_cycles

    def __repr__(self) -> str:
        """Return string representation."""
        return f"PositionCheckRunner(config={self.config})"
