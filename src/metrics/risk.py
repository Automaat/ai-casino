"""Institutional-grade risk metrics using Riskfolio-Lib."""

import math

import numpy as np
import riskfolio as rf
from loguru import logger
from pydantic import BaseModel, Field

EPSILON = 1e-10
MIN_RETURNS_FOR_CALCULATION = 2


class VaRMetrics(BaseModel):
    """Value at Risk at multiple confidence levels."""

    var_95: float = Field(description="VaR at 95% confidence")
    var_99: float = Field(description="VaR at 99% confidence")
    cvar_95: float = Field(description="CVaR (conditional VaR) at 95%")
    cvar_99: float = Field(description="CVaR at 99% confidence")


class DrawdownMetrics(BaseModel):
    """Drawdown-based risk metrics."""

    max_drawdown: float = Field(description="Maximum peak-to-trough drawdown")
    cdar_95: float = Field(description="Conditional Drawdown at Risk (95%)")
    avg_drawdown: float = Field(description="Average of all drawdowns")
    max_drawdown_duration_days: int = Field(description="Duration of max drawdown in trading days")


class RiskMetrics(BaseModel):
    """Comprehensive institutional risk metrics."""

    var_metrics: VaRMetrics
    drawdown_metrics: DrawdownMetrics
    volatility_annual: float = Field(description="Annualized standard deviation")
    downside_deviation: float = Field(description="Annualized downside deviation")


def calculate_volatility(returns: list[float]) -> float:
    """Calculate annualized volatility.

    Args:
        returns: List of returns (as decimals, e.g., 0.05 for 5%)

    Returns:
        Annualized volatility (standard deviation)
    """
    if not returns:
        logger.warning("Empty returns, returning zero volatility")
        return 0.0

    if len(returns) < MIN_RETURNS_FOR_CALCULATION:
        logger.warning("Insufficient returns for volatility (need ≥2), returning zero")
        return 0.0

    mean_return = sum(returns) / len(returns)
    variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
    std_dev = math.sqrt(variance)

    annualized_volatility = std_dev * math.sqrt(252)

    logger.debug(f"Volatility: {annualized_volatility:.4f} (std={std_dev:.4f}, trades={len(returns)})")
    return annualized_volatility


def calculate_downside_deviation(returns: list[float]) -> float:
    """Calculate annualized downside deviation.

    Args:
        returns: List of returns (as decimals)

    Returns:
        Annualized downside deviation
    """
    if not returns:
        logger.warning("Empty returns, returning zero downside deviation")
        return 0.0

    if len(returns) < MIN_RETURNS_FOR_CALCULATION:
        logger.warning("Insufficient returns for downside deviation (need ≥2), returning zero")
        return 0.0

    negative_returns = [r for r in returns if r < 0]

    if not negative_returns:
        logger.debug("No negative returns, using all returns for downside deviation")
        negative_returns = returns

    downside_variance = sum(r**2 for r in negative_returns) / len(negative_returns)
    downside_std = math.sqrt(downside_variance)

    annualized_downside = downside_std * math.sqrt(252)

    logger.debug(
        f"Downside deviation: {annualized_downside:.4f} "
        f"(downside_std={downside_std:.4f}, neg_returns={len(negative_returns)})"
    )
    return annualized_downside


def _calculate_drawdown_duration(returns: list[float]) -> int:
    """Calculate maximum drawdown duration in trading days.

    Args:
        returns: List of returns (as decimals)

    Returns:
        Maximum drawdown duration in days
    """
    if not returns:
        return 0

    cumulative_returns = [1.0]
    for r in returns:
        cumulative_returns.append(cumulative_returns[-1] * (1 + r))

    peak = cumulative_returns[0]
    max_duration = 0
    current_duration = 0
    peak_idx = 0
    max_dd = 0.0

    for i, value in enumerate(cumulative_returns):
        if value >= peak:
            peak = value
            peak_idx = i
            current_duration = 0
        else:
            current_duration = i - peak_idx
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
                max_duration = current_duration

    if cumulative_returns[-1] < peak and (len(cumulative_returns) - 1 - peak_idx) > max_duration:
        max_duration = len(cumulative_returns) - 1 - peak_idx

    return max_duration


def _calculate_var_internal(returns: list[float]) -> VaRMetrics:
    """Calculate VaR metrics using Riskfolio-Lib.

    Args:
        returns: List of returns (as decimals)

    Returns:
        VaRMetrics with VaR and CVaR at 95% and 99% confidence
    """
    if not returns:
        logger.warning("Empty returns, returning zero VaR metrics")
        return VaRMetrics(var_95=0.0, var_99=0.0, cvar_95=0.0, cvar_99=0.0)

    if len(returns) < MIN_RETURNS_FOR_CALCULATION:
        logger.warning("Insufficient returns for VaR (need ≥2), returning zeros")
        return VaRMetrics(var_95=0.0, var_99=0.0, cvar_95=0.0, cvar_99=0.0)

    try:
        returns_array = np.array(returns)

        var_95 = rf.VaR_Hist(returns_array, alpha=0.05)
        var_99 = rf.VaR_Hist(returns_array, alpha=0.01)
        cvar_95 = rf.CVaR_Hist(returns_array, alpha=0.05)
        cvar_99 = rf.CVaR_Hist(returns_array, alpha=0.01)

        logger.debug(
            f"VaR metrics: VaR95={var_95:.4f}, VaR99={var_99:.4f}, CVaR95={cvar_95:.4f}, CVaR99={cvar_99:.4f}"
        )
        return VaRMetrics(
            var_95=float(var_95), var_99=float(var_99), cvar_95=float(cvar_95), cvar_99=float(cvar_99)
        )
    except Exception as e:
        logger.error(f"VaR calculation failed: {e}")
        raise


def _calculate_drawdown_internal(returns: list[float]) -> DrawdownMetrics:
    """Calculate drawdown metrics using Riskfolio-Lib.

    Args:
        returns: List of returns (as decimals)

    Returns:
        DrawdownMetrics with max DD, CDaR, average DD, and duration
    """
    if not returns:
        logger.warning("Empty returns, returning zero drawdown metrics")
        return DrawdownMetrics(max_drawdown=0.0, cdar_95=0.0, avg_drawdown=0.0, max_drawdown_duration_days=0)

    if len(returns) < MIN_RETURNS_FOR_CALCULATION:
        logger.warning("Insufficient returns for drawdown (need ≥2), returning zeros")
        return DrawdownMetrics(max_drawdown=0.0, cdar_95=0.0, avg_drawdown=0.0, max_drawdown_duration_days=0)

    try:
        returns_array = np.array(returns)

        max_dd = rf.MDD_Abs(returns_array)
        cdar_95 = rf.CDaR_Abs(returns_array, alpha=0.05)
        avg_dd = rf.ADD_Abs(returns_array)
        duration = _calculate_drawdown_duration(returns)

        logger.debug(
            f"Drawdown metrics: max_dd={max_dd:.4f}, cdar_95={cdar_95:.4f}, "
            f"avg_dd={avg_dd:.4f}, duration={duration} days"
        )
        return DrawdownMetrics(
            max_drawdown=float(max_dd),
            cdar_95=float(cdar_95),
            avg_drawdown=float(avg_dd),
            max_drawdown_duration_days=duration,
        )
    except Exception as e:
        logger.error(f"Drawdown calculation failed: {e}")
        raise


class RiskMetricsCalculator:
    """Calculator for institutional-grade risk metrics."""

    def __init__(self, risk_free_rate: float = 0.02) -> None:
        """Initialize risk metrics calculator.

        Args:
            risk_free_rate: Annual risk-free rate (default 2%)
        """
        self.risk_free_rate = risk_free_rate
        logger.info(f"Initialized RiskMetricsCalculator (rf_rate={risk_free_rate})")

    def calculate_var(self, returns: list[float]) -> VaRMetrics:
        """Calculate VaR metrics at 95% and 99% confidence.

        Args:
            returns: List of returns (as decimals, e.g., 0.05 for 5%)

        Returns:
            VaRMetrics with VaR and CVaR at both confidence levels
        """
        return _calculate_var_internal(returns)

    def calculate_cvar(self, returns: list[float], confidence: float = 0.95) -> float:
        """Calculate CVaR at specified confidence.

        Args:
            returns: List of returns (as decimals)
            confidence: Confidence level (0.0-1.0, default 0.95)

        Returns:
            CVaR value (expected loss beyond VaR threshold)
        """
        if not 0 < confidence < 1:
            msg = f"Confidence must be between 0 and 1, got {confidence}"
            raise ValueError(msg)

        if not returns:
            logger.warning("Empty returns, returning zero CVaR")
            return 0.0

        if len(returns) < MIN_RETURNS_FOR_CALCULATION:
            logger.warning("Insufficient returns for CVaR (need ≥2), returning zero")
            return 0.0

        try:
            returns_array = np.array(returns)
            alpha = 1 - confidence
            cvar = rf.CVaR_Hist(returns_array, alpha=alpha)
            logger.debug(f"CVaR at {confidence * 100:.0f}% confidence: {cvar:.4f}")
            return float(cvar)
        except Exception as e:
            logger.error(f"CVaR calculation failed: {e}")
            raise

    def calculate_cdar(self, returns: list[float]) -> DrawdownMetrics:
        """Calculate conditional drawdown at risk and related metrics.

        Args:
            returns: List of returns (as decimals)

        Returns:
            DrawdownMetrics with CDaR, max DD, average DD, and duration
        """
        return _calculate_drawdown_internal(returns)

    def calculate_all(self, returns: list[float]) -> RiskMetrics:
        """Calculate comprehensive risk metrics.

        Args:
            returns: List of returns (as decimals)

        Returns:
            RiskMetrics with VaR, drawdown, volatility, and downside deviation
        """
        if not returns:
            logger.warning("Empty returns, returning zero risk metrics")
            return RiskMetrics(
                var_metrics=VaRMetrics(var_95=0.0, var_99=0.0, cvar_95=0.0, cvar_99=0.0),
                drawdown_metrics=DrawdownMetrics(
                    max_drawdown=0.0, cdar_95=0.0, avg_drawdown=0.0, max_drawdown_duration_days=0
                ),
                volatility_annual=0.0,
                downside_deviation=0.0,
            )

        var_metrics = _calculate_var_internal(returns)
        drawdown_metrics = _calculate_drawdown_internal(returns)
        volatility = calculate_volatility(returns)
        downside_dev = calculate_downside_deviation(returns)

        logger.info(
            f"Calculated comprehensive risk metrics: "
            f"VaR95={var_metrics.var_95:.4f}, volatility={volatility:.4f}, "
            f"max_dd={drawdown_metrics.max_drawdown:.4f}"
        )

        return RiskMetrics(
            var_metrics=var_metrics,
            drawdown_metrics=drawdown_metrics,
            volatility_annual=volatility,
            downside_deviation=downside_dev,
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return f"RiskMetricsCalculator(risk_free_rate={self.risk_free_rate})"
