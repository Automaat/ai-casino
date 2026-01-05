"""Performance metrics calculation functions."""

import math

from loguru import logger

EPSILON = 1e-10


def calculate_returns_from_trades(trades: list) -> list[float]:
    """Calculate returns from closed trades.

    Args:
        trades: List of TradeRecord objects (must be closed with pnl_percent)

    Returns:
        List of return percentages (as decimals, e.g., 0.05 for 5%)
    """
    returns = []
    for trade in trades:
        if trade.is_closed() and trade.pnl_percent is not None:
            returns.append(trade.pnl_percent / 100.0)

    logger.debug(f"Calculated {len(returns)} returns from {len(trades)} trades")
    return returns


def calculate_win_rate(trades: list) -> float:
    """Calculate win rate from closed trades.

    Args:
        trades: List of TradeRecord objects

    Returns:
        Win rate as percentage (0.0-100.0)
    """
    closed_trades = [t for t in trades if t.is_closed()]

    if not closed_trades:
        logger.debug("No closed trades, win rate = 0.0")
        return 0.0

    winning_trades = [t for t in closed_trades if t.pnl and t.pnl > 0]
    win_rate = (len(winning_trades) / len(closed_trades)) * 100

    logger.debug(f"Win rate: {win_rate:.2f}% ({len(winning_trades)}/{len(closed_trades)})")
    return win_rate


def calculate_sharpe_ratio(returns: list[float], risk_free_rate: float = 0.02) -> float:
    """Calculate annualized Sharpe ratio.

    Args:
        returns: List of returns (as decimals, e.g., 0.05 for 5%)
        risk_free_rate: Annual risk-free rate (default 2%)

    Returns:
        Annualized Sharpe ratio
    """
    if not returns:
        logger.debug("No returns, Sharpe ratio = 0.0")
        return 0.0

    mean_return = sum(returns) / len(returns)
    variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
    std_dev = math.sqrt(variance)

    if std_dev < EPSILON:
        logger.debug("Zero standard deviation, Sharpe ratio = 0.0")
        return 0.0

    daily_risk_free = risk_free_rate / 252

    sharpe = (mean_return - daily_risk_free) / std_dev

    annualized_sharpe = sharpe * math.sqrt(252)

    logger.debug(
        f"Sharpe ratio: {annualized_sharpe:.4f} "
        f"(mean={mean_return:.4f}, std={std_dev:.4f}, trades={len(returns)})"
    )
    return annualized_sharpe


def calculate_max_drawdown(trades: list) -> tuple[float, float]:
    """Calculate maximum drawdown from trade history.

    Args:
        trades: List of TradeRecord objects (must be closed with pnl)

    Returns:
        Tuple of (max_drawdown_dollars, max_drawdown_percent)
    """
    if not trades:
        logger.debug("No trades, max drawdown = (0.0, 0.0)")
        return 0.0, 0.0

    closed_trades = [t for t in trades if t.is_closed() and t.pnl is not None]

    if not closed_trades:
        logger.debug("No closed trades with PnL, max drawdown = (0.0, 0.0)")
        return 0.0, 0.0

    sorted_trades = sorted(closed_trades, key=lambda t: t.timestamp)

    equity_curve = []
    cumulative_pnl = 0.0
    initial_capital = 100000.0

    equity_curve.append(initial_capital)

    for trade in sorted_trades:
        cumulative_pnl += trade.pnl
        equity = initial_capital + cumulative_pnl
        equity_curve.append(equity)

    peak = equity_curve[0]
    max_drawdown_dollars = 0.0
    max_drawdown_percent = 0.0

    for equity in equity_curve:
        peak = max(peak, equity)

        drawdown = peak - equity
        drawdown_percent = (drawdown / peak) * 100 if peak > 0 else 0.0

        if drawdown > max_drawdown_dollars:
            max_drawdown_dollars = drawdown
            max_drawdown_percent = drawdown_percent

    logger.debug(
        f"Max drawdown: ${max_drawdown_dollars:.2f} ({max_drawdown_percent:.2f}%) "
        f"from {len(closed_trades)} trades"
    )
    return max_drawdown_dollars, max_drawdown_percent


def calculate_risk_adjusted_returns(returns: list[float], risk_values: list[float]) -> float:
    """Calculate risk-adjusted returns (Sortino-like metric).

    Args:
        returns: List of returns (as decimals)
        risk_values: List of risk values (e.g., stop-loss prices)

    Returns:
        Risk-adjusted return ratio
    """
    if not returns or not risk_values:
        logger.debug("No returns or risk values, risk-adjusted return = 0.0")
        return 0.0

    negative_returns = [r for r in returns if r < 0]

    if not negative_returns:
        logger.debug("No negative returns, using all returns for downside deviation")
        negative_returns = returns

    mean_return = sum(returns) / len(returns)

    downside_variance = sum(r**2 for r in negative_returns) / len(negative_returns)
    downside_deviation = math.sqrt(downside_variance)

    if downside_deviation == 0:
        logger.debug("Zero downside deviation, risk-adjusted return = 0.0")
        return 0.0

    risk_adjusted = mean_return / downside_deviation

    annualized = risk_adjusted * math.sqrt(252)

    logger.debug(
        f"Risk-adjusted return: {annualized:.4f} "
        f"(mean={mean_return:.4f}, downside_dev={downside_deviation:.4f})"
    )
    return annualized
