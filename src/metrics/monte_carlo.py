"""Monte Carlo portfolio stress testing using vectorized numpy simulations."""

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd
from loguru import logger
from pydantic import BaseModel

MIN_HISTORICAL_DAYS = 30
MAX_MEMORY_MB = 1024


class SimulationMethod(StrEnum):
    """Monte Carlo simulation methods."""

    PARAMETRIC = "PARAMETRIC"  # Normal distribution (default)
    BOOTSTRAP = "BOOTSTRAP"  # Historical resampling
    GBM = "GBM"  # Geometric Brownian Motion


@dataclass
class SimulationConfig:
    """Configuration for Monte Carlo simulation."""

    num_simulations: int = 1000
    horizon_days: int = 252
    method: SimulationMethod = SimulationMethod.PARAMETRIC
    random_seed: int | None = None
    loss_threshold: float = 0.10


class MonteCarloResult(BaseModel):
    """Monte Carlo stress test results."""

    simulation_method: SimulationMethod
    num_simulations: int
    horizon_days: int

    # Key metrics
    prob_loss_gt_threshold: float  # Probability portfolio loss exceeds threshold
    expected_worst_drawdown: float  # Mean of worst 5% max drawdowns
    var_95: float  # 5th percentile portfolio return (≈ 95% one-sided VaR)
    cvar_95: float  # Expected return in worst 5% scenarios (CVaR, loss beyond VaR)
    median_recovery_days: float | None  # Median days to recover from drawdown

    # Distribution stats
    mean_return: float
    std_return: float
    min_return: float
    max_return: float

    # Raw results (for debugging/visualization)
    simulated_returns: list[float]  # Final returns across all sims


class MonteCarloSimulator:
    """Portfolio Monte Carlo stress testing with vectorized numpy."""

    def __init__(self, historical_returns: pd.DataFrame) -> None:
        """Initialize simulator with historical returns.

        Args:
            historical_returns: DataFrame with columns=symbols, index=dates, values=daily returns
        """
        self.returns = historical_returns
        self.mean_returns = historical_returns.mean()
        self.cov_matrix = historical_returns.cov()

        # Validate covariance matrix (add ridge if singular)
        try:
            np.linalg.cholesky(self.cov_matrix)
        except np.linalg.LinAlgError:
            logger.warning("Singular covariance matrix, adding ridge regularization")
            self.cov_matrix += 1e-6 * np.eye(len(self.cov_matrix))

    def __repr__(self) -> str:
        """Return string representation."""
        return f"MonteCarloSimulator(symbols={len(self.returns.columns)}, days={len(self.returns)})"

    def simulate(
        self,
        positions: dict[str, float],
        config: SimulationConfig | None = None,
    ) -> MonteCarloResult:
        """Run Monte Carlo simulation on portfolio.

        Args:
            positions: {symbol: market_value}
            config: Simulation configuration (uses defaults if not provided)

        Returns:
            MonteCarloResult with tail risk metrics

        Raises:
            ValueError: If positions empty or insufficient data
        """
        cfg = config or SimulationConfig()
        if not positions:
            msg = "Portfolio positions cannot be empty"
            raise ValueError(msg)

        if len(self.returns) < MIN_HISTORICAL_DAYS:
            msg = (
                f"Insufficient historical data: {len(self.returns)} days "
                f"(minimum {MIN_HISTORICAL_DAYS} required)"
            )
            raise ValueError(msg)

        # Check memory requirements (warn if >1GB)
        num_assets = len(positions)
        memory_mb = (cfg.num_simulations * cfg.horizon_days * num_assets * 8) / (1024**2)
        if memory_mb > MAX_MEMORY_MB:
            logger.warning(
                f"Large simulation ({memory_mb:.0f} MB), consider reducing num_simulations or horizon_days"
            )

        rng = (
            np.random.default_rng(cfg.random_seed)
            if cfg.random_seed is not None
            else np.random.default_rng()
        )

        # Align positions with returns columns
        symbols = [sym for sym in self.returns.columns if sym in positions]
        if len(symbols) != len(positions):
            missing = set(positions.keys()) - set(symbols)
            msg = f"Missing historical data for symbols: {missing}"
            raise ValueError(msg)

        # Calculate weights
        weights = np.array([positions[sym] for sym in symbols])
        weights = weights / weights.sum()

        # Generate simulations based on method
        if cfg.method == SimulationMethod.PARAMETRIC:
            sim_returns = self._simulate_parametric(symbols, cfg.num_simulations, cfg.horizon_days, rng)
        elif cfg.method == SimulationMethod.BOOTSTRAP:
            sim_returns = self._simulate_bootstrap(symbols, cfg.num_simulations, cfg.horizon_days, rng)
        elif cfg.method == SimulationMethod.GBM:
            sim_returns = self._simulate_gbm(symbols, cfg.num_simulations, cfg.horizon_days, rng)
        else:
            msg = f"Unknown simulation method: {cfg.method}"
            raise ValueError(msg)

        # Validate simulation results
        if np.isnan(sim_returns).any() or np.isinf(sim_returns).any():
            msg = "Simulation produced NaN or Inf values"
            raise ValueError(msg)

        # Calculate portfolio returns (weighted)
        portfolio_returns = (sim_returns * weights).sum(axis=2)  # (num_sims, horizon_days)
        cumulative_returns = (1 + portfolio_returns).cumprod(axis=1) - 1  # Cumulative product
        final_returns = cumulative_returns[:, -1]  # (num_sims,)

        # Calculate metrics
        prob_loss_gt_threshold = float((final_returns < -cfg.loss_threshold).mean())
        var_95 = float(np.percentile(final_returns, 5))
        cvar_95 = float(final_returns[final_returns <= var_95].mean())

        # Calculate expected worst drawdown (true max drawdown from equity curve)
        wealth = 1.0 + cumulative_returns
        running_max = np.maximum.accumulate(wealth, axis=1)
        drawdowns = wealth / running_max - 1.0
        max_drawdowns = drawdowns.min(axis=1)
        worst_5pct_drawdowns = max_drawdowns[max_drawdowns <= np.percentile(max_drawdowns, 5)]
        expected_worst_drawdown = float(worst_5pct_drawdowns.mean())

        # Calculate recovery time (median days to recover from max drawdown)
        median_recovery_days = self._calculate_recovery_time(cumulative_returns)

        return MonteCarloResult(
            simulation_method=cfg.method,
            num_simulations=cfg.num_simulations,
            horizon_days=cfg.horizon_days,
            prob_loss_gt_threshold=prob_loss_gt_threshold,
            expected_worst_drawdown=expected_worst_drawdown,
            var_95=var_95,
            cvar_95=cvar_95,
            median_recovery_days=median_recovery_days,
            mean_return=float(final_returns.mean()),
            std_return=float(final_returns.std()),
            min_return=float(final_returns.min()),
            max_return=float(final_returns.max()),
            simulated_returns=final_returns.tolist(),
        )

    def _simulate_parametric(
        self, symbols: list[str], num_simulations: int, horizon_days: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Parametric simulation using multivariate normal distribution.

        Returns:
            Array of shape (num_sims, horizon_days, num_assets)
        """
        mean = self.mean_returns[symbols].values
        cov = self.cov_matrix.loc[symbols, symbols].values

        # Generate random returns: (num_sims, horizon_days, num_assets)
        mean_array = np.asarray(mean)
        return rng.multivariate_normal(mean=mean_array, cov=cov, size=(num_simulations, horizon_days))

    def _simulate_bootstrap(
        self, symbols: list[str], num_simulations: int, horizon_days: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Bootstrap simulation using historical resampling.

        Returns:
            Array of shape (num_sims, horizon_days, num_assets)
        """
        returns_data = self.returns[symbols].values

        # Resample with replacement (preserves correlation)
        indices = rng.integers(0, len(returns_data), size=(num_simulations, horizon_days))
        return returns_data[indices]  # (num_sims, horizon_days, num_assets)

    def _simulate_gbm(
        self, symbols: list[str], num_simulations: int, horizon_days: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Geometric Brownian Motion simulation.

        Returns:
            Array of shape (num_sims, horizon_days, num_assets)
        """
        mean = self.mean_returns[symbols].values
        cov = self.cov_matrix.loc[symbols, symbols].values

        dt = 1.0  # Daily time step
        num_assets = len(symbols)

        # Generate correlated random shocks
        z = rng.multivariate_normal(mean=np.zeros(num_assets), cov=cov, size=(num_simulations, horizon_days))

        # GBM returns using drift-diffusion formula
        sigma = np.sqrt(np.diag(cov))
        mean_array = np.asarray(mean)
        drift = (mean_array - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * z

        return np.exp(drift + diffusion) - 1  # (num_sims, horizon_days, num_assets)

    def _calculate_recovery_time(self, cumulative_returns: np.ndarray) -> float | None:
        """Calculate median recovery time from max drawdown.

        Args:
            cumulative_returns: Array of shape (num_sims, horizon_days)

        Returns:
            Median days to recover from max drawdown, or None if no recovery
        """
        recovery_times = []

        for sim_path in cumulative_returns:
            # Convert to equity (wealth)
            equity = 1.0 + sim_path
            running_max = np.maximum.accumulate(equity)
            drawdowns = (equity - running_max) / running_max
            max_dd_idx = np.argmin(drawdowns)

            if max_dd_idx == len(sim_path) - 1:
                # Max drawdown at end, no recovery
                continue

            # Find recovery point (first time equity >= running max)
            recovery_idx = np.where(equity[max_dd_idx:] >= running_max[max_dd_idx])[0]
            if len(recovery_idx) > 0:
                recovery_times.append(float(recovery_idx[0]))

        if not recovery_times:
            return None

        return float(np.median(recovery_times))
