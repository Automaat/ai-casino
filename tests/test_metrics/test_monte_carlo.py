"""Tests for Monte Carlo portfolio stress testing."""

import numpy as np
import pandas as pd
import pytest

from src.metrics.monte_carlo import MonteCarloResult, MonteCarloSimulator, SimulationMethod


@pytest.fixture
def sample_returns_df():
    """3 assets, 100 days of returns."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=100)
    return pd.DataFrame(
        {
            "AAPL": np.random.normal(0.001, 0.02, 100),
            "MSFT": np.random.normal(0.0008, 0.018, 100),
            "GOOGL": np.random.normal(0.0012, 0.022, 100),
        },
        index=dates,
    )


@pytest.fixture
def sample_positions():
    """Sample portfolio positions."""
    return {"AAPL": 10000.0, "MSFT": 8000.0, "GOOGL": 12000.0}


def test_parametric_simulation(sample_returns_df, sample_positions):
    """Test parametric method returns valid results."""
    simulator = MonteCarloSimulator(sample_returns_df)
    result = simulator.simulate(
        positions=sample_positions,
        num_simulations=1000,
        method=SimulationMethod.PARAMETRIC,
        random_seed=42,
    )

    assert isinstance(result, MonteCarloResult)
    assert result.simulation_method == SimulationMethod.PARAMETRIC
    assert result.num_simulations == 1000
    assert 0.0 <= result.prob_loss_gt_10pct <= 1.0
    assert result.cvar_95 <= result.var_95  # CVaR more negative than VaR (worse loss)
    assert len(result.simulated_returns) == 1000
    assert isinstance(result.mean_return, float)
    assert isinstance(result.std_return, float)


def test_bootstrap_simulation(sample_returns_df, sample_positions):
    """Test bootstrap method preserves correlation."""
    simulator = MonteCarloSimulator(sample_returns_df)
    result = simulator.simulate(
        positions=sample_positions,
        num_simulations=1000,
        method=SimulationMethod.BOOTSTRAP,
        random_seed=42,
    )

    assert isinstance(result, MonteCarloResult)
    assert result.simulation_method == SimulationMethod.BOOTSTRAP
    assert 0.0 <= result.prob_loss_gt_10pct <= 1.0
    assert result.cvar_95 <= result.var_95


def test_gbm_simulation(sample_returns_df, sample_positions):
    """Test GBM method produces path-dependent results."""
    simulator = MonteCarloSimulator(sample_returns_df)
    result = simulator.simulate(
        positions=sample_positions,
        num_simulations=1000,
        method=SimulationMethod.GBM,
        random_seed=42,
    )

    assert isinstance(result, MonteCarloResult)
    assert result.simulation_method == SimulationMethod.GBM
    assert 0.0 <= result.prob_loss_gt_10pct <= 1.0
    assert result.cvar_95 <= result.var_95


def test_singular_covariance_matrix():
    """Test ridge regularization for singular covariance."""
    # Create perfectly correlated returns (singular covariance)
    dates = pd.date_range("2024-01-01", periods=50)
    base_returns = np.random.normal(0.001, 0.02, 50)
    returns_df = pd.DataFrame(
        {
            "AAPL": base_returns,
            "MSFT": base_returns * 1.1,  # Perfectly correlated
            "GOOGL": base_returns * 0.9,  # Perfectly correlated
        },
        index=dates,
    )

    # Should handle singular matrix with ridge regularization
    simulator = MonteCarloSimulator(returns_df)
    result = simulator.simulate(
        positions={"AAPL": 10000.0, "MSFT": 10000.0, "GOOGL": 10000.0},
        num_simulations=100,
        random_seed=42,
    )

    assert isinstance(result, MonteCarloResult)
    assert 0.0 <= result.prob_loss_gt_10pct <= 1.0


def test_insufficient_data():
    """Test error handling with insufficient historical data."""
    dates = pd.date_range("2024-01-01", periods=20)  # Only 20 days
    returns_df = pd.DataFrame(
        {"AAPL": np.random.normal(0.001, 0.02, 20), "MSFT": np.random.normal(0.0008, 0.018, 20)},
        index=dates,
    )

    simulator = MonteCarloSimulator(returns_df)
    with pytest.raises(ValueError, match="Insufficient historical data"):
        simulator.simulate(positions={"AAPL": 10000.0, "MSFT": 10000.0}, num_simulations=100, random_seed=42)


def test_empty_portfolio():
    """Test error handling with no positions."""
    dates = pd.date_range("2024-01-01", periods=100)
    returns_df = pd.DataFrame({"AAPL": np.random.normal(0.001, 0.02, 100)}, index=dates)

    simulator = MonteCarloSimulator(returns_df)
    with pytest.raises(ValueError, match="Portfolio positions cannot be empty"):
        simulator.simulate(positions={}, num_simulations=100, random_seed=42)


def test_missing_symbols():
    """Test error when portfolio contains symbols not in historical data."""
    dates = pd.date_range("2024-01-01", periods=100)
    returns_df = pd.DataFrame({"AAPL": np.random.normal(0.001, 0.02, 100)}, index=dates)

    simulator = MonteCarloSimulator(returns_df)
    with pytest.raises(ValueError, match="Missing historical data for symbols"):
        simulator.simulate(
            positions={"AAPL": 10000.0, "TSLA": 10000.0},  # TSLA not in data
            num_simulations=100,
            random_seed=42,
        )


def test_metrics_consistency(sample_returns_df, sample_positions):
    """Test that metrics are consistent and within expected ranges."""
    simulator = MonteCarloSimulator(sample_returns_df)
    result = simulator.simulate(positions=sample_positions, num_simulations=5000, random_seed=42)

    # VaR should be negative (5th percentile of losses)
    assert result.var_95 <= 0.0

    # CVaR should be more negative than VaR
    assert result.cvar_95 <= result.var_95

    # Expected worst drawdown should be negative
    assert result.expected_worst_drawdown <= 0.0

    # Probability should be between 0 and 1
    assert 0.0 <= result.prob_loss_gt_10pct <= 1.0

    # Mean return can be positive or negative
    assert -1.0 <= result.mean_return <= 1.0

    # Std should be positive
    assert result.std_return > 0.0


def test_reproducibility(sample_returns_df, sample_positions):
    """Test that same seed produces same results."""
    simulator = MonteCarloSimulator(sample_returns_df)

    result1 = simulator.simulate(positions=sample_positions, num_simulations=1000, random_seed=123)
    result2 = simulator.simulate(positions=sample_positions, num_simulations=1000, random_seed=123)

    assert result1.prob_loss_gt_10pct == result2.prob_loss_gt_10pct
    assert result1.var_95 == result2.var_95
    assert result1.cvar_95 == result2.cvar_95
    assert result1.simulated_returns == result2.simulated_returns


def test_horizon_impact(sample_returns_df, sample_positions):
    """Test that longer horizons produce wider distribution."""
    simulator = MonteCarloSimulator(sample_returns_df)

    short_horizon = simulator.simulate(
        positions=sample_positions, num_simulations=1000, horizon_days=30, random_seed=42
    )
    long_horizon = simulator.simulate(
        positions=sample_positions, num_simulations=1000, horizon_days=252, random_seed=42
    )

    # Longer horizon should have higher volatility
    assert long_horizon.std_return > short_horizon.std_return


def test_recovery_time_calculation(sample_returns_df, sample_positions):
    """Test recovery time calculation."""
    simulator = MonteCarloSimulator(sample_returns_df)
    result = simulator.simulate(positions=sample_positions, num_simulations=1000, random_seed=42)

    # Recovery time can be None if no recovery, or positive float
    if result.median_recovery_days is not None:
        assert result.median_recovery_days >= 0.0
