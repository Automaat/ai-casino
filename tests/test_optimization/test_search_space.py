"""Tests for search space module."""

import pytest

from src.optimization.search_space import (
    ENSEMBLE_SEARCH_SPACE,
    MEAN_REVERSION_SEARCH_SPACE,
    MOMENTUM_SEARCH_SPACE,
    TREND_FOLLOWING_SEARCH_SPACE,
    ParamRange,
    StrategyType,
    get_search_space,
)


class TestParamRange:
    """Tests for ParamRange."""

    def test_int_param(self):
        """Test integer parameter range."""
        param = ParamRange(name="rsi_period", low=7, high=21, step=1, is_int=True)

        assert param.name == "rsi_period"
        assert param.low == 7
        assert param.high == 21
        assert param.is_int is True

    def test_float_param(self):
        """Test float parameter range."""
        param = ParamRange(name="bb_std", low=1.5, high=3.0, step=0.25)

        assert param.name == "bb_std"
        assert param.low == 1.5
        assert param.high == 3.0
        assert param.step == 0.25
        assert param.is_int is False


class TestSearchSpace:
    """Tests for SearchSpace."""

    def test_momentum_search_space(self):
        """Test momentum search space structure."""
        space = MOMENTUM_SEARCH_SPACE

        assert space.strategy == StrategyType.MOMENTUM
        assert len(space.params) == 6

        param_names = {p.name for p in space.params}
        assert "rsi_period" in param_names
        assert "macd_fast" in param_names
        assert "macd_slow" in param_names

    def test_trend_following_search_space(self):
        """Test trend following search space structure."""
        space = TREND_FOLLOWING_SEARCH_SPACE

        assert space.strategy == StrategyType.TREND_FOLLOWING
        assert len(space.params) == 4

        param_names = {p.name for p in space.params}
        assert "sma_fast" in param_names
        assert "sma_slow" in param_names
        assert "adx_period" in param_names

    def test_mean_reversion_search_space(self):
        """Test mean reversion search space structure."""
        space = MEAN_REVERSION_SEARCH_SPACE

        assert space.strategy == StrategyType.MEAN_REVERSION
        assert len(space.params) == 2

        param_names = {p.name for p in space.params}
        assert "bb_period" in param_names
        assert "bb_std" in param_names

    def test_ensemble_search_space(self):
        """Test ensemble search space structure."""
        space = ENSEMBLE_SEARCH_SPACE

        assert space.strategy == StrategyType.ENSEMBLE
        assert len(space.params) == 4

        param_names = {p.name for p in space.params}
        assert "momentum_weight" in param_names
        assert "mean_reversion_weight" in param_names
        assert "trend_following_weight" in param_names
        assert "ensemble_threshold" in param_names

    def test_constraints(self):
        """Test search space constraints."""
        assert MOMENTUM_SEARCH_SPACE.constraints is not None
        assert "macd_fast < macd_slow" in MOMENTUM_SEARCH_SPACE.constraints

        assert TREND_FOLLOWING_SEARCH_SPACE.constraints is not None
        assert "sma_fast < sma_slow" in TREND_FOLLOWING_SEARCH_SPACE.constraints


class TestGetSearchSpace:
    """Tests for get_search_space function."""

    def test_get_momentum(self):
        """Test getting momentum search space."""
        space = get_search_space("momentum")
        assert space.strategy == StrategyType.MOMENTUM

    def test_get_trend_following(self):
        """Test getting trend following search space."""
        space = get_search_space("trend_following")
        assert space.strategy == StrategyType.TREND_FOLLOWING

    def test_get_mean_reversion(self):
        """Test getting mean reversion search space."""
        space = get_search_space("mean_reversion")
        assert space.strategy == StrategyType.MEAN_REVERSION

    def test_get_ensemble(self):
        """Test getting ensemble search space."""
        space = get_search_space("ensemble")
        assert space.strategy == StrategyType.ENSEMBLE

    def test_case_insensitive(self):
        """Test case insensitivity."""
        space = get_search_space("MOMENTUM")
        assert space.strategy == StrategyType.MOMENTUM

    def test_invalid_strategy(self):
        """Test error on invalid strategy."""
        with pytest.raises(ValueError, match="Unknown strategy"):
            get_search_space("invalid_strategy")
