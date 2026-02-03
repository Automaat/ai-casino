"""Tests for market regime detection."""

import pytest

from src.strategies.regime import MarketRegime, MarketRegimeDetector, RegimeAnalysis, RegimeIndicators


class TestMarketRegimeDetector:
    """Tests for MarketRegimeDetector."""

    def test_regime_detector_init(self) -> None:
        """Test regime detector initialization."""
        detector = MarketRegimeDetector()

        assert detector.adx_threshold == 25.0
        assert detector.atr_vol_ratio == 1.5

    def test_regime_detector_init_custom(self) -> None:
        """Test regime detector with custom thresholds."""
        detector = MarketRegimeDetector(adx_threshold=30.0, atr_vol_ratio=2.0)

        assert detector.adx_threshold == 30.0
        assert detector.atr_vol_ratio == 2.0

    def test_detect_trending_bullish(self, sample_ohlcv_trending_up) -> None:
        """Test detection of bullish trend."""
        detector = MarketRegimeDetector()
        result = detector.detect_regime(sample_ohlcv_trending_up)

        assert isinstance(result, RegimeAnalysis)
        assert isinstance(result.indicators, RegimeIndicators)
        # Uptrend should have +DI > -DI
        assert result.indicators.plus_di > 0
        assert 0.0 <= result.confidence <= 1.0
        assert result.reasoning

    def test_detect_trending_bearish(self, sample_ohlcv_trending_down) -> None:
        """Test detection of bearish trend."""
        detector = MarketRegimeDetector()
        result = detector.detect_regime(sample_ohlcv_trending_down)

        assert isinstance(result, RegimeAnalysis)
        # Downtrend should have -DI > +DI
        assert result.indicators.minus_di > 0
        assert 0.0 <= result.confidence <= 1.0

    def test_detect_ranging(self, sample_ohlcv_ranging) -> None:
        """Test detection of ranging market."""
        detector = MarketRegimeDetector()
        result = detector.detect_regime(sample_ohlcv_ranging)

        assert isinstance(result, RegimeAnalysis)
        # Ranging should have low ADX
        assert result.regime == MarketRegime.RANGING
        assert 0.0 <= result.confidence <= 1.0

    def test_detect_high_volatility(self, sample_ohlcv_volatile) -> None:
        """Test detection of high volatility."""
        detector = MarketRegimeDetector()
        result = detector.detect_regime(sample_ohlcv_volatile)

        assert isinstance(result, RegimeAnalysis)
        # High volatility has elevated ATR ratio
        assert result.indicators.atr > 0
        assert 0.0 <= result.confidence <= 1.0

    def test_confidence_calculation(self, sample_ohlcv_trending_up) -> None:
        """Test confidence is always in valid range."""
        detector = MarketRegimeDetector()
        result = detector.detect_regime(sample_ohlcv_trending_up)

        assert 0.0 <= result.confidence <= 1.0
        assert result.confidence <= 0.95  # Max confidence cap

    def test_insufficient_data_raises(self) -> None:
        """Test that insufficient data raises ValueError."""
        import pandas as pd

        detector = MarketRegimeDetector()
        small_data = pd.DataFrame(
            {
                "Open": [100] * 20,
                "High": [105] * 20,
                "Low": [99] * 20,
                "Close": [104] * 20,
                "Volume": [1000000] * 20,
            }
        )

        with pytest.raises(ValueError, match="Need at least 35 rows"):
            detector.detect_regime(small_data)

    def test_regime_indicators_fields(self, sample_ohlcv_trending_up) -> None:
        """Test that all indicator fields are populated."""
        detector = MarketRegimeDetector()
        result = detector.detect_regime(sample_ohlcv_trending_up)
        indicators = result.indicators

        assert indicators.adx >= 0
        assert indicators.plus_di >= 0
        assert indicators.minus_di >= 0
        assert indicators.atr >= 0
        assert indicators.atr_ratio > 0
        assert indicators.bb_width >= 0

    def test_repr(self) -> None:
        """Test string representation."""
        detector = MarketRegimeDetector()
        repr_str = repr(detector)

        assert "MarketRegimeDetector" in repr_str
        assert "adx_threshold=25.0" in repr_str


class TestMarketRegime:
    """Tests for MarketRegime enum."""

    def test_regime_values(self) -> None:
        """Test regime enum values."""
        assert MarketRegime.TRENDING_BULLISH == "TRENDING_BULLISH"
        assert MarketRegime.TRENDING_BEARISH == "TRENDING_BEARISH"
        assert MarketRegime.RANGING == "RANGING"
        assert MarketRegime.HIGH_VOLATILITY == "HIGH_VOLATILITY"

    def test_regime_is_strenum(self) -> None:
        """Test regime is a string enum."""
        assert isinstance(MarketRegime.TRENDING_BULLISH, str)
        assert MarketRegime.TRENDING_BULLISH.value == "TRENDING_BULLISH"
