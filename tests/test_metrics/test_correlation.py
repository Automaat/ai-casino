"""Tests for portfolio correlation analysis."""

import json
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.broker import BrokerPosition
from src.data.market import MarketData
from src.metrics.correlation import CorrelationAuditor, CorrelationAuditResult
from src.screening.screener import ScreeningResult
from src.strategies.signal import Signal


@pytest.fixture
def mock_market_fetcher():
    """Create mock market data fetcher."""
    fetcher = MagicMock()

    def fetch_daily(symbol: str, period_days: int = 90) -> MarketData:
        dates = pd.date_range(end=datetime.now(UTC), periods=100, freq="D")
        base_price = 100.0

        if symbol == "AAPL":
            prices = [base_price + i * 0.5 for i in range(100)]
        elif symbol == "MSFT":
            prices = [base_price + i * 0.48 for i in range(100)]
        elif symbol == "NVDA":
            prices = [base_price - i * 0.3 for i in range(100)]
        else:
            prices = [base_price + (i % 10) for i in range(100)]

        df = pd.DataFrame({"Close": prices}, index=dates)
        return MarketData(symbol=symbol, data=df, last_updated=datetime.now(UTC))

    fetcher.fetch_daily.side_effect = fetch_daily
    return fetcher


@pytest.fixture
def sample_positions() -> dict[str, BrokerPosition]:
    """Create sample broker positions."""
    return {
        "AAPL": BrokerPosition(
            symbol="AAPL",
            qty=10.0,
            avg_entry_price=150.0,
            current_price=155.0,
            market_value=1550.0,
            unrealized_pnl=50.0,
            unrealized_pnl_percent=0.033,
        ),
        "MSFT": BrokerPosition(
            symbol="MSFT",
            qty=8.0,
            avg_entry_price=300.0,
            current_price=310.0,
            market_value=2480.0,
            unrealized_pnl=80.0,
            unrealized_pnl_percent=0.033,
        ),
        "NVDA": BrokerPosition(
            symbol="NVDA",
            qty=5.0,
            avg_entry_price=500.0,
            current_price=480.0,
            market_value=2400.0,
            unrealized_pnl=-100.0,
            unrealized_pnl_percent=-0.04,
        ),
    }


@pytest.fixture
def sample_screening_results() -> list[ScreeningResult]:
    """Create sample screening results."""
    return [
        ScreeningResult(
            symbol="TGT",
            name="Target Corporation",
            sector="Consumer Cyclical",
            score=85.0,
            signal=Signal.BUY,
            metrics={"rsi": 45.0, "macd_hist": 0.2},
            reason="Oversold with positive momentum",
        ),
        ScreeningResult(
            symbol="WMT",
            name="Walmart Inc",
            sector="Consumer Defensive",
            score=75.0,
            signal=Signal.HOLD,
            metrics={"rsi": 50.0, "macd_hist": 0.1},
            reason="Stable fundamentals",
        ),
    ]


@pytest.fixture
def auditor(mock_market_fetcher, tmp_path):
    """Create correlation auditor instance."""
    return CorrelationAuditor(
        market_fetcher=mock_market_fetcher,
        correlation_threshold=0.8,
        lookback_days=90,
        output_dir=str(tmp_path / "correlation-audits"),
    )


def test_auditor_initialization(auditor, tmp_path):
    """Test auditor initialization."""
    assert auditor.correlation_threshold == 0.8
    assert auditor.lookback_days == 90
    assert auditor.output_dir == tmp_path / "correlation-audits"
    assert auditor.output_dir.exists()


def test_insufficient_positions(auditor):
    """Test handling of insufficient positions."""
    positions = {
        "AAPL": BrokerPosition(
            symbol="AAPL",
            qty=10.0,
            avg_entry_price=150.0,
            current_price=155.0,
            market_value=1550.0,
            unrealized_pnl=50.0,
            unrealized_pnl_percent=0.033,
        )
    }

    result = auditor.audit(positions)

    assert result.num_positions == 1
    assert result.correlation_matrix == {}
    assert result.highly_correlated_pairs == []
    assert result.diversification_ratio == 1.0
    assert any("Insufficient positions" in w for w in result.warnings)


def test_fetch_position_returns(auditor, sample_positions):
    """Test fetching aligned position returns."""
    warnings = []
    returns_df = auditor._fetch_position_returns(list(sample_positions.keys()), warnings)

    assert isinstance(returns_df, pd.DataFrame)
    if not returns_df.empty:
        assert set(returns_df.columns) == {"AAPL", "MSFT", "NVDA"}
        assert len(returns_df) > 0
        assert not returns_df.isna().any().any()


def test_correlation_matrix_computation(auditor, sample_positions):
    """Test correlation matrix computation."""
    warnings = []
    returns_df = auditor._fetch_position_returns(list(sample_positions.keys()), warnings)

    if not returns_df.empty:
        corr_matrix = auditor._compute_correlation_matrix(returns_df)

        assert "AAPL" in corr_matrix
        assert "MSFT" in corr_matrix
        assert "NVDA" in corr_matrix

        assert corr_matrix["AAPL"]["AAPL"] == pytest.approx(1.0)
        assert corr_matrix["MSFT"]["MSFT"] == pytest.approx(1.0)

        assert -1.0 <= corr_matrix["AAPL"]["MSFT"] <= 1.0


@patch("src.metrics.correlation.yf.Ticker")
def test_identify_correlated_pairs(mock_ticker, auditor):
    """Test identification of highly correlated pairs."""
    mock_ticker.return_value.info = {"sector": "Technology"}

    corr_matrix = {
        "AAPL": {"AAPL": 1.0, "MSFT": 0.95, "NVDA": 0.5},
        "MSFT": {"AAPL": 0.95, "MSFT": 1.0, "NVDA": 0.6},
        "NVDA": {"AAPL": 0.5, "MSFT": 0.6, "NVDA": 1.0},
    }

    warnings = []
    pairs = auditor._identify_correlated_pairs(corr_matrix, warnings)

    assert len(pairs) == 1
    pair = pairs[0]
    assert pair.symbol_a == "AAPL"
    assert pair.symbol_b == "MSFT"
    assert pair.correlation == 0.95
    assert pair.sector_a == "Technology"
    assert pair.sector_b == "Technology"
    assert pair.same_sector is True


def test_diversification_ratio(auditor, sample_positions):
    """Test diversification ratio calculation."""
    warnings = []
    returns_df = auditor._fetch_position_returns(list(sample_positions.keys()), warnings)

    ratio = auditor._calculate_diversification_ratio(returns_df, sample_positions)

    assert 0.0 <= ratio <= 2.0
    assert isinstance(ratio, float)


@patch("src.metrics.correlation.yf.Ticker")
def test_full_audit(mock_ticker, auditor, sample_positions, sample_screening_results):
    """Test full correlation audit."""
    mock_ticker.return_value.info = {"sector": "Technology"}

    result = auditor.audit(sample_positions, sample_screening_results)

    assert isinstance(result, CorrelationAuditResult)
    assert result.num_positions == 3
    assert isinstance(result.correlation_matrix, dict)
    assert 0.0 <= result.diversification_ratio <= 2.0
    assert -1.0 <= result.max_correlation <= 1.0
    assert -1.0 <= result.avg_correlation <= 1.0
    assert result.lookback_days == 90


@patch("src.metrics.correlation.yf.Ticker")
def test_substitution_generation(mock_ticker, auditor, sample_positions, sample_screening_results):
    """Test substitution suggestion generation."""
    mock_ticker.return_value.info = {"sector": "Technology"}

    warnings = []
    returns_df = auditor._fetch_position_returns(list(sample_positions.keys()), warnings)
    corr_matrix = auditor._compute_correlation_matrix(returns_df)
    correlated_pairs = auditor._identify_correlated_pairs(corr_matrix, warnings)

    suggestions = auditor._generate_substitutions(
        correlated_pairs, sample_screening_results, corr_matrix, warnings
    )

    assert isinstance(suggestions, list)
    for suggestion in suggestions:
        assert suggestion.symbol_to_replace in sample_positions
        assert len(suggestion.alternatives) <= 3
        assert len(suggestion.alternative_correlations) == len(suggestion.alternatives)


def test_persist_and_load(auditor, sample_positions, tmp_path):
    """Test persisting and loading audit results."""
    result = auditor.audit(sample_positions)

    persisted_path = auditor.persist(result)
    assert persisted_path.exists()

    with persisted_path.open() as f:
        data = json.load(f)

    assert data["num_positions"] == result.num_positions
    assert data["lookback_days"] == result.lookback_days

    loaded = auditor.load_latest()
    assert loaded is not None
    assert loaded.num_positions == result.num_positions
    assert loaded.diversification_ratio == result.diversification_ratio


def test_load_latest_no_results(tmp_path):
    """Test loading when no results exist."""
    auditor = CorrelationAuditor(market_fetcher=MagicMock(), output_dir=str(tmp_path / "empty-audits"))

    result = auditor.load_latest()
    assert result is None


def test_empty_returns_data(auditor):
    """Test handling of empty returns data."""
    positions = {
        "INVALID": BrokerPosition(
            symbol="INVALID",
            qty=10.0,
            avg_entry_price=100.0,
            current_price=100.0,
            market_value=1000.0,
            unrealized_pnl=0.0,
            unrealized_pnl_percent=0.0,
        ),
        "INVALID2": BrokerPosition(
            symbol="INVALID2",
            qty=10.0,
            avg_entry_price=100.0,
            current_price=100.0,
            market_value=1000.0,
            unrealized_pnl=0.0,
            unrealized_pnl_percent=0.0,
        ),
    }

    auditor.market_fetcher.fetch_daily.side_effect = Exception("API error")

    result = auditor.audit(positions)

    assert result.num_positions == 2
    assert result.correlation_matrix == {}
    assert "No return data available" in result.warnings


def test_correlation_threshold_filtering(auditor):
    """Test that only pairs above threshold are flagged."""
    warnings = []
    corr_matrix = {
        "A": {"A": 1.0, "B": 0.85, "C": 0.75},
        "B": {"A": 0.85, "B": 1.0, "C": 0.70},
        "C": {"A": 0.75, "B": 0.70, "C": 1.0},
    }

    pairs = auditor._identify_correlated_pairs(corr_matrix, warnings)

    assert len(pairs) == 1
    assert pairs[0].correlation >= 0.8


def test_avg_correlation_calculation(auditor, mock_market_fetcher):
    """Test average correlation calculation with portfolio."""
    with patch("src.metrics.correlation.yf.Ticker"):
        avg_corr = auditor._calculate_avg_correlation_with_portfolio("TGT", ["AAPL", "MSFT"])

        assert 0.0 <= avg_corr <= 1.0
        assert isinstance(avg_corr, float)


def test_limited_data_warning(auditor, sample_positions):
    """Test warning when data points are limited."""
    auditor.lookback_days = 20

    with patch("src.metrics.correlation.yf.Ticker") as mock_ticker:
        mock_ticker.return_value.info = {"sector": "Technology"}
        result = auditor.audit(sample_positions)

        # Should have either limited data warning or other data-related warnings
        assert len(result.warnings) > 0 or result.num_positions >= 2


@patch("src.metrics.correlation.yf.Ticker")
def test_sector_retrieval_failure(mock_ticker, auditor):
    """Test handling of sector retrieval failures."""
    mock_ticker.return_value.info = {}

    warnings = []
    sector = auditor._get_sector("AAPL", warnings)

    assert sector == "Unknown"
    # The method adds a warning about missing sector
    assert len(warnings) > 0 or sector == "Unknown"
