"""Tests for stock universe fetcher."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.data.universe import StockInfo, StockUniverse, StockUniverseFetcher


@pytest.fixture
def mock_sp500_html():
    """Mock S&P 500 Wikipedia HTML."""
    return """
    <table id="constituents">
        <tr><th>Symbol</th><th>Security</th><th>CIK</th><th>GICS Sector</th><th>Sub-Industry</th></tr>
        <tr><td>AAPL</td><td>Apple Inc.</td><td>320193</td><td>Info Tech</td><td>Hardware</td></tr>
        <tr><td>MSFT</td><td>Microsoft Corp</td><td>789019</td><td>Info Tech</td><td>Software</td></tr>
        <tr><td>BRK.B</td><td>Berkshire</td><td>1067983</td><td>Financials</td><td>Holdings</td></tr>
    </table>
    """


@pytest.fixture
def mock_nasdaq100_html():
    """Mock NASDAQ 100 Wikipedia HTML."""
    return """
    <table id="constituents">
        <tr><th>Company</th><th>Ticker</th><th>GICS Sector</th><th>GICS Sub-Industry</th></tr>
        <tr><td>Apple Inc.</td><td>AAPL</td><td>Information Technology</td><td>Technology Hardware</td></tr>
        <tr><td>NVIDIA Corp</td><td>NVDA</td><td>Information Technology</td><td>Semiconductors</td></tr>
    </table>
    """


@pytest.fixture
def universe_fetcher(tmp_path):
    """Create StockUniverseFetcher with temp cache."""
    return StockUniverseFetcher(cache_dir=str(tmp_path / "universe_cache"))


class TestStockInfo:
    """Tests for StockInfo model."""

    def test_create(self):
        """Test StockInfo creation."""
        info = StockInfo(symbol="AAPL", name="Apple Inc.", sector="Technology", industry="Hardware")

        assert info.symbol == "AAPL"
        assert info.name == "Apple Inc."
        assert info.sector == "Technology"
        assert info.industry == "Hardware"


class TestStockUniverse:
    """Tests for StockUniverse model."""

    def test_create(self):
        """Test StockUniverse creation."""
        stocks = [
            StockInfo(symbol="AAPL", name="Apple", sector="Tech", industry="Hardware"),
            StockInfo(symbol="MSFT", name="Microsoft", sector="Tech", industry="Software"),
        ]
        universe = StockUniverse(name="TEST", stocks=stocks, fetched_at=datetime.now())

        assert universe.name == "TEST"
        assert len(universe.stocks) == 2
        assert universe.stocks[0].symbol == "AAPL"


class TestStockUniverseFetcher:
    """Tests for StockUniverseFetcher."""

    def test_init(self, tmp_path):
        """Test fetcher initialization."""
        cache_dir = tmp_path / "cache"
        fetcher = StockUniverseFetcher(cache_dir=str(cache_dir))

        assert cache_dir.exists()
        assert "StockUniverseFetcher" in repr(fetcher)

    def test_init_default_cache(self, tmp_path):
        """Test fetcher creates cache directory when given a path."""
        cache_dir = tmp_path / "data" / "cache" / "universe"
        fetcher = StockUniverseFetcher(cache_dir=str(cache_dir))
        assert cache_dir.exists()
        fetcher.clear_cache()

    @patch("src.data.universe.requests.get")
    def test_fetch_sp500(self, mock_get, universe_fetcher, mock_sp500_html):
        """Test fetching S&P 500 stocks."""
        mock_response = MagicMock()
        mock_response.text = mock_sp500_html
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        universe = universe_fetcher.fetch_sp500()

        assert universe.name == "SP500"
        assert len(universe.stocks) == 3
        assert universe.stocks[0].symbol == "AAPL"
        assert universe.stocks[2].symbol == "BRK-B"  # . replaced with -

    @patch("src.data.universe.requests.get")
    def test_fetch_nasdaq100(self, mock_get, universe_fetcher, mock_nasdaq100_html):
        """Test fetching NASDAQ 100 stocks."""
        mock_response = MagicMock()
        mock_response.text = mock_nasdaq100_html
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        universe = universe_fetcher.fetch_nasdaq100()

        assert universe.name == "NASDAQ100"
        assert len(universe.stocks) == 2
        assert universe.stocks[0].symbol == "AAPL"
        assert universe.stocks[1].symbol == "NVDA"

    @patch("src.data.universe.requests.get")
    def test_fetch_combined_deduplicates(
        self, mock_get, universe_fetcher, mock_sp500_html, mock_nasdaq100_html
    ):
        """Test combined fetch deduplicates stocks."""
        responses = [
            MagicMock(text=mock_sp500_html, raise_for_status=MagicMock()),
            MagicMock(text=mock_nasdaq100_html, raise_for_status=MagicMock()),
        ]
        mock_get.side_effect = responses

        universe = universe_fetcher.fetch_combined()

        assert universe.name == "COMBINED"
        symbols = [s.symbol for s in universe.stocks]
        assert symbols.count("AAPL") == 1  # Deduplicated
        assert "NVDA" in symbols
        assert "MSFT" in symbols

    @patch("src.data.universe.requests.get")
    def test_caching(self, mock_get, universe_fetcher, mock_sp500_html):
        """Test that results are cached."""
        mock_response = MagicMock()
        mock_response.text = mock_sp500_html
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        universe1 = universe_fetcher.fetch_sp500()
        universe2 = universe_fetcher.fetch_sp500()

        assert mock_get.call_count == 1  # Only called once
        assert universe1.stocks == universe2.stocks

    @patch("src.data.universe.requests.get")
    def test_clear_cache(self, mock_get, universe_fetcher, mock_sp500_html):
        """Test cache clearing."""
        mock_response = MagicMock()
        mock_response.text = mock_sp500_html
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        universe_fetcher.fetch_sp500()
        universe_fetcher.clear_cache()
        universe_fetcher.fetch_sp500()

        assert mock_get.call_count == 2

    @patch("src.data.universe.requests.get")
    def test_missing_table_raises(self, mock_get, universe_fetcher):
        """Test that missing table raises ValueError."""
        mock_response = MagicMock()
        mock_response.text = "<html><body>No table here</body></html>"
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        with pytest.raises(ValueError, match="table not found"):
            universe_fetcher.fetch_sp500()
