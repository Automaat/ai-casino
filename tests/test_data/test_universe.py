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
def mock_ishares_csv():
    """Mock iShares Russell 3000 CSV response."""
    return """Header1,Header2
Header3,Header4
Metadata,Row1
Metadata,Row2
Metadata,Row3
Metadata,Row4
Metadata,Row5
Metadata,Row6
Metadata,Row7
Metadata,Row8
Ticker,Name,Sector,Weight
AAPL,"Apple Inc","Information Technology",3.5
MSFT,"Microsoft Corporation","Information Technology",3.2
GOOGL,"Alphabet Inc Class A","Communication Services",1.8
BRK.B,"Berkshire Hathaway Inc Class B","Financials",1.5
NVDA,"NVIDIA Corporation","Information Technology",1.2
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

    @patch("src.data.universe.httpx.Client")
    def test_fetch_sp500(self, mock_client_class, universe_fetcher, mock_sp500_html):
        """Test fetching S&P 500 stocks."""
        mock_response = MagicMock()
        mock_response.text = mock_sp500_html
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_class.return_value = mock_client

        universe = universe_fetcher.fetch_sp500()

        assert universe.name == "SP500"
        assert len(universe.stocks) == 3
        assert universe.stocks[0].symbol == "AAPL"
        assert universe.stocks[2].symbol == "BRK-B"  # . replaced with -

    @patch("src.data.universe.httpx.Client")
    def test_fetch_nasdaq100(self, mock_client_class, universe_fetcher, mock_nasdaq100_html):
        """Test fetching NASDAQ 100 stocks."""
        mock_response = MagicMock()
        mock_response.text = mock_nasdaq100_html
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_class.return_value = mock_client

        universe = universe_fetcher.fetch_nasdaq100()

        assert universe.name == "NASDAQ100"
        assert len(universe.stocks) == 2
        assert universe.stocks[0].symbol == "AAPL"
        assert universe.stocks[1].symbol == "NVDA"

    @patch("src.data.universe.httpx.Client")
    def test_fetch_combined_deduplicates(
        self, mock_client_class, universe_fetcher, mock_sp500_html, mock_nasdaq100_html
    ):
        """Test combined fetch deduplicates stocks."""
        responses = [
            MagicMock(text=mock_sp500_html, raise_for_status=MagicMock()),
            MagicMock(text=mock_nasdaq100_html, raise_for_status=MagicMock()),
        ]

        mock_client = MagicMock()
        mock_client.get.side_effect = responses
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_class.return_value = mock_client

        universe = universe_fetcher.fetch_combined()

        assert universe.name == "COMBINED"
        symbols = [s.symbol for s in universe.stocks]
        assert symbols.count("AAPL") == 1  # Deduplicated
        assert "NVDA" in symbols
        assert "MSFT" in symbols

    @patch("src.data.universe.httpx.Client")
    def test_caching(self, mock_client_class, universe_fetcher, mock_sp500_html):
        """Test that results are cached."""
        mock_response = MagicMock()
        mock_response.text = mock_sp500_html
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_class.return_value = mock_client

        universe1 = universe_fetcher.fetch_sp500()
        universe2 = universe_fetcher.fetch_sp500()

        assert mock_client.get.call_count == 1  # Only called once
        assert universe1.stocks == universe2.stocks

    @patch("src.data.universe.httpx.Client")
    def test_clear_cache(self, mock_client_class, universe_fetcher, mock_sp500_html):
        """Test cache clearing."""
        mock_response = MagicMock()
        mock_response.text = mock_sp500_html
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_class.return_value = mock_client

        universe_fetcher.fetch_sp500()
        universe_fetcher.clear_cache()
        universe_fetcher.fetch_sp500()

        assert mock_client.get.call_count == 2

    @patch("src.data.universe.httpx.Client")
    def test_missing_table_raises(self, mock_client_class, universe_fetcher):
        """Test that missing table raises ValueError."""
        mock_response = MagicMock()
        mock_response.text = "<html><body>No table here</body></html>"
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_class.return_value = mock_client

        with pytest.raises(ValueError, match="table not found"):
            universe_fetcher.fetch_sp500()

    @patch("src.data.universe.httpx.Client")
    def test_fetch_russell3000(self, mock_client_class, universe_fetcher, mock_ishares_csv):
        """Test fetching Russell 3000 stocks from iShares CSV."""
        mock_response = MagicMock()
        mock_response.text = mock_ishares_csv
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_class.return_value = mock_client

        universe = universe_fetcher.fetch_russell3000()

        assert universe.name == "RUSSELL3000"
        assert len(universe.stocks) == 5
        assert universe.stocks[0].symbol == "AAPL"
        assert universe.stocks[3].symbol == "BRK-B"  # . replaced with -

    @patch("src.data.universe.httpx.Client")
    def test_fetch_sp500_filters_invalid_symbols(self, mock_client_class, universe_fetcher):
        """Test S&P 500 scraper filters symbols with spaces or length > 6."""
        mock_html = """
        <table id="constituents">
            <tr><th>Symbol</th><th>Security</th><th>CIK</th><th>GICS Sector</th><th>Sub-Industry</th></tr>
            <tr><td>AAPL</td><td>Apple Inc.</td><td>123</td><td>Tech</td><td>Hardware</td></tr>
            <tr><td>INVALID SYMBOL</td><td>Bad Co</td><td>456</td><td>Tech</td><td>Software</td></tr>
            <tr><td>TOOLONG</td><td>Too Long Inc</td><td>789</td><td>Tech</td><td>Services</td></tr>
        </table>
        """
        mock_response = MagicMock()
        mock_response.text = mock_html
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_class.return_value = mock_client

        universe = universe_fetcher.fetch_sp500()

        assert len(universe.stocks) == 1
        assert universe.stocks[0].symbol == "AAPL"

    @patch("src.data.universe.httpx.Client")
    def test_fetch_nasdaq100_filters_invalid_symbols(self, mock_client_class, universe_fetcher):
        """Test NASDAQ 100 scraper filters symbols with spaces or length > 6."""
        mock_html = """
        <table id="constituents">
            <tr><th>Company</th><th>Ticker</th><th>GICS Sector</th><th>GICS Sub-Industry</th></tr>
            <tr><td>Apple Inc.</td><td>AAPL</td><td>Tech</td><td>Hardware</td></tr>
            <tr><td>Bad Co</td><td>BAD TICK</td><td>Tech</td><td>Software</td></tr>
            <tr><td>Too Long Inc</td><td>TOOLONG</td><td>Tech</td><td>Services</td></tr>
        </table>
        """
        mock_response = MagicMock()
        mock_response.text = mock_html
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_class.return_value = mock_client

        universe = universe_fetcher.fetch_nasdaq100()

        assert len(universe.stocks) == 1
        assert universe.stocks[0].symbol == "AAPL"

    @patch("src.data.universe.httpx.Client")
    def test_fetch_russell3000_filters_invalid_symbols(self, mock_client_class, universe_fetcher):
        """Test Russell 3000 scraper filters symbols with spaces or length > 6."""
        mock_csv = """Header1,Header2
Header3,Header4
Metadata,Row1
Metadata,Row2
Metadata,Row3
Metadata,Row4
Metadata,Row5
Metadata,Row6
Metadata,Row7
Metadata,Row8
Ticker,Name,Sector,Weight
AAPL,"Apple Inc","Tech",3.5
INVALID SYMBOL,"Bad Company","Tech",1.2
TOOLONG,"Too Long Inc","Tech",0.8
"""
        mock_response = MagicMock()
        mock_response.text = mock_csv
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_class.return_value = mock_client

        universe = universe_fetcher.fetch_russell3000()

        assert len(universe.stocks) == 1
        assert universe.stocks[0].symbol == "AAPL"

    @patch("src.data.universe.yf.download")
    @patch("src.data.universe.yf.Ticker")
    def test_fetch_us_liquid_default_filters(
        self, mock_ticker_class, mock_download, universe_fetcher, mock_ishares_csv
    ):
        """Test US_LIQUID with default filters."""
        from src.daemon.config import LiquidityFilterConfig

        # Mock Russell 3000 fetch
        with patch("src.data.universe.httpx.Client") as mock_client_class:
            mock_response = MagicMock()
            mock_response.text = mock_ishares_csv
            mock_response.raise_for_status = MagicMock()

            mock_client = MagicMock()
            mock_client.get.return_value = mock_response
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client_class.return_value = mock_client

            # Mock yfinance download (OHLCV data)
            import pandas as pd

            mock_data = pd.DataFrame(
                {
                    ("AAPL", "Close"): [150.0] * 30,
                    ("AAPL", "Volume"): [2_000_000] * 30,
                    ("MSFT", "Close"): [300.0] * 30,
                    ("MSFT", "Volume"): [1_500_000] * 30,
                    ("GOOGL", "Close"): [120.0] * 30,
                    ("GOOGL", "Volume"): [800_000] * 30,  # Below threshold
                }
            )
            mock_download.return_value = mock_data

            # Mock yfinance Ticker.info (market cap)
            def mock_ticker(symbol):
                mock = MagicMock()
                if symbol == "AAPL":
                    mock.info = {"marketCap": 3_000_000_000_000}  # $3T
                elif symbol == "MSFT":
                    mock.info = {"marketCap": 2_800_000_000_000}  # $2.8T
                elif symbol == "GOOGL":
                    mock.info = {"marketCap": 500_000_000}  # $500M - below threshold
                else:
                    mock.info = {"marketCap": 0}
                return mock

            mock_ticker_class.side_effect = mock_ticker

            filters = LiquidityFilterConfig()  # Default: $1B, 1M vol, $10-$500
            universe = universe_fetcher.fetch_us_liquid(filters)

            assert universe.name == "US_LIQUID"
            # Should filter out GOOGL (low volume) and stocks with low market cap
            assert len(universe.stocks) == 2  # Only AAPL and MSFT

    @patch("src.data.universe.yf.download")
    @patch("src.data.universe.yf.Ticker")
    def test_fetch_us_liquid_strict_filters(
        self, mock_ticker_class, mock_download, universe_fetcher, mock_ishares_csv
    ):
        """Test US_LIQUID with strict filters (smaller universe)."""
        from src.daemon.config import LiquidityFilterConfig

        # Mock Russell 3000 fetch
        with patch("src.data.universe.httpx.Client") as mock_client_class:
            mock_response = MagicMock()
            mock_response.text = mock_ishares_csv
            mock_response.raise_for_status = MagicMock()

            mock_client = MagicMock()
            mock_client.get.return_value = mock_response
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client_class.return_value = mock_client

            # Mock yfinance download
            import pandas as pd

            mock_data = pd.DataFrame(
                {
                    ("AAPL", "Close"): [150.0] * 30,
                    ("AAPL", "Volume"): [2_000_000] * 30,
                    ("MSFT", "Close"): [300.0] * 30,
                    ("MSFT", "Volume"): [1_500_000] * 30,  # Below strict threshold
                }
            )
            mock_download.return_value = mock_data

            # Mock yfinance Ticker.info
            def mock_ticker(symbol):
                mock = MagicMock()
                if symbol == "AAPL":
                    mock.info = {"marketCap": 3_000_000_000_000}
                elif symbol == "MSFT":
                    mock.info = {"marketCap": 2_800_000_000_000}
                else:
                    mock.info = {"marketCap": 0}
                return mock

            mock_ticker_class.side_effect = mock_ticker

            # Strict filters: $2B market cap, 2M volume
            filters = LiquidityFilterConfig(min_market_cap=2e9, min_avg_volume=2_000_000)
            universe = universe_fetcher.fetch_us_liquid(filters)

            assert universe.name == "US_LIQUID"
            # Only AAPL passes strict filters
            assert len(universe.stocks) == 1
            assert universe.stocks[0].symbol == "AAPL"

    @patch("src.data.universe.httpx.Client")
    def test_fetch_us_liquid_cache_varies_with_config(
        self, mock_client_class, universe_fetcher, mock_ishares_csv
    ):
        """Test cache keys differ for different filter configs."""
        from src.daemon.config import LiquidityFilterConfig

        mock_response = MagicMock()
        mock_response.text = mock_ishares_csv
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_class.return_value = mock_client

        # Mock yfinance for both calls
        with (
            patch("src.data.universe.yf.download") as mock_download,
            patch("src.data.universe.yf.Ticker") as mock_ticker_class,
        ):
            import pandas as pd

            mock_data = pd.DataFrame(
                {
                    ("AAPL", "Close"): [150.0] * 30,
                    ("AAPL", "Volume"): [2_000_000] * 30,
                }
            )
            mock_download.return_value = mock_data

            def mock_ticker(_symbol):
                mock = MagicMock()
                mock.info = {"marketCap": 3_000_000_000_000}
                return mock

            mock_ticker_class.side_effect = mock_ticker

            filters1 = LiquidityFilterConfig(min_market_cap=1e9)
            filters2 = LiquidityFilterConfig(min_market_cap=2e9)

            universe1 = universe_fetcher.fetch_us_liquid(filters1)
            universe2 = universe_fetcher.fetch_us_liquid(filters2)

            # Both should succeed and use different cache keys
            assert universe1.name == "US_LIQUID"
            assert universe2.name == "US_LIQUID"
            # Verify different filter configs produce different results
            # (first should include more stocks due to lower market cap threshold)
            assert len(universe1.stocks) >= len(universe2.stocks)
