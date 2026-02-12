"""Tests for FundamentalDataFetcher."""

from unittest.mock import MagicMock, patch

import pytest

from src.data.fundamental import FundamentalDataFetcher


class TestFundamentalDataFetcher:
    """Tests for FundamentalDataFetcher."""

    def test_initialization_with_api_key(self):
        """Test initialization with explicit API key."""
        fetcher = FundamentalDataFetcher(api_key="test_key")

        assert fetcher.api_key == "test_key"

    def test_initialization_missing_api_key(self):
        """Test initialization succeeds without API key but fd is None."""
        fetcher = FundamentalDataFetcher()

        assert fetcher.api_key is None
        assert fetcher.fd is None

    def test_fetch_overview_without_api_key_raises(self):
        """Test fetch raises when API key not configured."""
        fetcher = FundamentalDataFetcher()

        with pytest.raises(ValueError, match="ALPHA_VANTAGE_API_KEY not configured"):
            fetcher.fetch_overview("AAPL")

    @patch("src.data.fundamental.FundamentalData")
    def test_fetch_overview_success(self, mock_fd_class, sample_fundamental_overview):
        """Test successful overview fetch."""
        mock_fd = MagicMock()
        mock_fd.get_company_overview.return_value = (sample_fundamental_overview, None)
        mock_fd_class.return_value = mock_fd

        fetcher = FundamentalDataFetcher(api_key="test_key")
        result = fetcher.fetch_overview("AAPL")

        assert result == sample_fundamental_overview
        assert result["Symbol"] == "AAPL"
        mock_fd.get_company_overview.assert_called_once_with("AAPL")

    @patch("src.data.fundamental.FundamentalData")
    def test_fetch_overview_empty_data(self, mock_fd_class):
        """Test fetch raises when no data available."""
        mock_fd = MagicMock()
        mock_fd.get_company_overview.return_value = ({}, None)
        mock_fd_class.return_value = mock_fd

        fetcher = FundamentalDataFetcher(api_key="test_key")

        with pytest.raises(ValueError, match="No fundamental data available for INVALID"):
            fetcher.fetch_overview("INVALID")

    @patch("src.data.fundamental.FundamentalData")
    def test_fetch_overview_no_symbol(self, mock_fd_class):
        """Test fetch raises when Symbol field missing."""
        mock_fd = MagicMock()
        mock_fd.get_company_overview.return_value = ({"Name": "Company"}, None)
        mock_fd_class.return_value = mock_fd

        fetcher = FundamentalDataFetcher(api_key="test_key")

        with pytest.raises(ValueError, match="No fundamental data available"):
            fetcher.fetch_overview("TEST")

    @patch("src.data.fundamental.FundamentalData")
    def test_fetch_overview_api_error(self, mock_fd_class):
        """Test fetch raises on API error."""
        mock_fd = MagicMock()
        mock_fd.get_company_overview.side_effect = Exception("API error")
        mock_fd_class.return_value = mock_fd

        fetcher = FundamentalDataFetcher(api_key="test_key")

        with pytest.raises(Exception, match="API error"):
            fetcher.fetch_overview("AAPL")

    def test_repr(self):
        """Test string representation."""
        fetcher = FundamentalDataFetcher(api_key="test_key")

        repr_str = repr(fetcher)

        assert "FundamentalDataFetcher" in repr_str
        assert "api_key=***" in repr_str

    def test_repr_no_api_key(self):
        """Test repr when no API key."""
        fetcher = FundamentalDataFetcher()

        repr_str = repr(fetcher)

        assert "api_key=None" in repr_str
