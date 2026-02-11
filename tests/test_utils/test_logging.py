"""Tests for logging utilities."""

from src.utils.logging import sanitize_log_record, sanitize_message


class TestSanitizeMessage:
    """Test sanitize_message function."""

    def test_sanitize_token_query_param(self) -> None:
        """Redact ?token= query parameter."""
        message = "Error fetching 'https://finnhub.io/api/v1/news?symbol=AAPL&token=abc123def456'"
        result = sanitize_message(message)
        assert "token=[REDACTED]" in result
        assert "abc123def456" not in result

    def test_sanitize_api_token_query_param(self) -> None:
        """Redact ?api_token= query parameter."""
        message = "Error fetching 'https://api.marketaux.com/v1/news/all?api_token=secret_key_xyz'"
        result = sanitize_message(message)
        assert "api_token=[REDACTED]" in result
        assert "secret_key_xyz" not in result

    def test_sanitize_api_key_query_param(self) -> None:
        """Redact ?api_key= query parameter."""
        message = "GET https://alpha-vantage.com/query?function=TIME_SERIES_DAILY&api_key=DEMO123"
        result = sanitize_message(message)
        assert "api_key=[REDACTED]" in result
        assert "DEMO123" not in result

    def test_sanitize_multiple_params_in_url(self) -> None:
        """Redact multiple sensitive parameters in single URL."""
        message = (
            "Client error '403 Forbidden' for url "
            "'https://finnhub.io/api/v1/stock/social-sentiment?"
            "symbol=AAPL&token=abc123&from=2024-01-01&to=2024-01-31'"
        )
        result = sanitize_message(message)
        assert "token=[REDACTED]" in result
        assert "abc123" not in result
        assert "symbol=AAPL" in result  # Non-sensitive params unchanged
        assert "from=2024-01-01" in result

    def test_sanitize_ampersand_separated_token(self) -> None:
        """Redact &token= (mid-URL parameter)."""
        message = "URL: https://api.example.com/data?symbol=META&token=xyz789&limit=10"
        result = sanitize_message(message)
        assert "token=[REDACTED]" in result
        assert "xyz789" not in result
        assert "symbol=META" in result
        assert "limit=10" in result

    def test_sanitize_case_insensitive(self) -> None:
        """Redact tokens case-insensitively (TOKEN, Token, token)."""
        messages = [
            "URL: https://api.example.com?TOKEN=secret1",
            "URL: https://api.example.com?Token=secret2",
            "URL: https://api.example.com?token=secret3",
        ]
        for msg in messages:
            result = sanitize_message(msg)
            assert "[REDACTED]" in result
            assert "secret" not in result

    def test_sanitize_password_query_param(self) -> None:
        """Redact ?password= query parameter."""
        message = "Auth failed: https://example.com/login?user=admin&password=pass123"
        result = sanitize_message(message)
        assert "password=[REDACTED]" in result
        assert "pass123" not in result

    def test_sanitize_secret_query_param(self) -> None:
        """Redact ?secret= query parameter."""
        message = "Request: https://example.com/api?id=1&secret=my_secret_key"
        result = sanitize_message(message)
        assert "secret=[REDACTED]" in result
        assert "my_secret_key" not in result

    def test_sanitize_auth_query_param(self) -> None:
        """Redact ?auth= query parameter."""
        message = "GET https://example.com/protected?auth=bearer_token_xyz"
        result = sanitize_message(message)
        assert "auth=[REDACTED]" in result
        assert "bearer_token_xyz" not in result

    def test_no_sanitization_when_no_sensitive_params(self) -> None:
        """Leave message unchanged if no sensitive params."""
        message = "Error fetching https://example.com/api?symbol=AAPL&limit=100"
        result = sanitize_message(message)
        assert result == message

    def test_sanitize_httpx_exception_message(self) -> None:
        """Redact tokens in realistic httpx exception messages."""
        message = (
            "Client error '403 Forbidden' for url "
            "'https://finnhub.io/api/v1/news-sentiment?symbol=META&token=d64c0f1r01qkcggr0jvg'\n"
            "For more information check: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/403"
        )
        result = sanitize_message(message)
        assert "token=[REDACTED]" in result
        assert "d64c0f1r01qkcggr0jvg" not in result
        assert "symbol=META" in result
        assert "For more information" in result  # Rest of message intact

    def test_sanitize_multiple_urls_in_message(self) -> None:
        """Redact tokens in multiple URLs within same message."""
        message = (
            "Retrying after error for https://api1.com?token=abc123 "
            "and fallback to https://api2.com?api_key=xyz789"
        )
        result = sanitize_message(message)
        assert "token=[REDACTED]" in result
        assert "api_key=[REDACTED]" in result
        assert "abc123" not in result
        assert "xyz789" not in result

    def test_sanitize_preserves_url_structure(self) -> None:
        """Ensure URL structure remains valid after sanitization."""
        message = "https://finnhub.io/api/v1/news?symbol=AAPL&token=secret&from=2024-01-01"
        result = sanitize_message(message)
        assert result == "https://finnhub.io/api/v1/news?symbol=AAPL&token=[REDACTED]&from=2024-01-01"

    def test_sanitize_empty_string(self) -> None:
        """Handle empty string gracefully."""
        result = sanitize_message("")
        assert result == ""

    def test_sanitize_no_urls(self) -> None:
        """Handle messages without URLs."""
        message = "This is a plain log message without any URLs"
        result = sanitize_message(message)
        assert result == message


class TestSanitizeLogRecord:
    """Test sanitize_log_record filter function."""

    def test_sanitize_log_record_modifies_message(self) -> None:
        """Filter modifies message field in record."""
        record = {
            "message": "Error: https://api.example.com?token=secret123",
            "level": {"name": "ERROR"},
        }
        result = sanitize_log_record(record)
        assert result is True  # Always returns True to allow logging
        assert "token=[REDACTED]" in record["message"]
        assert "secret123" not in record["message"]

    def test_sanitize_log_record_returns_true(self) -> None:
        """Filter always returns True (allows all logs)."""
        record = {"message": "Normal log message"}
        result = sanitize_log_record(record)
        assert result is True

    def test_sanitize_log_record_handles_missing_message(self) -> None:
        """Handle record without message field gracefully."""
        record = {"level": {"name": "INFO"}}
        result = sanitize_log_record(record)
        assert result is True
        assert "message" not in record

    def test_sanitize_log_record_preserves_other_fields(self) -> None:
        """Ensure other record fields remain untouched."""
        record = {
            "message": "URL: https://api.example.com?api_key=secret",
            "level": {"name": "ERROR"},
            "extra": {"custom_field": "value"},
            "time": "2024-01-01T00:00:00",
        }
        sanitize_log_record(record)
        assert record["level"]["name"] == "ERROR"
        assert record["extra"]["custom_field"] == "value"
        assert record["time"] == "2024-01-01T00:00:00"
        assert "api_key=[REDACTED]" in record["message"]
