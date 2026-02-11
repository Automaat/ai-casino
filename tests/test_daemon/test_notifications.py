"""Tests for notification system."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

from src.daemon.config import NotificationsConfig, NotificationTrigger, TelegramNotificationConfig
from src.daemon.notifications import NotificationMessage, NotificationRateLimiter, NotificationService


class TestNotificationRateLimiter:
    """Tests for NotificationRateLimiter."""

    def test_can_notify_first_time(self) -> None:
        """First notification always allowed."""
        limiter = NotificationRateLimiter(limit_minutes=60)
        assert limiter.can_notify("AAPL", NotificationTrigger.SIGNAL) is True

    def test_rate_limiting_enforced(self) -> None:
        """Notifications rate-limited within window."""
        limiter = NotificationRateLimiter(limit_minutes=60)

        limiter.record_notification("AAPL", NotificationTrigger.SIGNAL)
        assert limiter.can_notify("AAPL", NotificationTrigger.SIGNAL) is False

    def test_critical_triggers_bypass_rate_limit(self) -> None:
        """VaR breach and health failures always allowed."""
        limiter = NotificationRateLimiter(limit_minutes=60)

        assert limiter.can_notify("PORTFOLIO", NotificationTrigger.PORTFOLIO_VAR_BREACH) is True
        limiter.record_notification("PORTFOLIO", NotificationTrigger.PORTFOLIO_VAR_BREACH)
        assert limiter.can_notify("PORTFOLIO", NotificationTrigger.PORTFOLIO_VAR_BREACH) is True

        assert limiter.can_notify("SYSTEM", NotificationTrigger.HEALTH_FAILURE) is True


class TestNotificationService:
    """Tests for NotificationService."""

    async def test_notify_disabled(self) -> None:
        """No notifications sent when disabled."""
        config = NotificationsConfig(enabled=False)
        service = NotificationService(config)

        message = NotificationMessage(
            trigger=NotificationTrigger.SIGNAL,
            title="Test",
            body="Test body",
            metadata={"symbol": "AAPL"},
            timestamp=datetime.now(UTC),
        )

        await service.notify(NotificationTrigger.SIGNAL, message)
        assert len(service.channels) == 0

    async def test_notify_not_in_notify_on(self) -> None:
        """No notifications sent for disabled triggers."""
        config = NotificationsConfig(
            enabled=True,
            notify_on=[NotificationTrigger.SIGNAL],
        )
        service = NotificationService(config)

        message = NotificationMessage(
            trigger=NotificationTrigger.HEALTH_FAILURE,
            title="Test",
            body="Test body",
            metadata={"symbol": "SYSTEM"},
            timestamp=datetime.now(UTC),
        )

        with patch.object(service, "_send_to_channel", new_callable=AsyncMock) as mock_send:
            await service.notify(NotificationTrigger.HEALTH_FAILURE, message)
            mock_send.assert_not_called()

    async def test_rate_limiting_applied(self) -> None:
        """Rate limiting prevents duplicate notifications."""
        config = NotificationsConfig(
            enabled=True,
            rate_limit_enabled=True,
            rate_limit_per_symbol_minutes=60,
            telegram=TelegramNotificationConfig(bot_token="test", chat_id="test"),
        )
        service = NotificationService(config)

        message = NotificationMessage(
            trigger=NotificationTrigger.SIGNAL,
            title="BUY AAPL",
            body="Test",
            metadata={"symbol": "AAPL"},
            timestamp=datetime.now(UTC),
        )

        with patch.object(service.channels["telegram"], "send", return_value=True) as mock_send:
            await service.notify(NotificationTrigger.SIGNAL, message)
            assert mock_send.call_count == 1

            await service.notify(NotificationTrigger.SIGNAL, message)
            assert mock_send.call_count == 1  # Still 1, rate-limited


class TestTelegramChannel:
    """Tests for TelegramChannel."""

    def test_is_configured_with_credentials(self) -> None:
        """Channel configured when bot_token and chat_id present."""
        from src.daemon.notification_channels import TelegramChannel

        config = TelegramNotificationConfig(bot_token="test_token", chat_id="test_chat")
        channel = TelegramChannel(config)
        assert channel.is_configured() is True

    def test_is_configured_missing_credentials(self) -> None:
        """Channel not configured when credentials missing."""
        from src.daemon.notification_channels import TelegramChannel

        config = TelegramNotificationConfig(bot_token=None, chat_id=None)
        channel = TelegramChannel(config)
        assert channel.is_configured() is False

    async def test_send_success(self) -> None:
        """Telegram send succeeds with valid response."""
        from src.daemon.notification_channels import TelegramChannel

        config = TelegramNotificationConfig(bot_token="test_token", chat_id="test_chat")
        channel = TelegramChannel(config)

        message = NotificationMessage(
            trigger=NotificationTrigger.SIGNAL,
            title="BUY AAPL",
            body="Test",
            metadata={
                "symbol": "AAPL",
                "signal": "BUY",
                "confidence": 0.85,
                "price": 150.0,
                "risk_level": "LOW",
                "rsi": 45.3,
                "macd": 0.25,
                "reasoning": "Test",
                "session": "REGULAR",
            },
            timestamp=datetime.now(UTC),
        )

        with patch("src.daemon.notification_channels.httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"ok": True, "result": {"message_id": 123}}
            mock_client.return_value.__aenter__.return_value.post.return_value = mock_response

            success = await channel.send(message)
            assert success is True


class TestNotificationFormatter:
    """Tests for NotificationFormatter."""

    def test_format_signal(self) -> None:
        """Signal notification formatted correctly."""
        from src.daemon.notification_formatter import NotificationFormatter

        message = NotificationMessage(
            trigger=NotificationTrigger.SIGNAL,
            title="BUY AAPL",
            body="Test",
            metadata={
                "symbol": "AAPL",
                "signal": "BUY",
                "confidence": 0.85,
                "price": 150.0,
                "risk_level": "LOW",
                "rsi": 45.3,
                "macd": 0.25,
                "reasoning": "Strong momentum",
                "session": "REGULAR",
            },
            timestamp=datetime.now(UTC),
        )

        result = NotificationFormatter.format_for_telegram(message)
        assert "AAPL" in result
        assert "🟢" in result
        assert "85.0%" in result
        assert "LOW" in result

    def test_format_risk_rejection(self) -> None:
        """Risk rejection notification formatted correctly."""
        from src.daemon.notification_formatter import NotificationFormatter

        message = NotificationMessage(
            trigger=NotificationTrigger.RISK_REJECTION,
            title="Trade Blocked: AAPL",
            body="Exceeds risk limit",
            metadata={
                "symbol": "AAPL",
                "signal": "BUY",
                "price": 150.0,
                "confidence": 0.85,
                "rejection_reason": "Exceeds max position size",
                "risk_score": 0.95,
            },
            timestamp=datetime.now(UTC),
        )

        result = NotificationFormatter.format_for_telegram(message)
        assert "⛔" in result
        assert "AAPL" in result
        assert "Exceeds max position size" in result

    def test_format_var_breach(self) -> None:
        """VaR breach notification formatted correctly."""
        from src.daemon.notification_formatter import NotificationFormatter

        message = NotificationMessage(
            trigger=NotificationTrigger.PORTFOLIO_VAR_BREACH,
            title="Portfolio VaR Limit Breached",
            body="VaR limit exceeded",
            metadata={
                "symbol": "PORTFOLIO",
                "var_95": 0.035,
                "cvar_99": 0.055,
                "var_breached": True,
                "cvar_breached": False,
                "num_positions": 5,
            },
            timestamp=datetime.now(UTC),
        )

        result = NotificationFormatter.format_for_telegram(message)
        assert "⚠️" in result
        assert "3.5%" in result
        assert "5.5%" in result

    def test_format_health_failure(self) -> None:
        """Health failure notification formatted correctly."""
        from src.daemon.notification_formatter import NotificationFormatter

        message = NotificationMessage(
            trigger=NotificationTrigger.HEALTH_FAILURE,
            title="API Health Check Failed",
            body="Services down",
            metadata={
                "symbol": "SYSTEM",
                "failed_services": ["alpha_vantage", "marketaux"],
                "error_messages": ["Timeout", "Rate limit"],
            },
            timestamp=datetime.now(UTC),
        )

        result = NotificationFormatter.format_for_telegram(message)
        assert "⚠️" in result
        assert "alpha_vantage" in result
        assert "marketaux" in result

    def test_format_health_failure_degradation(self) -> None:
        """Health failure notification formatted correctly for degradation (unavailable_services)."""
        from src.daemon.notification_formatter import NotificationFormatter

        message = NotificationMessage(
            trigger=NotificationTrigger.HEALTH_FAILURE,
            title="Trading System DEGRADED",
            body="APIs down: alpha_vantage, marketaux",
            metadata={
                "tier": "DEGRADED",
                "unavailable_services": ["alpha_vantage", "marketaux"],
                "confidence_adjustment": -0.2,
            },
            timestamp=datetime.now(UTC),
        )

        result = NotificationFormatter.format_for_telegram(message)
        assert "⚠️" in result
        assert "alpha_vantage" in result
        assert "marketaux" in result
