"""Tests for notifications package."""

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from src.daemon.config.notifications import NotificationsConfig, TelegramNotificationConfig
from src.notifications.models import NotificationMessage, NotificationSeverity
from src.notifications.service import NotificationService


@pytest.mark.unit
class TestNotificationSeverity:
    """Tests for NotificationSeverity."""

    def test_from_priority_low(self) -> None:
        """LOW priority maps to INFO."""
        assert NotificationSeverity.from_priority("LOW") == NotificationSeverity.INFO

    def test_from_priority_medium(self) -> None:
        """MEDIUM priority maps to WARNING."""
        assert NotificationSeverity.from_priority("MEDIUM") == NotificationSeverity.WARNING

    def test_from_priority_high(self) -> None:
        """HIGH priority maps to ERROR."""
        assert NotificationSeverity.from_priority("HIGH") == NotificationSeverity.ERROR

    def test_from_priority_critical(self) -> None:
        """CRITICAL priority maps to CRITICAL."""
        assert NotificationSeverity.from_priority("CRITICAL") == NotificationSeverity.CRITICAL

    def test_from_priority_unknown(self) -> None:
        """Unknown priority defaults to INFO."""
        assert NotificationSeverity.from_priority("UNKNOWN") == NotificationSeverity.INFO

    def test_from_priority_case_insensitive(self) -> None:
        """Priority mapping is case-insensitive."""
        assert NotificationSeverity.from_priority("high") == NotificationSeverity.ERROR
        assert NotificationSeverity.from_priority("Low") == NotificationSeverity.INFO


@pytest.mark.unit
class TestNotificationMessage:
    """Tests for NotificationMessage model."""

    def test_default_severity_is_info(self) -> None:
        """Default severity is INFO."""
        msg = NotificationMessage(
            title="Test",
            body="body",
            metadata={},
            timestamp=datetime.now(UTC),
        )
        assert msg.severity == NotificationSeverity.INFO

    def test_metadata_accepts_scalar_types(self) -> None:
        """Metadata accepts str, int, float, bool values."""
        msg = NotificationMessage(
            title="Test",
            body="body",
            metadata={"s": "str", "i": 1, "f": 1.5, "b": True},
            timestamp=datetime.now(UTC),
        )
        assert msg.metadata["s"] == "str"
        assert msg.metadata["i"] == 1
        assert msg.metadata["f"] == 1.5
        assert msg.metadata["b"] is True

    def test_repr(self) -> None:
        """Repr contains title and severity."""
        msg = NotificationMessage(
            title="Alert",
            body="body",
            severity=NotificationSeverity.ERROR,
            metadata={},
            timestamp=datetime.now(UTC),
        )
        assert "Alert" in repr(msg)
        assert "error" in repr(msg)


@pytest.mark.unit
class TestNotificationService:
    """Tests for NotificationService."""

    async def test_notify_disabled(self) -> None:
        """No notifications sent when disabled."""
        config = NotificationsConfig(enabled=False)
        service = NotificationService(config)

        message = NotificationMessage(
            title="Test",
            body="body",
            metadata={},
            timestamp=datetime.now(UTC),
        )

        await service.notify(message)
        assert len(service.channels) == 0

    async def test_notify_no_channels(self) -> None:
        """notify() is no-op when channels empty."""
        config = NotificationsConfig(enabled=True, channels=[])
        service = NotificationService(config)
        service.channels = {}

        message = NotificationMessage(
            title="Test",
            body="body",
            metadata={},
            timestamp=datetime.now(UTC),
        )

        # Should not raise
        await service.notify(message)

    async def test_notify_sends_to_channel(self) -> None:
        """notify() calls send on all channels."""
        config = NotificationsConfig(
            enabled=True,
            telegram=TelegramNotificationConfig(bot_token="tok", chat_id="chat"),
        )
        service = NotificationService(config)

        message = NotificationMessage(
            title="Test",
            body="body",
            severity=NotificationSeverity.WARNING,
            metadata={"symbol": "AAPL"},
            timestamp=datetime.now(UTC),
        )

        with patch.object(service.channels["telegram"], "send", return_value=True) as mock_send:
            await service.notify(message)
            mock_send.assert_called_once_with(message)

    def test_repr(self) -> None:
        """Repr contains enabled and channel names."""
        config = NotificationsConfig(enabled=False)
        service = NotificationService(config)
        r = repr(service)
        assert "NotificationService" in r
        assert "enabled=False" in r


@pytest.mark.unit
class TestTelegramChannel:
    """Tests for TelegramChannel."""

    def test_is_configured_with_credentials(self) -> None:
        """Channel configured when bot_token and chat_id present."""
        from src.notifications.channels.telegram import TelegramChannel

        config = TelegramNotificationConfig(bot_token="test_token", chat_id="test_chat")
        channel = TelegramChannel(config)
        assert channel.is_configured() is True

    def test_is_configured_missing_credentials(self) -> None:
        """Channel not configured when credentials missing."""
        from src.notifications.channels.telegram import TelegramChannel

        config = TelegramNotificationConfig(bot_token=None, chat_id=None)
        channel = TelegramChannel(config)
        assert channel.is_configured() is False

    async def test_send_success(self) -> None:
        """Telegram send succeeds with valid response."""
        from src.notifications.channels.telegram import TelegramChannel

        config = TelegramNotificationConfig(bot_token="test_token", chat_id="test_chat")
        channel = TelegramChannel(config)

        message = NotificationMessage(
            title="BUY AAPL",
            body="Strong momentum",
            severity=NotificationSeverity.WARNING,
            metadata={"symbol": "AAPL", "signal": "BUY", "confidence": 0.85},
            timestamp=datetime.now(UTC),
        )

        with patch("src.notifications.channels.telegram.httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"ok": True, "result": {"message_id": 123}}
            mock_client.return_value.__aenter__.return_value.post.return_value = mock_response

            success = await channel.send(message)
            assert success is True

    async def test_send_failure_api_error(self) -> None:
        """Telegram send returns False on API error."""
        from src.notifications.channels.telegram import TelegramChannel

        config = TelegramNotificationConfig(bot_token="test_token", chat_id="test_chat")
        channel = TelegramChannel(config)

        message = NotificationMessage(
            title="Test",
            body="body",
            metadata={},
            timestamp=datetime.now(UTC),
        )

        with patch("src.notifications.channels.telegram.httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"ok": False, "description": "Bad Request"}
            mock_client.return_value.__aenter__.return_value.post.return_value = mock_response

            success = await channel.send(message)
            assert success is False


@pytest.mark.unit
class TestTelegramFormatter:
    """Tests for Telegram message formatter."""

    def test_format_includes_severity_emoji(self) -> None:
        """Formatted message contains correct severity emoji."""
        from src.notifications.channels.telegram import _format

        msg = NotificationMessage(
            title="Alert",
            body="something happened",
            severity=NotificationSeverity.CRITICAL,
            metadata={},
            timestamp=datetime.now(UTC),
        )
        result = _format(msg)
        assert "🚨" in result
        assert "Alert" in result

    def test_format_warning_emoji(self) -> None:
        """WARNING severity uses warning emoji."""
        from src.notifications.channels.telegram import _format

        msg = NotificationMessage(
            title="Signal",
            body="BUY",
            severity=NotificationSeverity.WARNING,
            metadata={"symbol": "AAPL"},
            timestamp=datetime.now(UTC),
        )
        result = _format(msg)
        assert "⚠️" in result

    def test_format_metadata_as_bullet_list(self) -> None:
        """Metadata rendered as bullet key-value list."""
        from src.notifications.channels.telegram import _format

        msg = NotificationMessage(
            title="Test",
            body="body",
            metadata={"symbol": "AAPL", "confidence": 0.85},
            timestamp=datetime.now(UTC),
        )
        result = _format(msg)
        assert "symbol" in result
        assert "AAPL" in result
        assert "•" in result

    def test_format_escapes_special_chars(self) -> None:
        """Special markdown chars escaped in title and body."""
        from src.notifications.channels.telegram import _format

        msg = NotificationMessage(
            title="Alert_with_underscores",
            body="body *bold* text",
            metadata={},
            timestamp=datetime.now(UTC),
        )
        result = _format(msg)
        assert "\\_" in result
        assert "\\*" in result

    def test_format_error_emoji(self) -> None:
        """ERROR severity uses error emoji."""
        from src.notifications.channels.telegram import _format

        msg = NotificationMessage(
            title="Error",
            body="service down",
            severity=NotificationSeverity.ERROR,
            metadata={},
            timestamp=datetime.now(UTC),
        )
        result = _format(msg)
        assert "❌" in result

    def test_format_info_emoji(self) -> None:
        """INFO severity uses info emoji."""
        from src.notifications.channels.telegram import _format

        msg = NotificationMessage(
            title="Info",
            body="paper trading ready",
            severity=NotificationSeverity.INFO,
            metadata={},
            timestamp=datetime.now(UTC),
        )
        result = _format(msg)
        assert "\U0001f4ac" in result
