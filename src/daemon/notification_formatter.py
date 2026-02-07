"""Notification message formatting for different channels."""

from src.daemon.config import NotificationTrigger
from src.daemon.notifications import NotificationMessage


class NotificationFormatter:
    """Format notification messages for different channels."""

    @staticmethod
    def _escape_markdown(text: str) -> str:
        """Escape markdown special characters.

        Args:
            text: Raw text with potential markdown chars

        Returns:
            Escaped text safe for Telegram Markdown
        """
        special_chars = [
            "_",
            "*",
            "[",
            "]",
            "(",
            ")",
            "~",
            "`",
            ">",
            "#",
            "+",
            "-",
            "=",
            "|",
            "{",
            "}",
            ".",
            "!",
        ]
        for char in special_chars:
            text = text.replace(char, f"\\{char}")
        return text

    @staticmethod
    def format_for_telegram(message: NotificationMessage) -> str:
        """Format message for Telegram.

        Args:
            message: Notification message

        Returns:
            Formatted markdown string
        """
        if message.trigger == NotificationTrigger.SIGNAL:
            return NotificationFormatter._format_signal(message)
        if message.trigger == NotificationTrigger.RISK_REJECTION:
            return NotificationFormatter._format_risk_rejection(message)
        if message.trigger == NotificationTrigger.PORTFOLIO_VAR_BREACH:
            return NotificationFormatter._format_var_breach(message)
        if message.trigger == NotificationTrigger.HEALTH_FAILURE:
            return NotificationFormatter._format_health_failure(message)
        return f"*{message.title}*\n\n{message.body}"

    @staticmethod
    def _format_signal(message: NotificationMessage) -> str:
        """Format trading signal notification.

        Args:
            message: Notification message

        Returns:
            Formatted markdown string
        """
        m = message.metadata
        signal_emoji = "🟢" if m["signal"] == "BUY" else "🔴"
        session_tag = " _(pre-market)_" if m.get("session") == "PRE_MARKET" else ""

        rsi_value = (
            f"{m.get('rsi'):.1f}" if isinstance(m.get("rsi"), (int, float)) else str(m.get("rsi", "N/A"))
        )
        macd_value = (
            f"{m.get('macd'):.2f}" if isinstance(m.get("macd"), (int, float)) else str(m.get("macd", "N/A"))
        )

        reasoning = NotificationFormatter._escape_markdown(m.get("reasoning", "No reasoning provided"))
        return (
            f"{signal_emoji} *{m['signal']} {m['symbol']}* at ${m['price']:.2f}{session_tag}\n\n"
            f"*Confidence:* {m['confidence']:.1%} | *Risk:* {m['risk_level']}\n"
            f"*RSI:* {rsi_value} | *MACD:* {macd_value}\n\n"
            f"_{reasoning}_"
        )

    @staticmethod
    def _format_risk_rejection(message: NotificationMessage) -> str:
        """Format risk rejection notification.

        Args:
            message: Notification message

        Returns:
            Formatted markdown string
        """
        m = message.metadata
        risk_score_str = (
            f"{m.get('risk_score'):.2f}" if isinstance(m.get("risk_score"), (int, float)) else "N/A"
        )

        rejection_reason = NotificationFormatter._escape_markdown(m["rejection_reason"])
        return (
            f"⛔ *Trade Blocked: {m['symbol']}*\n\n"
            f"*Action:* {m['signal']} at ${m['price']:.2f}\n"
            f"*Reason:* {rejection_reason}\n"
            f"*Confidence:* {m['confidence']:.1%}\n"
            f"*Risk Score:* {risk_score_str}"
        )

    @staticmethod
    def _format_var_breach(message: NotificationMessage) -> str:
        """Format VaR breach notification.

        Args:
            message: Notification message

        Returns:
            Formatted markdown string
        """
        m = message.metadata
        return (
            f"⚠️ *Portfolio VaR Limit Breached*\n\n"
            f"*VaR95:* {m['var_95']:.1%}\n"
            f"*CVaR99:* {m['cvar_99']:.1%}\n"
            f"*Positions:* {m['num_positions']}\n\n"
            f"Portfolio risk exceeds configured limits. Review risk report."
        )

    @staticmethod
    def _format_health_failure(message: NotificationMessage) -> str:
        """Format health failure notification.

        Args:
            message: Notification message

        Returns:
            Formatted markdown string
        """
        m = message.metadata
        services = ", ".join(m["failed_services"])
        return (
            f"⚠️ *API Health Check Failed*\n\n"
            f"*Services Down:* {services}\n\n"
            f"Analysis quality may be affected. Check health report."
        )
