"""Notification message formatting for different channels."""

from src.daemon.config import NotificationTrigger
from src.daemon.notifications import NotificationMessage


class NotificationFormatter:
    """Format notification messages for different channels."""

    def __repr__(self) -> str:
        """Return string representation."""
        return "NotificationFormatter()"

    @staticmethod
    def _escape_markdown(text: str) -> str:
        """Escape markdown special characters for Telegram Markdown (legacy) mode.

        Args:
            text: Raw text with potential markdown chars

        Returns:
            Escaped text safe for Telegram Markdown
        """
        # For Markdown (not MarkdownV2), only escape these chars:
        # Periods don't need escaping in legacy Markdown mode
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
        # Dispatch to specialized formatters
        formatters = {
            NotificationTrigger.SIGNAL: NotificationFormatter._format_signal,
            NotificationTrigger.RISK_REJECTION: NotificationFormatter._format_risk_rejection,
            NotificationTrigger.PORTFOLIO_VAR_BREACH: NotificationFormatter._format_var_breach,
            NotificationTrigger.HEALTH_FAILURE: NotificationFormatter._format_health_failure,
            NotificationTrigger.PAPER_TRADING_READY: NotificationFormatter._format_paper_trading_ready,
            NotificationTrigger.AGENT_ALERT: NotificationFormatter._format_agent_alert,
        }

        formatter = formatters.get(message.trigger)
        if formatter:
            return formatter(message)

        # Fallback for unknown triggers
        return f"*{message.title}*\n\n{message.body}"

    @staticmethod
    def _format_agent_alert(message: NotificationMessage) -> str:
        """Format agent alert notification.

        Args:
            message: Notification message

        Returns:
            Formatted markdown string
        """
        # Agent-provided content may contain markdown chars - escape for safe delivery
        title_escaped = NotificationFormatter._escape_markdown(message.title)
        body_escaped = NotificationFormatter._escape_markdown(message.body)
        return f"*{title_escaped}*\n\n{body_escaped}"

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

        reasoning_obj = m.get("reasoning", "No reasoning provided")
        reasoning = NotificationFormatter._escape_markdown(str(reasoning_obj))
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

        rejection_reason = NotificationFormatter._escape_markdown(str(m["rejection_reason"]))
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
        # Handle both "failed_services" (health check) and "unavailable_services" (degradation)
        services_obj = m.get("failed_services") or m.get("unavailable_services") or []
        services_raw = (
            ", ".join(str(s) for s in services_obj) if isinstance(services_obj, list) else str(services_obj)
        )
        # Escape markdown special chars (e.g., underscores in "llm_anthropic")
        services = NotificationFormatter._escape_markdown(services_raw)
        # Use message.title to reflect originating context (health check vs degradation)
        title = NotificationFormatter._escape_markdown(message.title)
        return (
            f"⚠️ *{title}*\n\n"
            f"*Services Down:* {services}\n\n"
            f"Analysis quality may be affected. Check health report."
        )

    @staticmethod
    def _format_paper_trading_ready(message: NotificationMessage) -> str:
        """Format paper trading readiness notification.

        Args:
            message: Notification message

        Returns:
            Formatted markdown string
        """
        m = message.metadata
        return (
            f"🎉 *Paper Trading Validation Complete*\n\n"
            f"*Duration:* {m['duration_days']} days\n"
            f"*Trades:* {m['total_trades']}\n"
            f"*Sharpe:* {m['sharpe']:.2f}\n"
            f"*Max DD:* {m['max_dd']:.1f}%\n\n"
            f"✅ Ready for live trading promotion\n"
            f"Run: `ai-casino validate-paper-trading` to review"
        )
