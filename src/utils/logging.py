"""Logging utilities for token sanitization."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from loguru import Record


# Patterns for sensitive query parameters in URLs
_SENSITIVE_PARAMS = [
    r"([?&])(token)=([^&\s]+)",  # ?token=XXX or &token=XXX
    r"([?&])(api_token)=([^&\s]+)",  # ?api_token=XXX or &api_token=XXX
    r"([?&])(api_key)=([^&\s]+)",  # ?api_key=XXX or &api_key=XXX
    r"([?&])(apikey)=([^&\s]+)",  # ?apikey=XXX or &apikey=XXX
    r"([?&])(key)=([^&\s]+)",  # ?key=XXX or &key=XXX (be cautious, may be too broad)
    r"([?&])(secret)=([^&\s]+)",  # ?secret=XXX or &secret=XXX
    r"([?&])(password)=([^&\s]+)",  # ?password=XXX or &password=XXX
    r"([?&])(auth)=([^&\s]+)",  # ?auth=XXX or &auth=XXX
]

# Compiled regex patterns for performance
_SANITIZE_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in _SENSITIVE_PARAMS]


def sanitize_message(message: str) -> str:
    """Sanitize sensitive tokens from log message.

    Replaces API keys/tokens in URL query parameters with [REDACTED].

    Args:
        message: Log message potentially containing URLs with tokens

    Returns:
        Sanitized message with tokens redacted
    """
    sanitized = message
    for pattern in _SANITIZE_PATTERNS:
        # Replace: ?token=abc123 -> ?token=[REDACTED]
        # Capture groups: (1) separator (?&), (2) param name, (3) value
        sanitized = pattern.sub(r"\1\2=[REDACTED]", sanitized)
    return sanitized


def sanitize_log_record(record: Record) -> bool:
    """Loguru filter function to sanitize sensitive tokens from log records.

    Applied as filter to logger.add() to redact tokens from all log messages.

    Args:
        record: Loguru record dict with 'message' key

    Returns:
        True to allow logging (always)
    """
    if "message" in record:
        record["message"] = sanitize_message(record["message"])
    return True
