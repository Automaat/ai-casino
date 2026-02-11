"""DEPRECATED: Import from src.daemon.config package instead.

This file exists for backward compatibility only.
All config classes have been moved to the src.daemon.config package.

Usage:
    # New (recommended)
    from src.daemon.config import DaemonConfig, ScheduleConfig

    # Old (deprecated, but still works)
    from src.daemon.config import DaemonConfig, ScheduleConfig
"""

import warnings

from src.daemon.config import *  # noqa: F403

warnings.warn(
    "Importing from src.daemon.config.py is deprecated. "
    "Import from src.daemon.config package instead. "
    "This shim will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)
