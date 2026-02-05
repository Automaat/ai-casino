"""Shared type aliases for TUI components."""

from collections.abc import Callable

# Progress callback: (step_id, status, detail)
ProgressCallback = Callable[[str, str, str], None]
