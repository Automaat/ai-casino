"""Nord theme definitions and dark/light detection."""

import os
import subprocess
import sys
from pathlib import Path

from loguru import logger
from textual.theme import Theme

# Nord color palette
NORD = {
    # Polar Night (dark backgrounds)
    "nord0": "#2E3440",
    "nord1": "#3B4252",
    "nord2": "#434C5E",
    "nord3": "#4C566A",
    # Snow Storm (light backgrounds/text)
    "nord4": "#D8DEE9",
    "nord5": "#E5E9F0",
    "nord6": "#ECEFF4",
    # Frost (accent colors)
    "nord7": "#8FBCBB",
    "nord8": "#88C0D0",
    "nord9": "#81A1C1",
    "nord10": "#5E81AC",
    # Aurora (semantic colors)
    "nord11": "#BF616A",  # red/error
    "nord12": "#D08770",  # orange
    "nord13": "#EBCB8B",  # yellow/warning
    "nord14": "#A3BE8C",  # green/success
    "nord15": "#B48EAD",  # purple
}

NORD_LIGHT_THEME = Theme(
    name="nord-light",
    primary=NORD["nord10"],
    secondary=NORD["nord9"],
    accent=NORD["nord8"],
    foreground=NORD["nord0"],
    background=NORD["nord6"],
    success=NORD["nord14"],
    warning=NORD["nord13"],
    error=NORD["nord11"],
    surface=NORD["nord5"],
    panel=NORD["nord4"],
    dark=False,
)


def _is_light_color(hex_color: str) -> bool:
    """Check if hex color is light based on luminance."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return False
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    # Perceived luminance formula
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return luminance > 128


def _detect_ghostty_theme() -> bool | None:
    """Detect theme from Ghostty config. Returns True=dark, False=light, None=unknown."""
    config_path = Path.home() / ".config" / "ghostty" / "config"
    if not config_path.exists():
        return None
    try:
        for raw_line in config_path.read_text().splitlines():
            stripped = raw_line.strip()
            if stripped.startswith("background") and "=" in stripped:
                bg_color = stripped.split("=", 1)[1].strip()
                if bg_color.startswith("#"):
                    is_dark = not _is_light_color(bg_color)
                    logger.debug(f"Ghostty detection: {'dark' if is_dark else 'light'} (bg={bg_color})")
                    return is_dark
    except Exception as e:
        logger.debug(f"Ghostty config read failed: {e}")
    return None


def detect_dark_mode(theme_config: str | None = None) -> bool:
    """Detect OS dark mode preference.

    Args:
        theme_config: Theme override from config (nord-dark, nord-light, or None for auto-detect)

    Returns:
        True for dark mode, False for light mode.
    """
    # 1. Config theme override
    config_result = _check_config_override(theme_config)
    if config_result is not None:
        return config_result

    # 2. Ghostty terminal config
    ghostty_result = _check_ghostty_terminal()
    if ghostty_result is not None:
        return ghostty_result

    # 3. COLORFGBG env (iTerm2, Konsole)
    colorfgbg_result = _check_colorfgbg_env()
    if colorfgbg_result is not None:
        return colorfgbg_result

    # 4. macOS detection via defaults (system-wide fallback)
    macos_result = _check_macos_theme()
    if macos_result is not None:
        return macos_result

    # 5. Default to dark
    logger.debug("Theme detection: defaulting to dark")
    return True


def _check_config_override(theme_config: str | None) -> bool | None:
    """Check config theme override. Returns True=dark, False=light, None=no override."""
    if not theme_config:
        return None
    override = theme_config.lower()
    if "dark" in override:
        logger.debug("Theme override from config: dark")
        return True
    if "light" in override:
        logger.debug("Theme override from config: light")
        return False
    return None


def _check_ghostty_terminal() -> bool | None:
    """Check Ghostty terminal theme. Returns True=dark, False=light, None=not Ghostty."""
    if os.getenv("TERM_PROGRAM") != "ghostty":
        return None
    return _detect_ghostty_theme()


def _check_colorfgbg_env() -> bool | None:
    """Check COLORFGBG env var. Returns True=dark, False=light, None=not available."""
    colorfgbg = os.getenv("COLORFGBG", "")
    if not colorfgbg:
        return None
    parts = colorfgbg.split(";")
    if len(parts) < 2:
        return None
    try:
        bg_color = int(parts[-1])
        is_dark = bg_color < 8
        logger.debug(f"COLORFGBG detection: {'dark' if is_dark else 'light'}")
        return is_dark
    except ValueError:
        return None


def _check_macos_theme() -> bool | None:
    """Check macOS theme. Returns True=dark, False=light, None=not macOS or failed."""
    if sys.platform != "darwin":
        return None
    try:
        result = subprocess.run(
            ["/usr/bin/defaults", "read", "-g", "AppleInterfaceStyle"],
            capture_output=True,
            text=True,
            timeout=1,
            check=False,
        )
        is_dark = result.returncode == 0 and "dark" in result.stdout.lower()
        logger.debug(f"macOS theme detection: {'dark' if is_dark else 'light'}")
        return is_dark
    except Exception as e:
        logger.debug(f"macOS theme detection failed: {e}")
        return None
