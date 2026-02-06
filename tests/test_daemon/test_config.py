"""Tests for daemon config."""

import tempfile
from pathlib import Path

from src.daemon.config import DaemonConfig, ScheduleConfig, StateConfig


class TestDaemonConfig:
    def test_default_config(self):
        config = DaemonConfig()

        assert config.watchlist == ["AAPL", "TSLA", "GOOGL", "MSFT"]
        assert config.interval_minutes == 30
        assert config.market_hours_only is True
        assert config.auto_trade is False
        assert config.max_concurrent_analyses == 3

    def test_custom_config(self):
        config = DaemonConfig(
            watchlist=["NVDA", "AMD"],
            interval_minutes=60,
            auto_trade=True,
        )

        assert config.watchlist == ["NVDA", "AMD"]
        assert config.interval_minutes == 60
        assert config.auto_trade is True

    def test_schedule_config_defaults(self):
        config = ScheduleConfig()

        assert config.start_time == "09:30"
        assert config.end_time == "16:00"
        assert config.timezone == "America/New_York"
        assert config.enable_pre_market is False

    def test_state_config_defaults(self):
        config = StateConfig()

        assert config.state_file == "~/.ai-casino/daemon-state.json"

    def test_from_toml(self):
        toml_content = """
[daemon]
watchlist = ["AAPL", "NVDA"]
interval_minutes = 15
market_hours_only = false
auto_trade = true
max_concurrent_analyses = 5

[daemon.schedule]
start_time = "08:00"
end_time = "17:00"
timezone = "America/Chicago"

[daemon.state]
state_file = "/tmp/test-state.json"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()
            path = Path(f.name)

        config = DaemonConfig.from_toml(path)

        assert config.watchlist == ["AAPL", "NVDA"]
        assert config.interval_minutes == 15
        assert config.market_hours_only is False
        assert config.auto_trade is True
        assert config.max_concurrent_analyses == 5
        assert config.schedule.start_time == "08:00"
        assert config.schedule.end_time == "17:00"
        assert config.schedule.timezone == "America/Chicago"
        assert "test-state.json" in config.state.state_file

        path.unlink()

    def test_schedule_config_pre_market_enabled(self):
        """Test pre-market can be enabled."""
        config = ScheduleConfig(enable_pre_market=True)
        assert config.enable_pre_market is True

    def test_from_toml_with_pre_market(self):
        """Test loading pre-market config from TOML."""
        toml_content = """
[daemon]
watchlist = ["AAPL"]
interval_minutes = 30

[daemon.schedule]
enable_pre_market = true
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()
            path = Path(f.name)

        config = DaemonConfig.from_toml(path)
        assert config.schedule.enable_pre_market is True

        path.unlink()

    def test_repr(self):
        config = DaemonConfig(watchlist=["AAPL"], interval_minutes=60, auto_trade=True)
        repr_str = repr(config)

        assert "AAPL" in repr_str
        assert "60" in repr_str
        assert "auto_trade=True" in repr_str
