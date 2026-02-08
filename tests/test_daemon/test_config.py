"""Tests for daemon config."""

import tempfile
from pathlib import Path

import pytest

from src.daemon.config import (
    DaemonConfig,
    EarningsCalendarConfig,
    HealthConfig,
    PaperTradingConfig,
    PeerAnalysisConfig,
    PortfolioRebalancingConfig,
    ReportingConfig,
    RiskLimitsConfig,
    ScheduleConfig,
    ScreeningConfig,
    SectorRotationConfig,
    StateConfig,
    TradingMode,
)


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
        assert config.enable_after_hours is False

    def test_state_config_defaults(self):
        config = StateConfig()

        assert config.state_file == "~/.ai-casino/daemon-state.json"

    def test_from_yaml(self):
        yaml_content = """
daemon:
  watchlist: ["AAPL", "NVDA"]
  interval_minutes: 15
  market_hours_only: false
  auto_trade: true
  max_concurrent_analyses: 5
  schedule:
    start_time: "08:00"
    end_time: "17:00"
    timezone: "America/Chicago"
  state:
    state_file: "/tmp/test-state.json"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            path = Path(f.name)

        config = DaemonConfig.from_yaml(path)

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

    def test_from_yaml_with_pre_market(self):
        """Test loading pre-market config from YAML."""
        yaml_content = """
daemon:
  watchlist: ["AAPL"]
  interval_minutes: 30
  schedule:
    enable_pre_market: true
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            path = Path(f.name)

        config = DaemonConfig.from_yaml(path)
        assert config.schedule.enable_pre_market is True

        path.unlink()

    def test_repr(self):
        config = DaemonConfig(watchlist=["AAPL"], interval_minutes=60, auto_trade=True)
        repr_str = repr(config)

        assert "AAPL" in repr_str
        assert "60" in repr_str
        assert "auto_trade=True" in repr_str


class TestScreeningConfig:
    def test_defaults(self):
        config = ScreeningConfig()

        assert config.enabled is False
        assert config.screen_time == "16:30"
        assert config.screen_days == ["mon", "tue", "wed", "thu", "fri"]
        assert config.criteria == "momentum"
        assert config.universe == "COMBINED"
        assert config.top_n == 10
        assert config.watchlist_name == "daemon-screening"

    def test_custom(self):
        config = ScreeningConfig(
            enabled=True,
            screen_time="17:00",
            screen_days=["mon", "wed", "fri"],
            criteria="breakout",
            universe="SP500",
            top_n=5,
            watchlist_name="custom-screen",
        )

        assert config.enabled is True
        assert config.screen_time == "17:00"
        assert config.screen_days == ["mon", "wed", "fri"]
        assert config.criteria == "breakout"
        assert config.universe == "SP500"
        assert config.top_n == 5
        assert config.watchlist_name == "custom-screen"

    def test_validate_screen_time_valid(self):
        config = ScreeningConfig(enabled=True, screen_time="18:00")
        assert config.screen_time == "18:00"

    def test_validate_screen_time_boundary_2000(self):
        config = ScreeningConfig(enabled=True, screen_time="20:00")
        assert config.screen_time == "20:00"

    def test_validate_screen_time_invalid_format(self):
        with pytest.raises(ValueError, match="HH:MM format"):
            ScreeningConfig(enabled=True, screen_time="bad")

    def test_validate_screen_time_out_of_range(self):
        with pytest.raises(ValueError, match="16:00-20:00"):
            ScreeningConfig(enabled=True, screen_time="21:00")

    def test_validate_screen_time_skipped_when_disabled(self):
        config = ScreeningConfig(enabled=False, screen_time="99:99")
        assert config.screen_time == "99:99"

    def test_validate_screen_time_strict_format(self):
        """Test strict HH:MM format enforcement."""
        with pytest.raises(ValueError, match="HH:MM format"):
            ScreeningConfig(enabled=True, screen_time="16:3")

    def test_validate_screen_time_invalid_minute(self):
        """Test minute bounds validation."""
        with pytest.raises(ValueError, match="HH:MM format"):
            ScreeningConfig(enabled=True, screen_time="19:99")

    def test_validate_screen_time_invalid_hour(self):
        """Test hour bounds validation."""
        with pytest.raises(ValueError, match="HH:MM format"):
            ScreeningConfig(enabled=True, screen_time="25:00")

    def test_daemon_config_has_screening(self):
        config = DaemonConfig()
        assert isinstance(config.screening, ScreeningConfig)
        assert config.screening.enabled is False

    def test_from_yaml_with_screening(self):
        yaml_content = """
daemon:
  watchlist: ["AAPL"]
  screening:
    enabled: true
    screen_time: "17:00"
    criteria: "breakout"
    universe: "SP500"
    top_n: 5
    watchlist_name: "my-screen"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            path = Path(f.name)

        config = DaemonConfig.from_yaml(path)

        assert config.screening.enabled is True
        assert config.screening.screen_time == "17:00"
        assert config.screening.criteria == "breakout"
        assert config.screening.universe == "SP500"
        assert config.screening.top_n == 5
        assert config.screening.watchlist_name == "my-screen"

        path.unlink()


class TestHealthConfig:
    def test_defaults(self):
        config = HealthConfig()

        assert config.enabled is True
        assert config.run_time == "17:00"
        assert config.archive_days == 30
        assert config.log_max_size_mb == 5
        assert config.health_dir == "~/.ai-casino/health"
        assert config.archive_dir == "~/.ai-casino/archive"

    def test_custom_values(self):
        config = HealthConfig(
            enabled=False,
            run_time="18:00",
            archive_days=60,
            log_max_size_mb=10,
        )

        assert config.enabled is False
        assert config.run_time == "18:00"
        assert config.archive_days == 60
        assert config.log_max_size_mb == 10

    def test_daemon_config_includes_health(self):
        config = DaemonConfig()
        assert isinstance(config.health, HealthConfig)
        assert config.health.enabled is True

    def test_from_yaml_with_health(self):
        yaml_content = """
daemon:
  watchlist: ["AAPL"]
  health:
    enabled: false
    run_time: "18:30"
    archive_days: 14
    log_max_size_mb: 2
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            path = Path(f.name)

        config = DaemonConfig.from_yaml(path)

        assert config.health.enabled is False
        assert config.health.run_time == "18:30"
        assert config.health.archive_days == 14
        assert config.health.log_max_size_mb == 2

        path.unlink()

    def test_from_yaml_without_health_uses_defaults(self):
        yaml_content = """
daemon:
  watchlist: ["AAPL"]
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            path = Path(f.name)

        config = DaemonConfig.from_yaml(path)

        assert config.health.enabled is True
        assert config.health.run_time == "17:00"

        path.unlink()


class TestSectorRotationConfig:
    def test_defaults(self):
        config = SectorRotationConfig()

        assert config.enabled is False
        assert config.run_time == "16:15"
        assert config.run_days == ["mon", "tue", "wed", "thu", "fri"]
        assert config.boost_factor == 0.15

    def test_custom(self):
        config = SectorRotationConfig(
            enabled=True,
            run_time="16:30",
            run_days=["mon", "wed", "fri"],
            boost_factor=0.20,
        )

        assert config.enabled is True
        assert config.run_time == "16:30"
        assert config.run_days == ["mon", "wed", "fri"]
        assert config.boost_factor == 0.20

    def test_validate_run_time_valid(self):
        config = SectorRotationConfig(enabled=True, run_time="17:00")
        assert config.run_time == "17:00"

    def test_validate_run_time_invalid_format(self):
        with pytest.raises(ValueError, match="HH:MM format"):
            SectorRotationConfig(enabled=True, run_time="bad")

    def test_validate_run_time_out_of_range(self):
        with pytest.raises(ValueError, match="16:00-20:00"):
            SectorRotationConfig(enabled=True, run_time="21:00")

    def test_validate_run_time_skipped_when_disabled(self):
        config = SectorRotationConfig(enabled=False, run_time="99:99")
        assert config.run_time == "99:99"

    def test_daemon_config_has_sector_rotation(self):
        config = DaemonConfig()
        assert isinstance(config.sector_rotation, SectorRotationConfig)
        assert config.sector_rotation.enabled is False

    def test_from_yaml_with_sector_rotation(self):
        yaml_content = """
daemon:
  watchlist: ["AAPL"]
  sector_rotation:
    enabled: true
    run_time: "16:30"
    run_days: ["mon", "wed", "fri"]
    boost_factor: 0.20
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            path = Path(f.name)

        config = DaemonConfig.from_yaml(path)

        assert config.sector_rotation.enabled is True
        assert config.sector_rotation.run_time == "16:30"
        assert config.sector_rotation.run_days == ["mon", "wed", "fri"]
        assert config.sector_rotation.boost_factor == 0.20

        path.unlink()

    def test_from_yaml_without_sector_rotation_uses_defaults(self):
        yaml_content = """
daemon:
  watchlist: ["AAPL"]
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            path = Path(f.name)

        config = DaemonConfig.from_yaml(path)

        assert config.sector_rotation.enabled is False
        assert config.sector_rotation.run_time == "16:15"

        path.unlink()


class TestEarningsCalendarConfig:
    def test_defaults(self):
        config = EarningsCalendarConfig()

        assert config.enabled is False
        assert config.fetch_time == "16:45"
        assert config.fetch_days == ["mon"]
        assert config.lookahead_days == 3
        assert config.reduce_position_t1 is False
        assert config.position_reduction_factor == 0.5

    def test_custom(self):
        config = EarningsCalendarConfig(
            enabled=True,
            fetch_time="17:30",
            fetch_days=["mon", "thu"],
            lookahead_days=5,
            reduce_position_t1=True,
            position_reduction_factor=0.3,
        )

        assert config.enabled is True
        assert config.fetch_time == "17:30"
        assert config.fetch_days == ["mon", "thu"]
        assert config.lookahead_days == 5
        assert config.reduce_position_t1 is True
        assert config.position_reduction_factor == 0.3

    def test_validate_fetch_time_valid(self):
        config = EarningsCalendarConfig(enabled=True, fetch_time="18:00")
        assert config.fetch_time == "18:00"

    def test_validate_fetch_time_invalid_format(self):
        with pytest.raises(ValueError, match="HH:MM format"):
            EarningsCalendarConfig(enabled=True, fetch_time="bad")

    def test_validate_fetch_time_out_of_range(self):
        with pytest.raises(ValueError, match="16:00-20:00"):
            EarningsCalendarConfig(enabled=True, fetch_time="21:00")

    def test_validate_fetch_time_skipped_when_disabled(self):
        config = EarningsCalendarConfig(enabled=False, fetch_time="99:99")
        assert config.fetch_time == "99:99"

    def test_daemon_config_has_earnings_calendar(self):
        config = DaemonConfig()
        assert isinstance(config.earnings_calendar, EarningsCalendarConfig)
        assert config.earnings_calendar.enabled is False

    def test_from_yaml_with_earnings_calendar(self):
        yaml_content = """
daemon:
  watchlist: ["AAPL"]
  earnings_calendar:
    enabled: true
    fetch_time: "17:00"
    fetch_days: ["mon", "thu"]
    lookahead_days: 5
    reduce_position_t1: true
    position_reduction_factor: 0.3
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            path = Path(f.name)

        config = DaemonConfig.from_yaml(path)

        assert config.earnings_calendar.enabled is True
        assert config.earnings_calendar.fetch_time == "17:00"
        assert config.earnings_calendar.fetch_days == ["mon", "thu"]
        assert config.earnings_calendar.lookahead_days == 5
        assert config.earnings_calendar.reduce_position_t1 is True
        assert config.earnings_calendar.position_reduction_factor == 0.3

        path.unlink()

    def test_from_yaml_without_earnings_calendar_uses_defaults(self):
        yaml_content = """
daemon:
  watchlist: ["AAPL"]
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            path = Path(f.name)

        config = DaemonConfig.from_yaml(path)

        assert config.earnings_calendar.enabled is False
        assert config.earnings_calendar.fetch_time == "16:45"

        path.unlink()


class TestPeerAnalysisConfig:
    def test_defaults(self):
        config = PeerAnalysisConfig()

        assert config.enabled is False
        assert config.run_time == "17:30"
        assert config.run_days == ["sun"]
        assert config.max_peers == 10
        assert config.output_dir == "~/.ai-casino/peer-analysis"
        assert config.rate_limit_sleep == 13.0

    def test_custom(self):
        config = PeerAnalysisConfig(
            enabled=True,
            run_time="18:00",
            run_days=["sat", "sun"],
            max_peers=15,
            rate_limit_sleep=15.0,
        )

        assert config.enabled is True
        assert config.run_time == "18:00"
        assert config.run_days == ["sat", "sun"]
        assert config.max_peers == 15

    def test_validate_run_time_valid(self):
        config = PeerAnalysisConfig(enabled=True, run_time="17:30")
        assert config.run_time == "17:30"

    def test_validate_run_time_invalid_format(self):
        with pytest.raises(ValueError, match="HH:MM format"):
            PeerAnalysisConfig(enabled=True, run_time="bad")

    def test_validate_run_time_skipped_when_disabled(self):
        config = PeerAnalysisConfig(enabled=False, run_time="99:99")
        assert config.run_time == "99:99"

    def test_daemon_config_has_peer_analysis(self):
        config = DaemonConfig()
        assert isinstance(config.peer_analysis, PeerAnalysisConfig)
        assert config.peer_analysis.enabled is False

    def test_from_yaml_with_peer_analysis(self):
        yaml_content = """
daemon:
  watchlist: ["AAPL"]
  peer_analysis:
    enabled: true
    run_time: "18:00"
    run_days: ["sat", "sun"]
    max_peers: 15
    rate_limit_sleep: 15.0
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            path = Path(f.name)

        config = DaemonConfig.from_yaml(path)

        assert config.peer_analysis.enabled is True
        assert config.peer_analysis.run_time == "18:00"
        assert config.peer_analysis.run_days == ["sat", "sun"]
        assert config.peer_analysis.max_peers == 15

        path.unlink()

    def test_from_yaml_without_peer_analysis_uses_defaults(self):
        yaml_content = """
daemon:
  watchlist: ["AAPL"]
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            path = Path(f.name)

        config = DaemonConfig.from_yaml(path)

        assert config.peer_analysis.enabled is False
        assert config.peer_analysis.run_time == "17:30"

        path.unlink()


class TestReportingConfig:
    def test_defaults(self):
        config = ReportingConfig()

        assert config.enabled is False
        assert config.tearsheet_time == "16:30"
        assert config.benchmark == "SPY"
        assert config.retention_days == 30

    def test_custom(self):
        config = ReportingConfig(
            enabled=True,
            tearsheet_time="17:00",
            benchmark="QQQ",
            retention_days=60,
        )

        assert config.enabled is True
        assert config.tearsheet_time == "17:00"
        assert config.benchmark == "QQQ"
        assert config.retention_days == 60

    def test_validate_tearsheet_time_valid(self):
        config = ReportingConfig(enabled=True, tearsheet_time="18:00")
        assert config.tearsheet_time == "18:00"

    def test_validate_tearsheet_time_invalid_format(self):
        with pytest.raises(ValueError, match="HH:MM format"):
            ReportingConfig(enabled=True, tearsheet_time="bad")

    def test_validate_tearsheet_time_out_of_range(self):
        with pytest.raises(ValueError, match="16:00-20:00"):
            ReportingConfig(enabled=True, tearsheet_time="21:00")

    def test_validate_tearsheet_time_skipped_when_disabled(self):
        config = ReportingConfig(enabled=False, tearsheet_time="99:99")
        assert config.tearsheet_time == "99:99"

    def test_validate_retention_days_invalid(self):
        with pytest.raises(ValueError, match="retention_days must be >= 1"):
            ReportingConfig(enabled=True, retention_days=0)

        with pytest.raises(ValueError, match="retention_days must be >= 1"):
            ReportingConfig(enabled=True, retention_days=-5)

    def test_validate_retention_days_skipped_when_disabled(self):
        config = ReportingConfig(enabled=False, retention_days=-1)
        assert config.retention_days == -1

    def test_daemon_config_has_reporting(self):
        config = DaemonConfig()
        assert isinstance(config.reporting, ReportingConfig)
        assert config.reporting.enabled is False

    def test_from_yaml_with_reporting(self):
        yaml_content = """
daemon:
  watchlist: ["AAPL"]
  reporting:
    enabled: true
    tearsheet_time: "17:00"
    benchmark: "QQQ"
    retention_days: 60
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            path = Path(f.name)

        config = DaemonConfig.from_yaml(path)

        assert config.reporting.enabled is True
        assert config.reporting.tearsheet_time == "17:00"
        assert config.reporting.benchmark == "QQQ"
        assert config.reporting.retention_days == 60

        path.unlink()

    def test_from_yaml_without_reporting_uses_defaults(self):
        yaml_content = """
daemon:
  watchlist: ["AAPL"]
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            path = Path(f.name)

        config = DaemonConfig.from_yaml(path)

        assert config.reporting.enabled is False
        assert config.reporting.tearsheet_time == "16:30"
        assert config.reporting.benchmark == "SPY"
        assert config.reporting.retention_days == 30

        path.unlink()


class TestRiskLimitsConfig:
    def test_defaults(self):
        config = RiskLimitsConfig()

        assert config.enabled is False
        assert config.max_var_95 == 0.03
        assert config.max_cvar_99 == 0.05
        assert config.lookback_days == 90
        assert config.adaptive_stop_loss is True
        assert config.cdar_stop_threshold == 0.10
        assert config.atr_multiplier_min == 1.0
        assert config.report_dir == "~/.ai-casino/risk-reports"

    def test_custom(self):
        config = RiskLimitsConfig(
            enabled=True,
            max_var_95=0.05,
            max_cvar_99=0.08,
            lookback_days=120,
            adaptive_stop_loss=False,
            cdar_stop_threshold=0.15,
            atr_multiplier_min=1.5,
            report_dir="~/.ai-casino/custom-reports",
        )

        assert config.enabled is True
        assert config.max_var_95 == 0.05
        assert config.max_cvar_99 == 0.08
        assert config.lookback_days == 120
        assert config.adaptive_stop_loss is False
        assert config.cdar_stop_threshold == 0.15
        assert config.atr_multiplier_min == 1.5
        assert config.report_dir == "~/.ai-casino/custom-reports"

    def test_validation_bounds(self):
        with pytest.raises(ValueError, match=r"greater than or equal to 0\.001"):
            RiskLimitsConfig(max_var_95=0.0)
        with pytest.raises(ValueError, match=r"less than or equal to 0\.2"):
            RiskLimitsConfig(max_var_95=0.25)
        with pytest.raises(ValueError, match=r"greater than or equal to 20"):
            RiskLimitsConfig(lookback_days=5)
        with pytest.raises(ValueError, match=r"less than or equal to 365"):
            RiskLimitsConfig(lookback_days=400)
        with pytest.raises(ValueError, match=r"greater than or equal to 0\.5"):
            RiskLimitsConfig(atr_multiplier_min=0.1)

    def test_daemon_config_has_risk_limits(self):
        config = DaemonConfig()
        assert isinstance(config.risk_limits, RiskLimitsConfig)
        assert config.risk_limits.enabled is False

    def test_from_yaml_with_risk_limits(self):
        yaml_content = """
daemon:
  watchlist: ["AAPL"]
  risk_limits:
    enabled: true
    max_var_95: 0.05
    max_cvar_99: 0.08
    lookback_days: 120
    adaptive_stop_loss: false
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            path = Path(f.name)

        config = DaemonConfig.from_yaml(path)

        assert config.risk_limits.enabled is True
        assert config.risk_limits.max_var_95 == 0.05
        assert config.risk_limits.max_cvar_99 == 0.08
        assert config.risk_limits.lookback_days == 120
        assert config.risk_limits.adaptive_stop_loss is False

        path.unlink()

    def test_from_yaml_without_risk_limits_uses_defaults(self):
        yaml_content = """
daemon:
  watchlist: ["AAPL"]
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            path = Path(f.name)

        config = DaemonConfig.from_yaml(path)

        assert config.risk_limits.enabled is False
        assert config.risk_limits.max_var_95 == 0.03

        path.unlink()


class TestPortfolioRebalancingConfig:
    def test_default_config(self):
        config = PortfolioRebalancingConfig()

        assert config.enabled is False
        assert config.method == "max_sharpe"
        assert config.run_time == "16:45"
        assert config.run_days == ["mon"]
        assert config.rebalance_threshold == 0.01
        assert config.lookback_days == 90

    def test_custom_config(self):
        config = PortfolioRebalancingConfig(
            enabled=True,
            method="min_volatility",
            run_time="17:00",
            run_days=["mon", "fri"],
            rebalance_threshold=0.05,
            lookback_days=180,
        )

        assert config.enabled is True
        assert config.method == "min_volatility"
        assert config.run_time == "17:00"
        assert config.run_days == ["mon", "fri"]
        assert config.rebalance_threshold == 0.05
        assert config.lookback_days == 180

    def test_validate_run_time_valid(self):
        config = PortfolioRebalancingConfig(enabled=True, run_time="16:30")
        assert config.run_time == "16:30"

        config = PortfolioRebalancingConfig(enabled=True, run_time="19:59")
        assert config.run_time == "19:59"

        config = PortfolioRebalancingConfig(enabled=True, run_time="20:00")
        assert config.run_time == "20:00"

    def test_validate_run_time_invalid_format(self):
        with pytest.raises(ValueError, match="must be in HH:MM format"):
            PortfolioRebalancingConfig(enabled=True, run_time="25:00")

        with pytest.raises(ValueError, match="must be in HH:MM format"):
            PortfolioRebalancingConfig(enabled=True, run_time="16:60")

        with pytest.raises(ValueError, match="must be in HH:MM format"):
            PortfolioRebalancingConfig(enabled=True, run_time="invalid")

    def test_validate_run_time_out_of_range(self):
        with pytest.raises(ValueError, match="must be between 16:00-20:00"):
            PortfolioRebalancingConfig(enabled=True, run_time="15:59")

        with pytest.raises(ValueError, match="must be between 16:00-20:00"):
            PortfolioRebalancingConfig(enabled=True, run_time="20:01")

        with pytest.raises(ValueError, match="must be between 16:00-20:00"):
            PortfolioRebalancingConfig(enabled=True, run_time="09:00")

    def test_validate_run_time_skipped_when_disabled(self):
        config = PortfolioRebalancingConfig(enabled=False, run_time="09:00")
        assert config.run_time == "09:00"

    def test_threshold_bounds(self):
        with pytest.raises(ValueError, match="rebalance_threshold"):
            PortfolioRebalancingConfig(rebalance_threshold=0.0005)

        with pytest.raises(ValueError, match="rebalance_threshold"):
            PortfolioRebalancingConfig(rebalance_threshold=0.25)

    def test_lookback_days_bounds(self):
        with pytest.raises(ValueError, match="lookback_days"):
            PortfolioRebalancingConfig(lookback_days=20)

        with pytest.raises(ValueError, match="lookback_days"):
            PortfolioRebalancingConfig(lookback_days=400)

    def test_daemon_config_has_rebalancing(self):
        config = DaemonConfig()
        assert isinstance(config.rebalancing, PortfolioRebalancingConfig)
        assert config.rebalancing.enabled is False

    def test_from_yaml_with_rebalancing(self):
        yaml_content = """
daemon:
  watchlist: ["AAPL"]
  rebalancing:
    enabled: true
    method: "hrp"
    run_time: "17:30"
    run_days: ["mon", "wed", "fri"]
    rebalance_threshold: 0.02
    lookback_days: 120
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            path = Path(f.name)

        config = DaemonConfig.from_yaml(path)

        assert config.rebalancing.enabled is True
        assert config.rebalancing.method == "hrp"
        assert config.rebalancing.run_time == "17:30"
        assert config.rebalancing.run_days == ["mon", "wed", "fri"]
        assert config.rebalancing.rebalance_threshold == 0.02
        assert config.rebalancing.lookback_days == 120

        path.unlink()


class TestPaperTradingConfig:
    def test_paper_trading_config_defaults(self):
        config = PaperTradingConfig()
        assert config.min_duration_days == 30
        assert config.min_trades == 20
        assert config.min_sharpe == 0.5
        assert config.max_drawdown_percent == 15.0
        assert config.min_win_rate == 0.45

    def test_trading_mode_parsing(self):
        config = DaemonConfig(trading_mode=TradingMode.PAPER)
        assert config.trading_mode == TradingMode.PAPER

        config = DaemonConfig(trading_mode=TradingMode.LIVE)
        assert config.trading_mode == TradingMode.LIVE

    def test_trading_mode_from_yaml(self):
        yaml_content = """
daemon:
  trading_mode: "paper"
  watchlist: ["AAPL"]
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            path = Path(f.name)

        config = DaemonConfig.from_yaml(path)
        assert config.trading_mode == TradingMode.PAPER

        path.unlink()

    def test_daemon_config_has_paper_trading(self):
        config = DaemonConfig()
        assert isinstance(config.paper_trading, PaperTradingConfig)
        assert config.paper_trading.min_duration_days == 30
