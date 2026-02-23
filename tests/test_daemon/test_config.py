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
    PositionSizingConfig,
    ReportingConfig,
    RiskLimitsConfig,
    ScheduleConfig,
    SectorRotationConfig,
    StateConfig,
    TradingMode,
)
from src.v1.coordinator.models import CoordinatorConfig


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


class TestHealthConfig:
    def test_defaults(self):
        config = HealthConfig()

        assert config.enabled is True
        assert config.check_interval_seconds == 5
        assert config.archive_days == 30
        assert config.log_max_size_mb == 5
        assert config.health_dir == "~/.ai-casino/health"
        assert config.archive_dir == "~/.ai-casino/archive"

    def test_custom_values(self):
        config = HealthConfig(
            enabled=False,
            check_interval_seconds=10,
            archive_days=60,
            log_max_size_mb=10,
        )

        assert config.enabled is False
        assert config.check_interval_seconds == 10
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
    check_interval_seconds: 10
    archive_days: 14
    log_max_size_mb: 2
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            path = Path(f.name)

        config = DaemonConfig.from_yaml(path)

        assert config.health.enabled is False
        assert config.health.check_interval_seconds == 10
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
        assert config.health.check_interval_seconds == 5

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
        # Use variables to bypass type checker for validation tests
        invalid_max_var_low = 0.0
        invalid_max_var_high = 0.25
        invalid_lookback_low = 5
        invalid_lookback_high = 400
        invalid_atr_multiplier = 0.1

        with pytest.raises(ValueError, match=r"greater than or equal to 0\.001"):
            RiskLimitsConfig(max_var_95=invalid_max_var_low)
        with pytest.raises(ValueError, match=r"less than or equal to 0\.2"):
            RiskLimitsConfig(max_var_95=invalid_max_var_high)
        with pytest.raises(ValueError, match=r"greater than or equal to 20"):
            RiskLimitsConfig(lookback_days=invalid_lookback_low)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match=r"less than or equal to 365"):
            RiskLimitsConfig(lookback_days=invalid_lookback_high)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match=r"greater than or equal to 0\.5"):
            RiskLimitsConfig(atr_multiplier_min=invalid_atr_multiplier)

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
        # Use variables to bypass type checker for validation tests
        invalid_lookback_low = 20
        invalid_lookback_high = 400

        with pytest.raises(ValueError, match="lookback_days"):
            PortfolioRebalancingConfig(lookback_days=invalid_lookback_low)  # type: ignore[arg-type]

        with pytest.raises(ValueError, match="lookback_days"):
            PortfolioRebalancingConfig(lookback_days=invalid_lookback_high)  # type: ignore[arg-type]

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

    def test_from_yaml_with_position_sizing(self):
        """Test loading position sizing config from YAML."""
        yaml_content = """
daemon:
  watchlist: ["AAPL"]
  position_sizing:
    primary_goal: "maximize_returns"
    risk_tolerance: "aggressive"
    complexity: "advanced"
    max_risk_per_trade_pct: 3.0
    max_single_position_pct: 25.0
    max_total_exposure_pct: 90.0
    blend_weight_optimization: 0.7
    blend_weight_risk_based: 0.3
    confidence_scaling_enabled: true
    confidence_high_threshold: 0.85
    confidence_low_threshold: 0.65
    confidence_low_reduction_factor: 0.6
    use_monte_carlo_adjustment: true
    monte_carlo_risk_multiplier: 0.8
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            path = Path(f.name)

        config = DaemonConfig.from_yaml(path)
        assert isinstance(config.position_sizing, PositionSizingConfig)
        assert config.position_sizing.primary_goal == "maximize_returns"
        assert config.position_sizing.risk_tolerance == "aggressive"
        assert config.position_sizing.complexity == "advanced"
        assert config.position_sizing.max_risk_per_trade_pct == 3.0
        assert config.position_sizing.max_single_position_pct == 25.0
        assert config.position_sizing.max_total_exposure_pct == 90.0
        assert config.position_sizing.blend_weight_optimization == 0.7
        assert config.position_sizing.blend_weight_risk_based == 0.3
        assert config.position_sizing.confidence_scaling_enabled is True
        assert config.position_sizing.confidence_high_threshold == 0.85
        assert config.position_sizing.confidence_low_threshold == 0.65
        assert config.position_sizing.confidence_low_reduction_factor == 0.6
        assert config.position_sizing.use_monte_carlo_adjustment is True
        assert config.position_sizing.monte_carlo_risk_multiplier == 0.8

        path.unlink()

    def test_position_sizing_blend_weights_validation(self):
        """Test blend weights validation."""
        # Valid: weights sum to 1.0
        config = PositionSizingConfig(
            blend_weight_optimization=0.6,
            blend_weight_risk_based=0.4,
        )
        assert config.blend_weight_optimization == 0.6
        assert config.blend_weight_risk_based == 0.4

        # Invalid: weights sum to 0.9
        with pytest.raises(ValueError, match=r"Blend weights must sum to 1\.0"):
            PositionSizingConfig(
                blend_weight_optimization=0.6,
                blend_weight_risk_based=0.3,
            )


class TestCoordinatorConfig:
    def test_defaults(self):
        config = CoordinatorConfig()

        assert config.enabled is True
        assert config.max_tool_calls == 25
        assert config.temperature == 0.5
        assert config.model_override is None
        assert config.confirmation_mode == "auto"
        assert config.cycle_timeout_seconds == 600
        assert config.max_daily_trades == 10
        assert config.max_position_pct == 10.0
        assert config.min_confidence_to_trade == 0.6

    def test_custom(self):
        config = CoordinatorConfig(
            enabled=True,
            max_tool_calls=30,
            temperature=0.7,
            model_override="claude-sonnet-4",
            confirmation_mode="manual",
            cycle_timeout_seconds=900,
            max_daily_trades=20,
            max_position_pct=15.0,
            min_confidence_to_trade=0.8,
        )

        assert config.enabled is True
        assert config.max_tool_calls == 30
        assert config.temperature == 0.7
        assert config.model_override == "claude-sonnet-4"
        assert config.confirmation_mode == "manual"
        assert config.cycle_timeout_seconds == 900
        assert config.max_daily_trades == 20
        assert config.max_position_pct == 15.0
        assert config.min_confidence_to_trade == 0.8

    def test_validation_bounds(self):
        # Use variables to bypass type checker for validation tests
        invalid_tool_calls_low = 3
        invalid_tool_calls_high = 60
        invalid_timeout_low = 30
        invalid_timeout_high = 4000
        invalid_trades_low = 0
        invalid_trades_high = 150

        with pytest.raises(ValueError, match=r"greater than or equal to 5"):
            CoordinatorConfig(max_tool_calls=invalid_tool_calls_low)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match=r"less than or equal to 50"):
            CoordinatorConfig(max_tool_calls=invalid_tool_calls_high)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match=r"greater than or equal to 60"):
            CoordinatorConfig(cycle_timeout_seconds=invalid_timeout_low)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match=r"less than or equal to 3600"):
            CoordinatorConfig(cycle_timeout_seconds=invalid_timeout_high)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match=r"greater than or equal to 1"):
            CoordinatorConfig(max_daily_trades=invalid_trades_low)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match=r"less than or equal to 100"):
            CoordinatorConfig(max_daily_trades=invalid_trades_high)  # type: ignore[arg-type]

    def test_daemon_config_has_coordinator(self):
        config = DaemonConfig()
        assert isinstance(config.coordinator, CoordinatorConfig)
        assert config.coordinator.enabled is True

    def test_from_yaml_with_coordinator(self):
        yaml_content = """
daemon:
  watchlist: ["AAPL"]
  coordinator:
    enabled: true
    max_tool_calls: 30
    temperature: 0.7
    model_override: "claude-sonnet-4"
    confirmation_mode: "manual"
    cycle_timeout_seconds: 900
    max_daily_trades: 20
    max_position_pct: 15.0
    min_confidence_to_trade: 0.8
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            path = Path(f.name)

        config = DaemonConfig.from_yaml(path)

        assert config.coordinator.enabled is True
        assert config.coordinator.max_tool_calls == 30
        assert config.coordinator.temperature == 0.7
        assert config.coordinator.model_override == "claude-sonnet-4"
        assert config.coordinator.confirmation_mode == "manual"
        assert config.coordinator.cycle_timeout_seconds == 900
        assert config.coordinator.max_daily_trades == 20
        assert config.coordinator.max_position_pct == 15.0
        assert config.coordinator.min_confidence_to_trade == 0.8

        path.unlink()

    def test_from_yaml_without_coordinator_uses_defaults(self):
        yaml_content = """
daemon:
  watchlist: ["AAPL"]
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            path = Path(f.name)

        config = DaemonConfig.from_yaml(path)

        assert config.coordinator.enabled is True
        assert config.coordinator.max_tool_calls == 25
        assert config.coordinator.temperature == 0.5

        path.unlink()


class TestRemoveWatchlistSymbol:
    def test_removes_symbol_from_memory(self):
        config = DaemonConfig(watchlist=["AAPL", "TSLA", "GOOGL"])
        config.remove_watchlist_symbol("TSLA")
        assert "TSLA" not in config.watchlist
        assert config.watchlist == ["AAPL", "GOOGL"]

    def test_no_op_when_symbol_absent(self):
        config = DaemonConfig(watchlist=["AAPL", "TSLA"])
        config.remove_watchlist_symbol("NVDA")
        assert config.watchlist == ["AAPL", "TSLA"]

    def test_no_config_path_skips_persistence(self):
        config = DaemonConfig(watchlist=["AAPL", "TSLA"])
        assert config._config_path is None
        config.remove_watchlist_symbol("TSLA")
        assert "TSLA" not in config.watchlist

    def test_persists_removal_to_yaml(self):
        yaml_content = """
daemon:
  watchlist: ["AAPL", "TSLA", "GOOGL"]
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            path = Path(f.name)

        import yaml

        config = DaemonConfig.from_yaml(path)
        config.remove_watchlist_symbol("TSLA")

        with path.open() as f:
            data = yaml.safe_load(f)

        assert "TSLA" not in data["daemon"]["watchlist"]
        assert "AAPL" in data["daemon"]["watchlist"]
        assert "GOOGL" in data["daemon"]["watchlist"]

        path.unlink()

    def test_persists_removal_empty_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            f.flush()
            path = Path(f.name)

        config = DaemonConfig(watchlist=["AAPL"])
        config._config_path = path
        config.remove_watchlist_symbol("AAPL")

        assert "AAPL" not in config.watchlist

        path.unlink()


class TestDatabaseConfig:
    def test_database_config_pool_params(self):
        """Test database pool configuration parameters."""
        from src.daemon.config.infrastructure import DatabaseConfig

        config = DatabaseConfig(
            pool_size=10,
            max_overflow=20,
            pool_timeout=60.0,
            pool_recycle=1800,
            pool_pre_ping=False,
        )
        assert config.pool_size == 10
        assert config.max_overflow == 20
        assert config.pool_timeout == 60.0
        assert config.pool_recycle == 1800
        assert config.pool_pre_ping is False

    def test_from_yaml_with_database_pool_params(self):
        """Test loading database pool params from YAML."""
        yaml_content = """
daemon:
  watchlist: ["AAPL"]
  database:
    pool_size: 10
    max_overflow: 20
    pool_timeout: 60.0
    pool_recycle: 1800
    pool_pre_ping: false
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            path = Path(f.name)

        config = DaemonConfig.from_yaml(path)
        assert config.database.pool_size == 10
        assert config.database.max_overflow == 20
        assert config.database.pool_timeout == 60.0
        assert config.database.pool_recycle == 1800
        assert config.database.pool_pre_ping is False

        path.unlink()
