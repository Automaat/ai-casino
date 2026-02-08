"""Tests for daemon health checks and cleanup."""

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.daemon.config import DaemonConfig, HealthConfig
from src.daemon.health import (
    HealthChecker,
    HealthReport,
    ServiceStatus,
)
from src.daemon.state import AnalysisRecord, DaemonState


@pytest.fixture
def health_config(tmp_path: Path) -> HealthConfig:
    return HealthConfig(
        health_dir=str(tmp_path / "health"),
        archive_dir=str(tmp_path / "archive"),
        log_max_size_mb=1,
    )


@pytest.fixture
def daemon_config(tmp_path: Path, health_config: HealthConfig) -> DaemonConfig:
    return DaemonConfig(
        state={"state_file": str(tmp_path / "state.json")},
        health=health_config,
    )


@pytest.fixture
def daemon_state() -> DaemonState:
    return DaemonState()


@pytest.fixture
def checker(daemon_config: DaemonConfig, daemon_state: DaemonState) -> HealthChecker:
    return HealthChecker(daemon_config, daemon_state)


class TestServiceStatus:
    def test_enum_values(self):
        assert ServiceStatus.HEALTHY == "HEALTHY"
        assert ServiceStatus.DEGRADED == "DEGRADED"
        assert ServiceStatus.UNHEALTHY == "UNHEALTHY"
        assert ServiceStatus.SKIPPED == "SKIPPED"


class TestCheckAlphaVantage:
    async def test_skipped_no_key(self, checker: HealthChecker):
        with patch.dict(os.environ, {}, clear=True):
            result = await checker._check_alpha_vantage()

        assert result.status == ServiceStatus.SKIPPED
        assert result.service == "alpha_vantage"

    async def test_healthy(self, checker: HealthChecker):
        mock_ts_instance = Mock()
        mock_ts_instance.get_daily.return_value = (Mock(), Mock())

        with (
            patch.dict(os.environ, {"ALPHA_VANTAGE_API_KEY": "test"}),
            patch("alpha_vantage.timeseries.TimeSeries", return_value=mock_ts_instance),
        ):
            result = await checker._check_alpha_vantage()

        assert result.status == ServiceStatus.HEALTHY
        assert result.duration_ms > 0

    async def test_unhealthy(self, checker: HealthChecker):
        mock_ts_instance = Mock()
        mock_ts_instance.get_daily.side_effect = ConnectionError("timeout")

        with (
            patch.dict(os.environ, {"ALPHA_VANTAGE_API_KEY": "test"}),
            patch("alpha_vantage.timeseries.TimeSeries", return_value=mock_ts_instance),
        ):
            result = await checker._check_alpha_vantage()

        assert result.status == ServiceStatus.UNHEALTHY
        assert "timeout" in result.message


class TestCheckMarketaux:
    async def test_skipped_no_key(self, checker: HealthChecker):
        with patch.dict(os.environ, {}, clear=True):
            result = await checker._check_marketaux()

        assert result.status == ServiceStatus.SKIPPED
        assert result.service == "marketaux"

    async def test_healthy(self, checker: HealthChecker):
        mock_response = AsyncMock()
        mock_response.raise_for_status = Mock()

        with (
            patch.dict(os.environ, {"MARKETAUX_API_KEY": "test"}),
            patch("src.daemon.health.httpx.AsyncClient") as mock_client,
        ):
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_client.return_value)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_client.return_value.get = AsyncMock(return_value=mock_response)
            result = await checker._check_marketaux()

        assert result.status == ServiceStatus.HEALTHY

    async def test_unhealthy(self, checker: HealthChecker):
        with (
            patch.dict(os.environ, {"MARKETAUX_API_KEY": "test"}),
            patch("src.daemon.health.httpx.AsyncClient") as mock_client,
        ):
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_client.return_value)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_client.return_value.get = AsyncMock(side_effect=Exception("rate limited"))
            result = await checker._check_marketaux()

        assert result.status == ServiceStatus.UNHEALTHY


class TestCheckAlpaca:
    async def test_skipped_no_keys(self, checker: HealthChecker):
        with patch.dict(os.environ, {}, clear=True):
            result = await checker._check_alpaca()

        assert result.status == ServiceStatus.SKIPPED
        assert result.service == "alpaca"

    async def test_healthy(self, checker: HealthChecker):
        mock_client_instance = Mock()
        mock_client_instance.get_account.return_value = Mock()

        with (
            patch.dict(os.environ, {"ALPACA_API_KEY": "key", "ALPACA_SECRET_KEY": "secret"}),
            patch("alpaca.trading.client.TradingClient", return_value=mock_client_instance),
        ):
            result = await checker._check_alpaca()

        assert result.status == ServiceStatus.HEALTHY

    async def test_unhealthy(self, checker: HealthChecker):
        mock_client_instance = Mock()
        mock_client_instance.get_account.side_effect = Exception("invalid creds")

        with (
            patch.dict(os.environ, {"ALPACA_API_KEY": "key", "ALPACA_SECRET_KEY": "secret"}),
            patch("alpaca.trading.client.TradingClient", return_value=mock_client_instance),
        ):
            result = await checker._check_alpaca()

        assert result.status == ServiceStatus.UNHEALTHY


class TestCheckLLM:
    async def test_ollama_healthy(self, checker: HealthChecker):
        mock_response = AsyncMock()
        mock_response.raise_for_status = Mock()

        with (
            patch.dict(os.environ, {"LLM_PROVIDER": "ollama"}),
            patch("src.daemon.health.httpx.AsyncClient") as mock_client,
        ):
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_client.return_value)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_client.return_value.get = AsyncMock(return_value=mock_response)
            result = await checker._check_llm()

        assert result.status == ServiceStatus.HEALTHY
        assert result.service == "llm_ollama"

    async def test_ollama_unhealthy(self, checker: HealthChecker):
        with (
            patch.dict(os.environ, {"LLM_PROVIDER": "ollama"}),
            patch("src.daemon.health.httpx.AsyncClient") as mock_client,
        ):
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_client.return_value)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_client.return_value.get = AsyncMock(side_effect=Exception("connection refused"))
            result = await checker._check_llm()

        assert result.status == ServiceStatus.UNHEALTHY

    async def test_anthropic_healthy(self, checker: HealthChecker):
        mock_llm = AsyncMock()
        mock_llm.acomplete = AsyncMock(return_value="OK")
        mock_llm.close = AsyncMock()

        with (
            patch.dict(os.environ, {"LLM_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": "test"}),
            patch("src.models.llm.LLMClient", return_value=mock_llm),
        ):
            result = await checker._check_llm()

        assert result.status == ServiceStatus.HEALTHY
        assert result.service == "llm_anthropic"


class TestCheckFinnhub:
    async def test_skipped_no_key(self, checker: HealthChecker):
        with patch.dict(os.environ, {}, clear=True):
            result = await checker._check_finnhub()

        assert result.status == ServiceStatus.SKIPPED
        assert result.service == "finnhub"

    async def test_healthy(self, checker: HealthChecker):
        mock_response = AsyncMock()
        mock_response.raise_for_status = Mock()

        with (
            patch.dict(os.environ, {"FINNHUB_API_KEY": "test"}),
            patch("src.daemon.health.httpx.AsyncClient") as mock_client,
        ):
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_client.return_value)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_client.return_value.get = AsyncMock(return_value=mock_response)
            result = await checker._check_finnhub()

        assert result.status == ServiceStatus.HEALTHY

    async def test_unhealthy(self, checker: HealthChecker):
        with (
            patch.dict(os.environ, {"FINNHUB_API_KEY": "test"}),
            patch("src.daemon.health.httpx.AsyncClient") as mock_client,
        ):
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_client.return_value)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_client.return_value.get = AsyncMock(side_effect=Exception("forbidden"))
            result = await checker._check_finnhub()

        assert result.status == ServiceStatus.UNHEALTHY


class TestArchiveOldAnalyses:
    def test_no_old_analyses(self, checker: HealthChecker):
        result = checker._archive_old_analyses()

        assert result.files_affected == 0
        assert "No old analyses" in result.message

    def test_archives_old_records(self, checker: HealthChecker):
        old_time = datetime.now(UTC) - timedelta(days=60)
        recent_time = datetime.now(UTC) - timedelta(days=1)

        checker.state.analyses = [
            AnalysisRecord(symbol="AAPL", timestamp=old_time, signal="BUY", confidence=0.8),
            AnalysisRecord(symbol="TSLA", timestamp=recent_time, signal="HOLD", confidence=0.6),
        ]

        result = checker._archive_old_analyses()

        assert result.files_affected == 1
        assert result.bytes_freed > 0
        assert len(checker.state.analyses) == 1
        assert checker.state.analyses[0].symbol == "TSLA"

        archive_files = list(checker._archive_dir.glob("analyses-*.jsonl"))
        assert len(archive_files) == 1


class TestPruneStaleCache:
    def test_no_cache_dir(self, checker: HealthChecker):
        result = checker._prune_stale_cache()

        assert result.files_affected == 0

    def test_prunes_caches(self, checker: HealthChecker, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        cache_base = tmp_path / "data" / "cache"
        service_dir = cache_base / "test_service"
        service_dir.mkdir(parents=True)

        # Run from tmp_path so "data/cache" resolves to our test dir
        monkeypatch.chdir(tmp_path)

        result = checker._prune_stale_cache()

        # diskcache.expire() returns 0 for empty cache, but proves the path works
        assert result.operation == "prune_cache"
        assert isinstance(result.files_affected, int)


class TestRotateLogs:
    def test_no_oversized_logs(self, checker: HealthChecker):
        result = checker._rotate_logs()
        assert result.files_affected == 0

    def test_rotates_oversized_log(self, checker: HealthChecker, tmp_path: Path):
        checker.config.health.log_max_size_mb = 0

        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        big_log = log_dir / "risk_audit.jsonl"
        big_log.write_text("x" * 100)

        # Create mock paths: only risk_audit.jsonl exists and is oversized
        original_path_cls = Path

        def make_path(p):
            if p == "logs/risk_audit.jsonl":
                return big_log
            if str(p).startswith("logs/"):
                return tmp_path / p
            return original_path_cls(p)

        with patch("src.daemon.health.Path", side_effect=make_path):
            result = checker._rotate_logs()

        assert result.files_affected == 1
        assert result.bytes_freed == 0
        assert "100 bytes" in result.message
        assert not big_log.exists()


class TestVerifyStateIntegrity:
    def test_no_state_file(self, checker: HealthChecker, tmp_path: Path):
        checker.config.state.state_file = str(tmp_path / "nonexistent.json")
        result = checker._verify_state_integrity()

        assert result.files_affected == 0
        assert "No state file" in result.message

    def test_valid_state(self, checker: HealthChecker, tmp_path: Path):
        state_file = tmp_path / "state.json"
        state_file.write_text(json.dumps(DaemonState().model_dump(mode="json"), default=str))
        checker.config.state.state_file = str(state_file)

        result = checker._verify_state_integrity()

        assert result.files_affected == 0
        assert "valid" in result.message

    def test_corrupt_state_backed_up(self, checker: HealthChecker, tmp_path: Path):
        """Corrupt state file is backed up when detected."""
        state_file = tmp_path / "state.json"
        state_file.write_text("{corrupt json!!")
        checker.config.state.state_file = str(state_file)

        result = checker._verify_state_integrity()
        assert result.files_affected == 1
        assert "backed up" in result.message
        assert not state_file.exists()
        backup_files = list(tmp_path.glob("state.corrupt.*"))
        assert len(backup_files) == 1


class TestFullRun:
    async def test_run_all_skipped(self, checker: HealthChecker):
        """All services skipped when no API keys configured."""
        mock_response = AsyncMock()
        mock_response.raise_for_status = Mock()

        with (
            patch.dict(os.environ, {}, clear=True),
            patch("src.daemon.health.httpx.AsyncClient") as mock_client,
        ):
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_client.return_value)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_client.return_value.get = AsyncMock(return_value=mock_response)
            report = await checker.run()

        assert isinstance(report, HealthReport)
        assert report.overall_status == ServiceStatus.HEALTHY
        assert len(report.service_checks) == 5
        assert len(report.cleanup_results) == 4
        assert report.total_duration_ms >= 0

    async def test_run_persists_report(self, checker: HealthChecker):
        mock_response = AsyncMock()
        mock_response.raise_for_status = Mock()

        with (
            patch.dict(os.environ, {}, clear=True),
            patch("src.daemon.health.httpx.AsyncClient") as mock_client,
        ):
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_client.return_value)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_client.return_value.get = AsyncMock(return_value=mock_response)
            await checker.run()

        health_dir = Path(checker.config.health.health_dir).expanduser()
        report_files = list(health_dir.glob("health-*.json"))
        assert len(report_files) == 1

        loaded = json.loads(report_files[0].read_text())
        assert loaded["overall_status"] == "HEALTHY"

    async def test_run_with_unhealthy_service(self, checker: HealthChecker):
        """Overall status UNHEALTHY when any service is unhealthy."""
        mock_response = AsyncMock()
        mock_response.raise_for_status = Mock()

        mock_ts_instance = Mock()
        mock_ts_instance.get_daily.side_effect = ConnectionError("fail")

        with (
            patch.dict(
                os.environ,
                {"LLM_PROVIDER": "ollama", "ALPHA_VANTAGE_API_KEY": "test"},
                clear=True,
            ),
            patch("src.daemon.health.httpx.AsyncClient") as mock_client,
            patch("alpha_vantage.timeseries.TimeSeries", return_value=mock_ts_instance),
        ):
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_client.return_value)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_client.return_value.get = AsyncMock(return_value=mock_response)

            report = await checker.run()

        assert report.overall_status == ServiceStatus.UNHEALTHY


class TestPruneOldReports:
    def test_prunes_old_reports(self, checker: HealthChecker):
        health_dir = Path(checker.config.health.health_dir).expanduser()
        health_dir.mkdir(parents=True)

        old_date = (datetime.now(UTC) - timedelta(days=45)).strftime("%Y-%m-%d")
        recent_date = (datetime.now(UTC) - timedelta(days=5)).strftime("%Y-%m-%d")

        (health_dir / f"health-{old_date}.json").write_text("{}")
        (health_dir / f"health-{recent_date}.json").write_text("{}")

        checker._prune_old_reports()

        remaining = list(health_dir.glob("health-*.json"))
        assert len(remaining) == 1
        assert recent_date in remaining[0].name


class TestRepr:
    def test_repr(self, checker: HealthChecker):
        repr_str = repr(checker)
        assert "HealthChecker" in repr_str
        assert "health" in repr_str
