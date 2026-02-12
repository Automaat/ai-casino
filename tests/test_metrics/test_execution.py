"""Tests for execution performance metrics."""

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.metrics.execution import (
    ExecutionMetricsCollector,
    LLMCallMetric,
    LLMUsageStats,
    SubOperationMetric,
    WorkflowExecutionMetrics,
    current_agent,
    current_collector,
    is_metrics_enabled,
    persist_jsonl,
    timed_operation,
)


class TestLLMUsageStats:
    """Tests for LLMUsageStats model."""

    def test_with_tokens(self):
        stats = LLMUsageStats(input_tokens=100, output_tokens=50)
        assert stats.input_tokens == 100
        assert stats.output_tokens == 50

    def test_defaults_to_none(self):
        stats = LLMUsageStats()
        assert stats.input_tokens is None
        assert stats.output_tokens is None


class TestLLMCallMetric:
    """Tests for LLMCallMetric model."""

    def test_successful_call(self):
        metric = LLMCallMetric(
            timestamp="2024-01-01T00:00:00Z",
            agent_name="technical",
            method="acomplete",
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            latency_ms=1234.5,
            input_tokens=100,
            output_tokens=50,
            estimated_cost_usd=0.001,
            success=True,
        )
        assert metric.success is True
        assert metric.error is None
        assert metric.latency_ms == 1234.5

    def test_failed_call(self):
        metric = LLMCallMetric(
            timestamp="2024-01-01T00:00:00Z",
            agent_name="news",
            method="astructured",
            provider="openai",
            model="gpt-4o",
            latency_ms=500.0,
            success=False,
            error="Rate limit exceeded",
        )
        assert metric.success is False
        assert metric.error == "Rate limit exceeded"
        assert metric.input_tokens is None


class TestSubOperationMetric:
    """Tests for SubOperationMetric model."""

    def test_with_metadata(self):
        metric = SubOperationMetric(
            name="market_data_fetch",
            latency_ms=1200.0,
            metadata={"source": "yfinance", "rows": 90},
        )
        assert metric.name == "market_data_fetch"
        assert metric.metadata is not None
        assert metric.metadata["source"] == "yfinance"

    def test_without_metadata(self):
        metric = SubOperationMetric(name="finbert_inference", latency_ms=300.0)
        assert metric.metadata is None


class TestExecutionMetricsCollector:
    """Tests for ExecutionMetricsCollector."""

    def test_record_and_finalize(self):
        collector = ExecutionMetricsCollector("AAPL", "anthropic", "claude-sonnet-4-20250514")

        # Set agent context
        token = current_agent.set("technical")
        try:
            collector.record_llm_call(
                method="acomplete",
                latency_ms=1000.0,
                usage=LLMUsageStats(input_tokens=500, output_tokens=200),
                success=True,
            )
            collector.record_llm_call(
                method="astructured",
                latency_ms=800.0,
                usage=LLMUsageStats(input_tokens=300, output_tokens=100),
                success=True,
            )
        finally:
            current_agent.reset(token)

        collector.record_agent_timing("technical", 1800.0)
        collector.record_pipeline_stage("analyses", 5000.0)
        collector.record_sub_operation("market_data_fetch", 1200.0, {"source": "yfinance"})

        result = collector.finalize()

        assert isinstance(result, WorkflowExecutionMetrics)
        assert result.symbol == "AAPL"
        assert result.provider == "anthropic"
        assert result.model == "claude-sonnet-4-20250514"
        assert len(result.llm_calls) == 2
        assert result.total_input_tokens == 800
        assert result.total_output_tokens == 300
        assert result.total_latency_ms > 0
        assert len(result.agent_timings) == 1
        assert result.agent_timings[0].llm_calls == 2
        assert len(result.pipeline_stages) == 1
        assert len(result.sub_operations) == 1

    def test_cost_estimation_known_model(self):
        cost = ExecutionMetricsCollector._estimate_cost(
            "anthropic", "claude-sonnet-4-20250514", 1_000_000, 1_000_000
        )
        # $3/M input + $15/M output = $18
        assert cost == pytest.approx(18.0)

    def test_cost_estimation_unknown_model(self):
        cost = ExecutionMetricsCollector._estimate_cost("local", "my-model", 1000, 500)
        assert cost is None

    def test_cost_estimation_no_tokens(self):
        cost = ExecutionMetricsCollector._estimate_cost("anthropic", "claude-sonnet-4-20250514", None, None)
        assert cost is None

    def test_empty_collector(self):
        collector = ExecutionMetricsCollector("TSLA", "ollama", "qwen3:14b")
        result = collector.finalize()
        assert result.total_input_tokens == 0
        assert result.total_output_tokens == 0
        assert result.total_estimated_cost_usd == 0.0
        assert len(result.llm_calls) == 0

    def test_record_sub_operation(self):
        collector = ExecutionMetricsCollector("AAPL", "anthropic", "claude-sonnet-4-20250514")
        collector.record_sub_operation("finbert_inference", 350.0, {"batch_size": 10})
        collector.record_sub_operation("pandas_ta_indicators", 50.0, {"rows": 90})

        result = collector.finalize()
        assert len(result.sub_operations) == 2
        assert result.sub_operations[0].name == "finbert_inference"
        assert result.sub_operations[0].metadata is not None
        assert result.sub_operations[0].metadata["batch_size"] == 10
        assert result.sub_operations[1].name == "pandas_ta_indicators"


class TestPersistJsonl:
    """Tests for JSONL persistence."""

    def test_persist_creates_file(self, tmp_path):
        collector = ExecutionMetricsCollector("AAPL", "anthropic", "claude-sonnet-4-20250514")
        metrics = collector.finalize()

        output_path = str(tmp_path / "metrics.jsonl")
        persist_jsonl(metrics, path=output_path)

        with Path(output_path).open() as f:
            lines = f.readlines()
        assert len(lines) == 1

        data = json.loads(lines[0])
        assert data["symbol"] == "AAPL"
        assert data["provider"] == "anthropic"

    def test_persist_appends(self, tmp_path):
        output_path = str(tmp_path / "metrics.jsonl")

        for symbol in ("AAPL", "TSLA"):
            collector = ExecutionMetricsCollector(symbol, "anthropic", "claude-sonnet-4-20250514")
            persist_jsonl(collector.finalize(), path=output_path)

        with Path(output_path).open() as f:
            lines = f.readlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["symbol"] == "AAPL"
        assert json.loads(lines[1])["symbol"] == "TSLA"


class TestIsMetricsEnabled:
    """Tests for is_metrics_enabled()."""

    def test_enabled_by_default(self, monkeypatch):
        monkeypatch.delenv("EXECUTION_METRICS", raising=False)
        assert is_metrics_enabled() is True

    def test_enabled_true(self, monkeypatch):
        monkeypatch.setenv("EXECUTION_METRICS", "true")
        assert is_metrics_enabled() is True

    def test_enabled_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("EXECUTION_METRICS", "True")
        assert is_metrics_enabled() is True

    def test_disabled_false(self, monkeypatch):
        monkeypatch.setenv("EXECUTION_METRICS", "false")
        assert is_metrics_enabled() is False

    def test_disabled_random_value(self, monkeypatch):
        monkeypatch.setenv("EXECUTION_METRICS", "yes")
        assert is_metrics_enabled() is False


class TestTimedOperation:
    """Tests for timed_operation context manager."""

    def test_noop_without_collector(self):
        """timed_operation should be a no-op when no collector is active."""
        # Smoke test: verifies no exception when no collector is active
        token = current_collector.set(None)
        try:
            with timed_operation("test_op", key="value"):
                pass
            assert True  # Reached here without exception
        finally:
            current_collector.reset(token)

    def test_records_with_collector(self):
        collector = ExecutionMetricsCollector("AAPL", "anthropic", "claude-sonnet-4-20250514")
        token = current_collector.set(collector)
        try:
            with timed_operation("test_op", source="test"):
                pass
        finally:
            current_collector.reset(token)

        result = collector.finalize()
        assert len(result.sub_operations) == 1
        assert result.sub_operations[0].name == "test_op"
        assert result.sub_operations[0].latency_ms >= 0
        assert result.sub_operations[0].metadata is not None
        assert result.sub_operations[0].metadata["source"] == "test"

    def test_records_timing(self):
        import time

        collector = ExecutionMetricsCollector("AAPL", "anthropic", "claude-sonnet-4-20250514")
        token = current_collector.set(collector)
        try:
            with timed_operation("slow_op"):
                time.sleep(0.05)
        finally:
            current_collector.reset(token)

        result = collector.finalize()
        assert result.sub_operations[0].latency_ms >= 40  # ~50ms sleep


class TestContextVarPropagation:
    """Tests for ContextVar async propagation."""

    async def test_agent_context_propagates(self):
        """Test current_agent ContextVar works correctly in async context."""
        collector = ExecutionMetricsCollector("AAPL", "anthropic", "claude-sonnet-4-20250514")

        async def simulate_agent(name: str) -> None:
            token = current_agent.set(name)
            try:
                collector.record_llm_call(
                    method="acomplete",
                    latency_ms=100.0,
                    usage=LLMUsageStats(input_tokens=10, output_tokens=5),
                    success=True,
                )
            finally:
                current_agent.reset(token)

        async with asyncio.TaskGroup() as tg:
            tg.create_task(simulate_agent("technical"))
            tg.create_task(simulate_agent("news"))

        result = collector.finalize()
        agent_names = {call.agent_name for call in result.llm_calls}
        assert "technical" in agent_names
        assert "news" in agent_names

    async def test_collector_contextvar_propagates(self):
        """Test current_collector ContextVar works correctly."""
        collector = ExecutionMetricsCollector("AAPL", "anthropic", "claude-sonnet-4-20250514")
        token = current_collector.set(collector)
        try:
            with timed_operation("async_op"):
                await asyncio.sleep(0.01)
        finally:
            current_collector.reset(token)

        result = collector.finalize()
        assert len(result.sub_operations) == 1
        assert result.sub_operations[0].name == "async_op"


class TestProviderUsageCapture:
    """Tests that providers capture _last_usage."""

    async def test_anthropic_captures_usage(self):
        from unittest.mock import AsyncMock, patch

        with patch("src.models.providers.anthropic.AsyncAnthropic") as mock_cls:
            client = MagicMock()
            mock_cls.return_value = client

            mock_response = MagicMock()
            mock_response.content = [MagicMock(text="hello", type="text")]
            mock_response.usage.input_tokens = 100
            mock_response.usage.output_tokens = 50
            client.messages.create = AsyncMock(return_value=mock_response)

            from src.models.providers.anthropic import AnthropicProvider

            # Pass API key explicitly (no env var fallback after refactoring)
            provider = AnthropicProvider(model="claude-sonnet-4-20250514", api_key="test-key")
            await provider.acomplete([{"role": "user", "content": "test"}])

            assert provider.last_usage is not None
            assert provider.last_usage.input_tokens == 100
            assert provider.last_usage.output_tokens == 50

    async def test_openai_captures_usage(self):
        from unittest.mock import AsyncMock, patch

        with patch("src.models.providers.openai.AsyncOpenAI") as mock_cls:
            client = MagicMock()
            mock_cls.return_value = client

            mock_response = MagicMock()
            mock_response.choices = [MagicMock(message=MagicMock(content="hello"))]
            mock_response.usage.prompt_tokens = 200
            mock_response.usage.completion_tokens = 80
            client.chat.completions.create = AsyncMock(return_value=mock_response)

            from src.models.providers.openai import OpenAIProvider

            # Pass API key explicitly (no env var fallback after refactoring)
            provider = OpenAIProvider(model="gpt-4o", api_key="test-key")
            await provider.acomplete([{"role": "user", "content": "test"}])

            assert provider.last_usage is not None
            assert provider.last_usage.input_tokens == 200
            assert provider.last_usage.output_tokens == 80

    async def test_ollama_captures_usage(self):
        from unittest.mock import patch

        with patch("src.models.providers.ollama.httpx.Client") as mock_cls:
            client = MagicMock()
            mock_cls.return_value = client

            mock_response = MagicMock()
            mock_response.json.return_value = {
                "message": {"content": "hello"},
                "prompt_eval_count": 150,
                "eval_count": 60,
            }
            mock_response.raise_for_status = MagicMock()
            client.post.return_value = mock_response

            from src.models.providers.ollama import OllamaProvider

            provider = OllamaProvider(model="qwen3:14b")
            await provider.acomplete([{"role": "user", "content": "test"}])

            assert provider.last_usage is not None
            assert provider.last_usage.input_tokens == 150
            assert provider.last_usage.output_tokens == 60
