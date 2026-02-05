"""Tests for LLM providers."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from src.models.providers.base import StructuredOutputError


class SampleResponseModel(BaseModel):
    """Sample response model for testing structured output."""

    answer: str = Field(description="The answer")
    confidence: float = Field(description="Confidence score", ge=0.0, le=1.0)


class TestOpenAIProviderStructured:
    """Tests for OpenAI provider structured output."""

    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI async client."""
        with patch("src.models.providers.openai.AsyncOpenAI") as mock:
            client = MagicMock()
            mock.return_value = client
            yield client

    @pytest.mark.asyncio
    async def test_astructured_returns_validated_model(self, mock_openai_client, monkeypatch):
        """Test OpenAI astructured returns validated Pydantic model."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        from src.models.providers.openai import OpenAIProvider

        # Mock response
        mock_message = MagicMock()
        mock_message.content = '{"answer": "42", "confidence": 0.95}'
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_response)

        provider = OpenAIProvider(model="gpt-4o")
        result = await provider.astructured(
            [{"role": "user", "content": "test"}], SampleResponseModel, temperature=0.5
        )

        assert isinstance(result, SampleResponseModel)
        assert result.answer == "42"
        assert result.confidence == 0.95

    @pytest.mark.asyncio
    async def test_astructured_raises_on_validation_error(self, mock_openai_client, monkeypatch):
        """Test OpenAI astructured raises StructuredOutputError on validation failure."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        from src.models.providers.openai import OpenAIProvider

        mock_message = MagicMock()
        mock_message.content = '{"answer": "test", "confidence": 1.5}'
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_response)

        provider = OpenAIProvider(model="gpt-4o")
        with pytest.raises(StructuredOutputError) as exc_info:
            await provider.astructured(
                [{"role": "user", "content": "test"}], SampleResponseModel, temperature=0.5
            )

        assert "Validation failed" in str(exc_info.value)
        assert exc_info.value.raw_response is not None

    def test_supports_structured_output(self, mock_openai_client, monkeypatch):
        """Test OpenAI provider supports structured output."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        _ = mock_openai_client  # Fixture needed for mock to work

        from src.models.providers.openai import OpenAIProvider

        provider = OpenAIProvider(model="gpt-4o")
        assert provider.supports_structured_output is True


class TestAnthropicProviderStructured:
    """Tests for Anthropic provider structured output."""

    @pytest.fixture
    def mock_anthropic_client(self):
        """Mock Anthropic async client."""
        with patch("src.models.providers.anthropic.AsyncAnthropic") as mock:
            client = MagicMock()
            mock.return_value = client
            yield client

    @pytest.mark.asyncio
    async def test_astructured_returns_validated_model(self, mock_anthropic_client, monkeypatch):
        """Test Anthropic astructured returns validated Pydantic model via tool use."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        from src.models.providers.anthropic import AnthropicProvider

        # Mock tool use response
        mock_tool_block = MagicMock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "respond"
        mock_tool_block.input = {"answer": "test answer", "confidence": 0.8}
        mock_response = MagicMock()
        mock_response.content = [mock_tool_block]
        mock_anthropic_client.messages.create = AsyncMock(return_value=mock_response)

        provider = AnthropicProvider(model="claude-sonnet-4-20250514")
        result = await provider.astructured(
            [{"role": "user", "content": "test"}], SampleResponseModel, temperature=0.5
        )

        assert isinstance(result, SampleResponseModel)
        assert result.answer == "test answer"
        assert result.confidence == 0.8

    @pytest.mark.asyncio
    async def test_astructured_raises_when_no_tool_block(self, mock_anthropic_client, monkeypatch):
        """Test Anthropic astructured raises error when no tool_use block."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        from src.models.providers.anthropic import AnthropicProvider

        # Mock text-only response (no tool use)
        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "Some text response"
        mock_response = MagicMock()
        mock_response.content = [mock_text_block]
        mock_anthropic_client.messages.create = AsyncMock(return_value=mock_response)

        provider = AnthropicProvider(model="claude-sonnet-4-20250514")
        with pytest.raises(StructuredOutputError) as exc_info:
            await provider.astructured(
                [{"role": "user", "content": "test"}], SampleResponseModel, temperature=0.5
            )

        assert "No tool_use block" in str(exc_info.value)

    def test_supports_structured_output(self, mock_anthropic_client, monkeypatch):
        """Test Anthropic provider supports structured output."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        _ = mock_anthropic_client  # Fixture needed for mock to work

        from src.models.providers.anthropic import AnthropicProvider

        provider = AnthropicProvider(model="claude-sonnet-4-20250514")
        assert provider.supports_structured_output is True


class TestOllamaProviderStructured:
    """Tests for Ollama provider structured output."""

    @pytest.fixture
    def mock_httpx_client(self):
        """Mock httpx client for Ollama."""
        with patch("src.models.providers.ollama.httpx.Client") as mock:
            client = MagicMock()
            mock.return_value = client
            yield client

    @pytest.mark.asyncio
    async def test_astructured_returns_validated_model(self, mock_httpx_client):
        """Test Ollama astructured returns validated Pydantic model."""
        from src.models.providers.ollama import OllamaProvider

        # Mock successful JSON response
        mock_response = MagicMock()
        mock_response.json.return_value = {"message": {"content": '{"answer": "test", "confidence": 0.75}'}}
        mock_response.raise_for_status = MagicMock()
        mock_httpx_client.post.return_value = mock_response

        provider = OllamaProvider(model="qwen3:14b")
        result = await provider.astructured(
            [{"role": "user", "content": "test"}], SampleResponseModel, temperature=0.5
        )

        assert isinstance(result, SampleResponseModel)
        assert result.answer == "test"
        assert result.confidence == 0.75

    @pytest.mark.asyncio
    async def test_astructured_retries_on_validation_error(self, mock_httpx_client):
        """Test Ollama astructured retries once on validation failure."""
        from src.models.providers.ollama import OllamaProvider

        # First response invalid (missing confidence), second valid
        invalid_response = MagicMock()
        invalid_response.json.return_value = {"message": {"content": '{"answer": "test"}'}}
        invalid_response.raise_for_status = MagicMock()

        valid_response = MagicMock()
        valid_response.json.return_value = {"message": {"content": '{"answer": "test", "confidence": 0.8}'}}
        valid_response.raise_for_status = MagicMock()

        mock_httpx_client.post.side_effect = [invalid_response, valid_response]

        provider = OllamaProvider(model="qwen3:14b")
        result = await provider.astructured(
            [{"role": "user", "content": "test"}], SampleResponseModel, temperature=0.5
        )

        assert isinstance(result, SampleResponseModel)
        assert result.answer == "test"
        assert result.confidence == 0.8
        assert mock_httpx_client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_astructured_raises_after_retries_exhausted(self, mock_httpx_client):
        """Test Ollama astructured raises StructuredOutputError after retries."""
        from src.models.providers.ollama import OllamaProvider

        # Both responses invalid
        invalid_response = MagicMock()
        invalid_response.json.return_value = {"message": {"content": '{"answer": "test"}'}}
        invalid_response.raise_for_status = MagicMock()

        mock_httpx_client.post.return_value = invalid_response

        provider = OllamaProvider(model="qwen3:14b")
        with pytest.raises(StructuredOutputError) as exc_info:
            await provider.astructured(
                [{"role": "user", "content": "test"}], SampleResponseModel, temperature=0.5
            )

        assert "Validation failed after 2 attempts" in str(exc_info.value)
        assert mock_httpx_client.post.call_count == 2

    def test_supports_structured_output(self, mock_httpx_client):
        """Test Ollama provider supports structured output."""
        _ = mock_httpx_client  # Fixture needed for mock to work

        from src.models.providers.ollama import OllamaProvider

        provider = OllamaProvider(model="qwen3:14b")
        assert provider.supports_structured_output is True


class TestStructuredOutputError:
    """Tests for StructuredOutputError exception."""

    def test_error_with_raw_response(self):
        """Test error stores raw response."""
        error = StructuredOutputError("Test error", raw_response='{"invalid": "json"}')

        assert str(error) == "Test error"
        assert error.raw_response == '{"invalid": "json"}'

    def test_error_without_raw_response(self):
        """Test error works without raw response."""
        error = StructuredOutputError("Test error")

        assert str(error) == "Test error"
        assert error.raw_response is None
