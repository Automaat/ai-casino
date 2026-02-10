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


class TestOpenAISchemaProcessing:
    """Tests for OpenAI schema processing (_ensure_additional_properties_false)."""

    @pytest.fixture
    def provider(self, monkeypatch):
        """Create OpenAI provider instance."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        with patch("src.models.providers.openai.AsyncOpenAI"):
            from src.models.providers.openai import OpenAIProvider

            return OpenAIProvider(model="gpt-4o")

    def test_nested_objects_get_additional_properties_false(self, provider):
        """Test nested objects receive additionalProperties: false."""
        schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                },
                "address": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        }

        result = provider._ensure_additional_properties_false(schema)

        assert result["additionalProperties"] is False
        assert result["properties"]["user"]["additionalProperties"] is False
        assert result["properties"]["address"]["additionalProperties"] is False

    def test_array_of_objects_items_as_dict(self, provider):
        """Test items (dict) processed for array schemas."""
        schema = {
            "type": "array",
            "items": {"type": "object", "properties": {"id": {"type": "integer"}}},
        }

        result = provider._ensure_additional_properties_false(schema)

        assert result["items"]["additionalProperties"] is False

    def test_tuple_validation_items_as_list(self, provider):
        """Test items (list) processed for tuple validation."""
        schema = {
            "type": "array",
            "items": [
                {"type": "object", "properties": {"first": {"type": "string"}}},
                {"type": "object", "properties": {"second": {"type": "integer"}}},
            ],
        }

        result = provider._ensure_additional_properties_false(schema)

        assert result["items"][0]["additionalProperties"] is False
        assert result["items"][1]["additionalProperties"] is False

    def test_allof_anyof_oneof_combinators(self, provider):
        """Test allOf/anyOf/oneOf combinators processed."""
        schema = {
            "allOf": [
                {"type": "object", "properties": {"a": {"type": "string"}}},
                {"type": "object", "properties": {"b": {"type": "integer"}}},
            ],
            "anyOf": [{"type": "object", "properties": {"c": {"type": "boolean"}}}],
            "oneOf": [{"type": "object", "properties": {"d": {"type": "number"}}}],
        }

        result = provider._ensure_additional_properties_false(schema)

        assert result["allOf"][0]["additionalProperties"] is False
        assert result["allOf"][1]["additionalProperties"] is False
        assert result["anyOf"][0]["additionalProperties"] is False
        assert result["oneOf"][0]["additionalProperties"] is False

    def test_schemas_without_explicit_type_field(self, provider):
        """Test schemas with properties but no type field."""
        schema = {"properties": {"name": {"type": "string"}}}

        result = provider._ensure_additional_properties_false(schema)

        assert result["additionalProperties"] is False

    def test_circular_references_no_infinite_recursion(self, provider):
        """Test circular references don't cause infinite recursion."""
        # Create circular reference
        inner_schema: dict = {"type": "object", "properties": {}}
        schema = {"type": "object", "properties": {"child": inner_schema}}
        inner_schema["properties"]["parent"] = schema

        # Should not raise RecursionError
        result = provider._ensure_additional_properties_false(schema)

        # Both schemas should be processed
        assert result["additionalProperties"] is False
        assert result["properties"]["child"]["additionalProperties"] is False

    async def test_astructured_does_not_mutate_cached_schema(self, provider):
        """Test deep copy prevents mutation of cached schema."""

        class TestModel(BaseModel):
            """Test model."""

            value: str

        # Get original schema
        original_schema = TestModel.model_json_schema()
        original_keys = set(original_schema.keys())

        # Mock OpenAI response
        mock_message = MagicMock()
        mock_message.content = '{"value": "test"}'
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        provider._client.chat.completions.create = AsyncMock(return_value=mock_response)

        # Call astructured
        await provider.astructured([{"role": "user", "content": "test"}], TestModel)

        # Original schema should not be mutated
        current_schema = TestModel.model_json_schema()
        assert set(current_schema.keys()) == original_keys

    def test_all_properties_required(self, provider):
        """Test all properties are in required array (OpenAI strict mode)."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "email": {"type": "string"},
            },
        }

        result = provider._ensure_additional_properties_false(schema)

        assert "required" in result
        assert set(result["required"]) == {"name", "age", "email"}

    def test_nested_properties_all_required(self, provider):
        """Test nested object properties all required recursively."""
        schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                    },
                },
                "address": {
                    "type": "object",
                    "properties": {
                        "street": {"type": "string"},
                        "city": {"type": "string"},
                    },
                },
            },
        }

        result = provider._ensure_additional_properties_false(schema)

        # Top-level properties required
        assert set(result["required"]) == {"user", "address"}
        # Nested user properties required
        assert set(result["properties"]["user"]["required"]) == {"name", "age"}
        # Nested address properties required
        assert set(result["properties"]["address"]["required"]) == {"street", "city"}


class TestAnthropicProviderStructured:
    """Tests for Anthropic provider structured output."""

    @pytest.fixture
    def mock_anthropic_client(self):
        """Mock Anthropic async client."""
        with patch("src.models.providers.anthropic.AsyncAnthropic") as mock:
            client = MagicMock()
            mock.return_value = client
            yield client

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
