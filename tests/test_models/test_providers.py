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

    async def test_astructured_returns_validated_model(self, mock_openai_client):
        """Test OpenAI astructured returns validated Pydantic model."""
        from src.models.providers.openai import OpenAIProvider

        # Mock response
        mock_message = MagicMock()
        mock_message.content = '{"answer": "42", "confidence": 0.95}'
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_response)

        # Pass API key explicitly (no env var fallback after refactoring)
        provider = OpenAIProvider(model="gpt-4o", api_key="test-key")
        result = await provider.astructured(
            [{"role": "user", "content": "test"}], SampleResponseModel, temperature=0.5
        )

        assert isinstance(result, SampleResponseModel)
        assert result.answer == "42"
        assert result.confidence == 0.95

    async def test_astructured_raises_on_validation_error(self, mock_openai_client):
        """Test OpenAI astructured raises StructuredOutputError on validation failure."""
        from src.models.providers.openai import OpenAIProvider

        mock_message = MagicMock()
        mock_message.content = '{"answer": "test", "confidence": 1.5}'
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_response)

        # Pass API key explicitly (no env var fallback after refactoring)
        provider = OpenAIProvider(model="gpt-4o", api_key="test-key")
        with pytest.raises(StructuredOutputError) as exc_info:
            await provider.astructured(
                [{"role": "user", "content": "test"}], SampleResponseModel, temperature=0.5
            )

        assert "Validation failed" in str(exc_info.value)
        assert exc_info.value.raw_response is not None

    def test_supports_structured_output(self, mock_openai_client):
        """Test OpenAI provider supports structured output."""
        _ = mock_openai_client  # Fixture needed for mock to work

        from src.models.providers.openai import OpenAIProvider

        # Pass API key explicitly (no env var fallback after refactoring)
        provider = OpenAIProvider(model="gpt-4o", api_key="test-key")
        assert provider.supports_structured_output is True


class TestOpenAISchemaProcessing:
    """Tests for OpenAI schema processing (_ensure_additional_properties_false)."""

    @pytest.fixture
    def provider(self):
        """Create OpenAI provider instance."""
        with patch("src.models.providers.openai.AsyncOpenAI"):
            from src.models.providers.openai import OpenAIProvider

            # Pass API key explicitly (no env var fallback after refactoring)
            return OpenAIProvider(model="gpt-4o", api_key="test-key")

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


class TestOpenAIJSONRepair:
    """Tests for OpenAI JSON repair logic."""

    @pytest.fixture
    def provider(self):
        """Create OpenAI provider instance."""
        with patch("src.models.providers.openai.AsyncOpenAI"):
            from src.models.providers.openai import OpenAIProvider

            return OpenAIProvider(model="gpt-4o", api_key="test-key")

    def test_repair_double_quotes_before_enum(self, provider):
        """Test repair fixes double quotes before enum values."""
        malformed = '{"universe": ""COMBINED", "criteria": "momentum"}'
        repaired = provider._repair_json(malformed)
        expected = '{"universe": "COMBINED", "criteria": "momentum"}'
        assert repaired == expected

    def test_repair_concatenated_values(self, provider):
        """Test repair removes concatenated uppercase words from enum values."""
        malformed = '{"criteria": "momentumCOMBINED"}'
        repaired = provider._repair_json(malformed)
        expected = '{"criteria": "momentum"}'
        assert repaired == expected

    def test_repair_duplicate_keys(self, provider):
        """Test repair removes duplicate keys (keeps last)."""
        malformed = '{"top_n": 10, "criteria": "value", "criteria": "momentum"}'
        repaired = provider._repair_json(malformed)
        # Should have single criteria key
        import json

        parsed = json.loads(repaired)
        assert parsed["criteria"] == "momentum"
        assert parsed["top_n"] == 10

    def test_repair_complex_malformed_json(self, provider):
        """Test repair handles multiple issues simultaneously."""
        malformed = (
            '{"top_n": 10, "universe": ""COMBINED", "criteria": "momentumCOMBINED", "criteria": "momentum"}'
        )
        repaired = provider._repair_json(malformed)
        import json

        # Should be valid JSON after repair
        parsed = json.loads(repaired)
        assert parsed["top_n"] == 10
        assert parsed["universe"] == "COMBINED"
        assert parsed["criteria"] == "momentum"

    def test_repair_clean_json_unchanged(self, provider):
        """Test repair leaves clean JSON unchanged."""
        clean = '{"top_n": 10, "universe": "COMBINED", "criteria": "momentum"}'
        repaired = provider._repair_json(clean)
        # Only whitespace normalization expected
        import json

        assert json.loads(repaired) == json.loads(clean)

    async def test_acomplete_with_tools_repairs_malformed_json(self, provider):
        """Test acomplete_with_tools attempts repair on malformed tool call JSON."""
        from src.models.providers.base import ToolCall

        # Mock malformed tool call
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "screen_stocks"
        mock_tool_call.function.arguments = (
            '{"top_n": 10, "universe": ""COMBINED", "criteria": "momentumCOMBINED", "criteria": "momentum"}'
        )

        mock_message = MagicMock()
        mock_message.tool_calls = [mock_tool_call]
        mock_message.content = None

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = None

        provider._client.chat.completions.create = AsyncMock(return_value=mock_response)

        # Should repair and succeed
        text, tool_calls = await provider.acomplete_with_tools(
            messages=[{"role": "user", "content": "test"}],
            tools=[{"type": "function", "function": {"name": "screen_stocks"}}],
        )

        assert text is None
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert isinstance(tool_calls[0], ToolCall)
        assert tool_calls[0].name == "screen_stocks"
        assert tool_calls[0].arguments["top_n"] == 10
        assert tool_calls[0].arguments["universe"] == "COMBINED"
        assert tool_calls[0].arguments["criteria"] == "momentum"

    def test_repair_nested_objects_with_duplicate_keys(self, provider):
        """Nested objects with duplicate keys should result in valid JSON."""
        import json

        # Duplicate keys in nested object
        raw = '{"outer": {"inner": {"key": "first", "key": "second"}}, "other": 1}'
        repaired = provider._repair_json(raw)
        data = json.loads(repaired)

        assert "outer" in data
        assert "inner" in data["outer"]
        # Standard JSON parsers keep the last value for duplicate keys
        assert data["outer"]["inner"]["key"] == "second"
        assert data["other"] == 1

    def test_repair_strings_with_escaped_quotes(self, provider):
        """String values with escaped quotes should remain valid after repair."""
        import json

        raw = r'{"message": "She said, \"hello\"", "ok": true}'
        repaired = provider._repair_json(raw)
        data = json.loads(repaired)

        assert data["ok"] is True
        assert data["message"] == 'She said, "hello"'

    def test_repair_multiple_malformations_combined(self, provider):
        """Combination of duplicate keys and minor quoting issues should be repairable."""
        import json

        raw = '{"config": {"mode": "fast", "mode": "slow"}, "description": ""Run"}'
        repaired = provider._repair_json(raw)
        data = json.loads(repaired)

        # Last duplicate key value should win
        assert data["config"]["mode"] == "slow"
        assert data["description"] == "Run"

    def test_repair_duplicate_key_at_end_without_trailing_comma(self, provider):
        """Duplicate key at the end of an object should still yield valid JSON."""
        import json

        raw = '{"a": 1, "b": 2, "b": 3}'
        repaired = provider._repair_json(raw)
        data = json.loads(repaired)

        assert data["a"] == 1
        # Last value for "b" is kept
        assert data["b"] == 3

    def test_repair_values_with_commas_and_braces(self, provider):
        """Values that contain commas/braces/brackets should not confuse repair logic."""
        import json

        raw = '{"pattern": "{[1,2,3], [4,5,6]}", "note": "a,b,c"}'
        repaired = provider._repair_json(raw)
        data = json.loads(repaired)

        assert data["pattern"] == "{[1,2,3], [4,5,6]}"
        assert data["note"] == "a,b,c"

    def test_repair_empty_objects_and_arrays(self, provider):
        """Empty objects and arrays should pass through unchanged."""
        import json

        raw = '{"empty_obj": {}, "empty_array": []}'
        repaired = provider._repair_json(raw)
        data = json.loads(repaired)

        assert data["empty_obj"] == {}
        assert data["empty_array"] == []


class TestAnthropicProviderStructured:
    """Tests for Anthropic provider structured output."""

    @pytest.fixture
    def mock_anthropic_client(self):
        """Mock Anthropic async client."""
        with patch("src.models.providers.anthropic.AsyncAnthropic") as mock:
            client = MagicMock()
            mock.return_value = client
            yield client

    async def test_astructured_returns_validated_model(self, mock_anthropic_client):
        """Test Anthropic astructured returns validated Pydantic model via tool use."""
        from src.models.providers.anthropic import AnthropicProvider

        # Mock tool use response
        mock_tool_block = MagicMock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "respond"
        mock_tool_block.input = {"answer": "test answer", "confidence": 0.8}
        mock_response = MagicMock()
        mock_response.content = [mock_tool_block]
        mock_anthropic_client.messages.create = AsyncMock(return_value=mock_response)

        # Pass API key explicitly (no env var fallback after refactoring)
        provider = AnthropicProvider(model="claude-sonnet-4-20250514", api_key="test-key")
        result = await provider.astructured(
            [{"role": "user", "content": "test"}], SampleResponseModel, temperature=0.5
        )

        assert isinstance(result, SampleResponseModel)
        assert result.answer == "test answer"
        assert result.confidence == 0.8

    async def test_astructured_raises_when_no_tool_block(self, mock_anthropic_client):
        """Test Anthropic astructured raises error when no tool_use block."""
        from src.models.providers.anthropic import AnthropicProvider

        # Mock text-only response (no tool use)
        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "Some text response"
        mock_response = MagicMock()
        mock_response.content = [mock_text_block]
        mock_anthropic_client.messages.create = AsyncMock(return_value=mock_response)

        # Pass API key explicitly (no env var fallback after refactoring)
        provider = AnthropicProvider(model="claude-sonnet-4-20250514", api_key="test-key")
        with pytest.raises(StructuredOutputError) as exc_info:
            await provider.astructured(
                [{"role": "user", "content": "test"}], SampleResponseModel, temperature=0.5
            )

        assert "No tool_use block" in str(exc_info.value)

    def test_supports_structured_output(self, mock_anthropic_client):
        """Test Anthropic provider supports structured output."""
        _ = mock_anthropic_client  # Fixture needed for mock to work

        from src.models.providers.anthropic import AnthropicProvider

        # Pass API key explicitly (no env var fallback after refactoring)
        provider = AnthropicProvider(model="claude-sonnet-4-20250514", api_key="test-key")
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
