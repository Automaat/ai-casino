"""Tests for LLM client."""

from unittest.mock import MagicMock, patch

import pytest

from src.models.llm import LLMClient


@pytest.fixture
def mock_completion():
    with patch("src.models.llm.completion") as mock:
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = "Mocked response"
        mock.return_value = response
        yield mock


def test_llm_client_init_ollama(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("LLM_MODEL", "qwen3:14b")

    client = LLMClient()

    assert client.provider == "ollama"
    assert client.model == "qwen3:14b"
    assert client._model_id == "ollama/qwen3:14b"


def test_llm_client_init_anthropic(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_MODEL", "claude-sonnet-4-20250514")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    client = LLMClient()

    assert client.provider == "anthropic"
    assert client.model == "claude-sonnet-4-20250514"
    assert client._model_id == "anthropic/claude-sonnet-4-20250514"


def test_llm_client_init_openai(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    client = LLMClient()

    assert client.provider == "openai"
    assert client.model == "gpt-4o"
    assert client._model_id == "openai/gpt-4o"


def test_llm_client_unsupported_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "invalid")

    with pytest.raises(ValueError, match="Unsupported provider: invalid"):
        LLMClient()


def test_complete_with_system_prompt(mock_completion, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    client = LLMClient()

    result = client.complete("Test prompt", system="System message", temperature=0.5)

    assert result == "Mocked response"
    mock_completion.assert_called_once()
    call_args = mock_completion.call_args
    assert call_args.kwargs["messages"] == [
        {"role": "system", "content": "System message"},
        {"role": "user", "content": "Test prompt"},
    ]
    assert call_args.kwargs["temperature"] == 0.5


def test_complete_without_system_prompt(mock_completion, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    client = LLMClient()

    result = client.complete("Test prompt")

    assert result == "Mocked response"
    call_args = mock_completion.call_args
    assert call_args.kwargs["messages"] == [
        {"role": "user", "content": "Test prompt"},
    ]


def test_chat(mock_completion, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    client = LLMClient()

    messages = [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
        {"role": "user", "content": "How are you?"},
    ]

    result = client.chat(messages, temperature=0.3)

    assert result == "Mocked response"
    mock_completion.assert_called_once()
    call_args = mock_completion.call_args
    assert call_args.kwargs["messages"] == messages
    assert call_args.kwargs["temperature"] == 0.3


def test_complete_handles_exception(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    client = LLMClient()

    with patch("src.models.llm.completion", side_effect=Exception("API Error")):
        with pytest.raises(Exception, match="API Error"):
            client.complete("Test prompt")


def test_repr(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("LLM_MODEL", "qwen3:14b")

    client = LLMClient()

    assert repr(client) == "LLMClient(provider=ollama, model=qwen3:14b)"
