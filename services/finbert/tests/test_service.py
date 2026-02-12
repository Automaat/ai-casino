"""Tests for FinBERT FastAPI service."""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

from services.finbert.src.main import app
from services.finbert.src.models import SentimentScore


@pytest.fixture
def mock_finbert_model():
    """Mock FinBERTInference for testing."""
    mock_model = Mock()
    mock_model.device = "cpu"
    mock_model.analyze_batch_async = AsyncMock(
        return_value=[
            SentimentScore(positive=0.8, negative=0.1, neutral=0.1),
            SentimentScore(positive=0.2, negative=0.7, neutral=0.1),
        ]
    )
    return mock_model


@pytest.fixture
def client(mock_finbert_model):
    """TestClient with mocked model."""
    with patch("src.main.finbert_model", mock_finbert_model):
        yield TestClient(app)


def test_health_endpoint_model_loaded(client, mock_finbert_model):
    """Test health endpoint when model is loaded."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["device"] == "cpu"
    assert data["uptime_seconds"] >= 0.0


def test_health_endpoint_model_not_loaded():
    """Test health endpoint when model is not loaded."""
    with patch("src.main.finbert_model", None):
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["model_loaded"] is False


@pytest.mark.asyncio
async def test_analyze_batch_success(client, mock_finbert_model):
    """Test successful batch analysis."""
    response = client.post(
        "/analyze",
        json={"texts": ["Strong earnings beat expectations", "Weak guidance disappoints investors"]},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["scores"]) == 2
    assert data["batch_size"] == 2
    assert all(0.0 <= s["positive"] <= 1.0 for s in data["scores"])
    assert all(0.0 <= s["negative"] <= 1.0 for s in data["scores"])
    assert all(0.0 <= s["neutral"] <= 1.0 for s in data["scores"])
    assert data["inference_time_ms"] >= 0.0


def test_analyze_empty_batch_validation_error(client):
    """Test validation error for empty batch."""
    response = client.post("/analyze", json={"texts": []})
    assert response.status_code == 422


def test_analyze_too_many_texts_validation_error(client):
    """Test validation error for batch > 100 texts."""
    response = client.post("/analyze", json={"texts": ["text"] * 101})
    assert response.status_code == 422


def test_analyze_model_not_loaded():
    """Test error when model not loaded."""
    with patch("src.main.finbert_model", None):
        client = TestClient(app)
        response = client.post("/analyze", json={"texts": ["test"]})
        assert response.status_code == 503
        assert "Model not loaded" in response.json()["detail"]


@pytest.mark.asyncio
async def test_analyze_inference_failure(client, mock_finbert_model):
    """Test error handling when inference fails."""
    mock_finbert_model.analyze_batch_async.side_effect = RuntimeError("Inference failed")

    response = client.post("/analyze", json={"texts": ["test"]})
    assert response.status_code == 500
    assert "Inference failed" in response.json()["detail"]
