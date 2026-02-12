"""Tests for FinBERT sentiment analyzer."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from src.models.sentiment import FinBERTSentiment, SentimentScore


def test_sentiment_score_dominant():
    score = SentimentScore(positive=0.8, negative=0.1, neutral=0.1)
    assert score.dominant == "positive"

    score = SentimentScore(positive=0.1, negative=0.8, neutral=0.1)
    assert score.dominant == "negative"

    score = SentimentScore(positive=0.1, negative=0.1, neutral=0.8)
    assert score.dominant == "neutral"


def test_sentiment_score_score():
    score = SentimentScore(positive=0.8, negative=0.1, neutral=0.1)
    assert score.score == pytest.approx(0.7)

    score = SentimentScore(positive=0.2, negative=0.7, neutral=0.1)
    assert score.score == pytest.approx(-0.5)


@pytest.fixture
def mock_finbert():
    with (
        patch("src.models.sentiment.AutoTokenizer.from_pretrained") as mock_tokenizer,
        patch("src.models.sentiment.AutoModelForSequenceClassification.from_pretrained") as mock_model,
    ):
        tokenizer = MagicMock()
        model = MagicMock()

        mock_tokenizer.return_value = tokenizer
        mock_model.return_value = model

        tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

        outputs = MagicMock()
        outputs.logits = torch.tensor([[2.0, 0.5, 0.3]])
        model.return_value = outputs

        yield tokenizer, model


def test_finbert_init(mock_finbert):
    tokenizer, model = mock_finbert

    sentiment = FinBERTSentiment(device="cpu")

    assert sentiment.device == "cpu"
    assert sentiment.tokenizer == tokenizer
    assert sentiment.model == model
    model.to.assert_called_once_with("cpu")
    model.eval.assert_called_once()


def test_finbert_analyze(mock_finbert):
    tokenizer, model = mock_finbert

    sentiment = FinBERTSentiment(device="cpu")
    result = sentiment.analyze("Apple reports strong earnings")

    assert isinstance(result, SentimentScore)
    tokenizer.assert_called_once()
    model.assert_called_once()


def test_finbert_analyze_empty_text(mock_finbert):
    tokenizer, _ = mock_finbert

    sentiment = FinBERTSentiment(device="cpu")
    result = sentiment.analyze("")

    assert result.positive == 0.0
    assert result.negative == 0.0
    assert result.neutral == 1.0
    tokenizer.assert_not_called()


def test_finbert_analyze_batch(mock_finbert):
    tokenizer, model = mock_finbert

    sentiment = FinBERTSentiment(device="cpu")

    model.return_value.logits = torch.tensor(
        [
            [2.0, 0.5, 0.3],
            [0.3, 2.0, 0.5],
        ]
    )

    texts = ["Good news", "Bad news"]
    results = sentiment.analyze_batch(texts)

    assert len(results) == 2
    assert all(isinstance(r, SentimentScore) for r in results)
    tokenizer.assert_called_once()


def test_finbert_analyze_batch_empty():
    with (
        patch("src.models.sentiment.AutoTokenizer.from_pretrained"),
        patch("src.models.sentiment.AutoModelForSequenceClassification.from_pretrained"),
    ):
        sentiment = FinBERTSentiment(device="cpu")
        results = sentiment.analyze_batch([])

        assert results == []


def test_finbert_repr(mock_finbert):
    sentiment = FinBERTSentiment(device="cpu")
    assert repr(sentiment) == "FinBERTSentiment(device=cpu)"


def test_get_finbert_singleton_caching():
    """Verify singleton returns same instance across calls."""
    from src.models.sentiment import clear_finbert_sentiment, get_finbert_sentiment

    clear_finbert_sentiment()

    with (
        patch("src.models.sentiment.AutoTokenizer.from_pretrained"),
        patch("src.models.sentiment.AutoModelForSequenceClassification.from_pretrained"),
    ):
        instance1 = get_finbert_sentiment()
        instance2 = get_finbert_sentiment()

        assert instance1 is instance2

    clear_finbert_sentiment()


def test_get_finbert_device_parameter_first_call_only():
    """Device param only used on first initialization."""
    from src.models.sentiment import clear_finbert_sentiment, get_finbert_sentiment

    clear_finbert_sentiment()

    with (
        patch("src.models.sentiment.AutoTokenizer.from_pretrained"),
        patch("src.models.sentiment.AutoModelForSequenceClassification.from_pretrained"),
    ):
        instance1 = get_finbert_sentiment(device="cpu")
        assert instance1.device == "cpu"

        # Second call ignores device param
        instance2 = get_finbert_sentiment(device="cuda")
        assert instance2.device == "cpu"  # Still cpu
        assert instance1 is instance2

    clear_finbert_sentiment()


def test_get_finbert_thread_safety():
    """Verify thread safety of concurrent singleton initialization."""
    import concurrent.futures

    from src.models.sentiment import clear_finbert_sentiment, get_finbert_sentiment

    clear_finbert_sentiment()

    def get_instance():
        return get_finbert_sentiment()

    with (
        patch("src.models.sentiment.AutoTokenizer.from_pretrained"),
        patch("src.models.sentiment.AutoModelForSequenceClassification.from_pretrained"),
    ):
        # 10 threads simultaneously request instance
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(get_instance) for _ in range(10)]
            instances = [f.result() for f in futures]

        # All should get same instance
        assert all(inst is instances[0] for inst in instances)

    clear_finbert_sentiment()


def test_analyze_batch_worker():
    """Test worker function returns correct dict schema."""
    from src.models.sentiment import _analyze_batch_worker

    with (
        patch("src.models.sentiment.AutoTokenizer.from_pretrained"),
        patch("src.models.sentiment.AutoModelForSequenceClassification.from_pretrained") as mock_model,
    ):
        mock_model.return_value.return_value.logits = torch.tensor([[2.0, 0.5, 0.3], [0.3, 2.0, 0.5]])

        texts = ["Good news", "Bad news"]
        results = _analyze_batch_worker(texts, device="cpu")

        assert len(results) == 2
        assert all(isinstance(r, dict) for r in results)
        assert all(set(r.keys()) == {"positive", "negative", "neutral"} for r in results)
        assert all(isinstance(r["positive"], float) for r in results)


def test_shutdown_finbert_executor_idempotent():
    """Test executor shutdown is safe to call multiple times."""
    from src.models.sentiment import shutdown_finbert_executor

    # Should not raise on repeated calls
    shutdown_finbert_executor()
    shutdown_finbert_executor()
