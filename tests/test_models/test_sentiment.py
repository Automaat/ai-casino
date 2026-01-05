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
        patch(
            "src.models.sentiment.AutoModelForSequenceClassification.from_pretrained"
        ) as mock_model,
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
