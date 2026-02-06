"""FinBERT sentiment analysis model wrapper."""

# NOTE: This module intentionally imports torch at module level (it USES torch for inference).
# Environment config handled by src/models/torch_config.py before import cascade reaches here.

import threading

import torch
from loguru import logger
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import logging as hf_logging

from src.metrics.execution import timed_operation

# Suppress transformers logging (the env var alone doesn't catch everything)
hf_logging.set_verbosity_error()

# Module-level singleton state
_finbert_instance: "FinBERTSentiment | None" = None
_finbert_lock = threading.Lock()


class SentimentScore(BaseModel):
    """Sentiment analysis result."""

    positive: float
    negative: float
    neutral: float

    @property
    def dominant(self) -> str:
        """Get dominant sentiment label."""
        if self.positive > self.negative and self.positive > self.neutral:
            return "positive"
        if self.negative > self.positive and self.negative > self.neutral:
            return "negative"
        return "neutral"

    @property
    def score(self) -> float:
        """Get overall sentiment score (-1 to 1)."""
        return self.positive - self.negative


class FinBERTSentiment:
    """FinBERT sentiment analyzer for financial text."""

    MODEL_NAME = "ProsusAI/finbert"

    def __init__(self, device: str | None = None) -> None:
        """Initialize FinBERT model.

        Args:
            device: Device for inference (cuda/cpu). Auto-detect if None.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading FinBERT model on {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_NAME)
        self.model.to(self.device)
        self.model.eval()
        self._lock = threading.Lock()

        logger.info("FinBERT model loaded successfully")

    def analyze(self, text: str) -> SentimentScore:
        """Analyze sentiment of financial text.

        Args:
            text: Financial text to analyze

        Returns:
            SentimentScore with positive/negative/neutral probabilities
        """
        if not text.strip():
            logger.warning("Empty text provided for sentiment analysis")
            return SentimentScore(positive=0.0, negative=0.0, neutral=1.0)

        with self._lock:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]

        labels = ["positive", "negative", "neutral"]
        result = SentimentScore(**{labels[i]: float(probs[i]) for i in range(3)})

        logger.debug(f"Sentiment: {result.dominant} (score={result.score:.3f})")
        return result

    def analyze_batch(self, texts: list[str]) -> list[SentimentScore]:
        """Analyze sentiment of multiple texts.

        Args:
            texts: List of financial texts to analyze

        Returns:
            List of SentimentScore objects
        """
        if not texts:
            return []

        texts = [t.strip() for t in texts if t.strip()]
        if not texts:
            return []

        with timed_operation("finbert_inference", batch_size=len(texts)):
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

        labels = ["positive", "negative", "neutral"]
        results = [
            SentimentScore(**{labels[i]: float(probs[j][i]) for i in range(3)}) for j in range(len(texts))
        ]

        logger.debug(f"Analyzed {len(results)} texts")
        return results

    def __repr__(self) -> str:
        """String representation."""
        return f"FinBERTSentiment(device={self.device})"


def get_finbert_sentiment(device: str | None = None) -> FinBERTSentiment:
    """Get or create singleton FinBERT sentiment analyzer.

    Lazy-loads model on first call. Subsequent calls return cached instance.
    Thread-safe. Device parameter only used on first initialization.

    Args:
        device: Device for inference (cuda/cpu). Auto-detect if None.
                Only used on first call; ignored on subsequent calls.

    Returns:
        FinBERTSentiment singleton instance
    """
    global _finbert_instance  # noqa: PLW0603

    # Fast path: already initialized (no lock needed)
    if _finbert_instance is not None:
        return _finbert_instance

    # Slow path: first call, acquire lock
    with _finbert_lock:
        # Double-check after lock (another thread may have initialized)
        if _finbert_instance is not None:
            return _finbert_instance

        _finbert_instance = FinBERTSentiment(device=device)
        logger.info(f"FinBERT singleton initialized on {_finbert_instance.device}")
        return _finbert_instance


def clear_finbert_sentiment() -> None:
    """Clear cached FinBERT instance (for testing/cleanup)."""
    global _finbert_instance  # noqa: PLW0603

    with _finbert_lock:
        _finbert_instance = None
        logger.debug("FinBERT singleton cleared")
