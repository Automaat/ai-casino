"""FinBERT model inference wrapper (extracted from src/models/sentiment.py)."""

import asyncio
import threading

import torch
from loguru import logger
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import logging as hf_logging

from .models import SentimentScore

# Suppress transformers logging
hf_logging.set_verbosity_error()


class FinBERTInference:
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

        if self.tokenizer is None:
            msg = "Tokenizer failed to load"
            raise RuntimeError(msg)
        logger.info("FinBERT model loaded successfully")

    def analyze_batch(self, texts: list[str]) -> list[SentimentScore]:
        """Analyze sentiment of multiple texts (synchronous).

        Args:
            texts: List of financial texts to analyze

        Returns:
            List of SentimentScore objects
        """
        if not texts:
            return []

        # Filter empty texts
        texts = [t.strip() for t in texts if t.strip()]
        if not texts:
            return []

        with self._lock:
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

    async def analyze_batch_async(self, texts: list[str]) -> list[SentimentScore]:
        """Analyze sentiment asynchronously using thread offload.

        Args:
            texts: List of financial texts to analyze

        Returns:
            List of SentimentScore objects
        """
        return await asyncio.to_thread(self.analyze_batch, texts)

    def __repr__(self) -> str:
        """String representation."""
        return f"FinBERTInference(device={self.device})"
