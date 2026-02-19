"""FinBERT sentiment analysis model wrapper."""

# NOTE: This module intentionally imports torch at module level (it USES torch for inference).
# Environment config handled by src/models/torch_config.py before import cascade reaches here.

import atexit
import os
import threading
from concurrent.futures import ProcessPoolExecutor
from typing import Protocol, cast

import torch
from loguru import logger
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import logging as hf_logging

from src.metrics.execution import timed_operation

# Suppress transformers logging (the env var alone doesn't catch everything)
hf_logging.set_verbosity_error()

# Feature flag for FinBERT mode (local in-process vs remote microservice)
FINBERT_MODE = os.getenv("FINBERT_MODE", "local")
FINBERT_SERVICE_URL = os.getenv("FINBERT_SERVICE_URL", "http://localhost:8485")
FINBERT_WORKERS = os.getenv("FINBERT_WORKERS")
MAX_WORKERS = 32
DEFAULT_WORKERS = 4


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


class FinBERTProtocol(Protocol):
    """Protocol defining FinBERT interface (local or remote)."""

    def analyze_batch(self, texts: list[str]) -> list[SentimentScore]:
        """Analyze sentiment of multiple texts."""
        ...


# Process pool executor holder for parallel FinBERT inference (avoids GIL)
class _ExecutorHolder:
    """Holder for process pool executor (allows recreation after shutdown)."""

    executor: ProcessPoolExecutor | None = None
    lock = threading.Lock()


class _FinBERTHolder:
    """Singleton holder for FinBERT instance (local or remote)."""

    instance: FinBERTProtocol | None = None
    lock = threading.Lock()


class FinBERTSentiment(FinBERTProtocol):
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
            inputs = self.tokenizer(  # type: ignore[misc]
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
            inputs = self.tokenizer(  # type: ignore[misc]
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


def _analyze_batch_worker(texts: list[str], device: str | None = None) -> list[dict[str, float]]:
    """Worker function for ProcessPoolExecutor to analyze sentiment batch.

    This function reconstructs FinBERT in the worker process (singleton per worker).
    Returns plain dicts instead of SentimentScore to ensure picklability.

    Args:
        texts: List of texts to analyze
        device: Device for inference (cuda/cpu)

    Returns:
        List of dicts with sentiment scores (positive, negative, neutral)
    """
    finbert_obj = get_finbert_sentiment(device=device)
    if not hasattr(finbert_obj, "analyze_batch"):
        msg = "FinBERT object missing analyze_batch method"
        raise AttributeError(msg)
    scores = finbert_obj.analyze_batch(texts)
    return [{"positive": s.positive, "negative": s.negative, "neutral": s.neutral} for s in scores]


def get_finbert_sentiment(device: str | None = None) -> FinBERTProtocol:
    """Get or create singleton FinBERT sentiment analyzer (local or remote based on FINBERT_MODE).

    Lazy-loads model on first call. Subsequent calls return cached instance.
    Thread-safe. Device parameter only used on first initialization.

    Args:
        device: Device for inference (cuda/cpu). Auto-detect if None.
                Only used on first call in local mode; ignored in remote mode and on subsequent calls.

    Returns:
        FinBERTSentiment or FinBERTClient instance (both provide analyze_batch method)
    """
    if FINBERT_MODE == "remote":
        # Remote mode: HTTP client
        with _FinBERTHolder.lock:
            if _FinBERTHolder.instance is None:
                from src.models.sentiment_client import FinBERTClient

                _FinBERTHolder.instance = cast("FinBERTProtocol", FinBERTClient(base_url=FINBERT_SERVICE_URL))
                logger.info(f"FinBERT remote client initialized (url={FINBERT_SERVICE_URL})")
            instance = _FinBERTHolder.instance
            if instance is None:
                msg = "Failed to initialize FinBERT client"
                raise RuntimeError(msg)
            return instance

    # Local mode: existing in-process implementation
    # Fast path: already initialized (no lock needed)
    if _FinBERTHolder.instance is not None:
        if (
            device is not None
            and hasattr(_FinBERTHolder.instance, "device")
            and device != _FinBERTHolder.instance.device
        ):
            cached_device = _FinBERTHolder.instance.device
            logger.warning(
                f"Device parameter '{device}' ignored - using cached instance with device '{cached_device}'"
            )
        return _FinBERTHolder.instance

    # Slow path: first call, acquire lock
    with _FinBERTHolder.lock:
        # Double-check after lock (another thread may have initialized)
        if _FinBERTHolder.instance is not None:
            return _FinBERTHolder.instance

        model = FinBERTSentiment(device=device)
        _FinBERTHolder.instance = model
        logger.info(f"FinBERT singleton initialized on {model.device}")
        instance = _FinBERTHolder.instance
        if instance is None:
            msg = "Failed to initialize FinBERT"
            raise RuntimeError(msg)
        return instance


def clear_finbert_sentiment() -> None:
    """Clear cached FinBERT instance (for testing/cleanup)."""
    with _FinBERTHolder.lock:
        _FinBERTHolder.instance = None
        logger.debug("FinBERT singleton cleared")


def get_finbert_executor() -> ProcessPoolExecutor:
    """Get or create process pool executor.

    Creates executor on first call or after shutdown. Thread-safe.
    Worker count: FINBERT_WORKERS env var > os.cpu_count() > 4

    Returns:
        ProcessPoolExecutor for FinBERT inference
    """
    if _ExecutorHolder.executor is not None:
        return _ExecutorHolder.executor

    with _ExecutorHolder.lock:
        if _ExecutorHolder.executor is None:
            workers = _get_worker_count()
            _ExecutorHolder.executor = ProcessPoolExecutor(max_workers=workers)
            logger.debug(f"FinBERT executor created with {workers} workers")
        return _ExecutorHolder.executor


def _get_worker_count() -> int:
    """Determine optimal worker count for FinBERT executor.

    Priority: FINBERT_WORKERS env var > os.cpu_count() > DEFAULT_WORKERS

    Returns:
        Worker count (1-MAX_WORKERS)
    """
    if FINBERT_WORKERS:
        try:
            workers = int(FINBERT_WORKERS)
            if workers < 1 or workers > MAX_WORKERS:
                logger.warning(f"FINBERT_WORKERS={workers} out of range (1-{MAX_WORKERS}), using cpu_count()")
            else:
                return workers
        except ValueError:
            logger.warning(f"Invalid FINBERT_WORKERS={FINBERT_WORKERS}, using cpu_count()")

    return os.cpu_count() or DEFAULT_WORKERS


def shutdown_finbert_executor() -> None:
    """Shutdown the process pool executor (for cleanup).

    Should be called at application shutdown to properly cleanup worker processes.
    Registered with atexit for automatic cleanup, but can be called explicitly.
    """
    with _ExecutorHolder.lock:
        if _ExecutorHolder.executor is not None:
            _ExecutorHolder.executor.shutdown(wait=True)
            _ExecutorHolder.executor = None
            logger.debug("FinBERT executor shutdown")


# Register shutdown handler for deterministic cleanup
atexit.register(shutdown_finbert_executor)
