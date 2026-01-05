"""FinBERT sentiment analysis model wrapper."""

import torch
from loguru import logger
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer


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
