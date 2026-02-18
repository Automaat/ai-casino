"""FastAPI application for FinBERT sentiment analysis service."""

import sys
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from loguru import logger

from .config import Settings
from .inference import FinBERTInference
from .models import BatchRequest, BatchResponse, HealthResponse

# Global state
settings = Settings()
finbert_model: FinBERTInference | None = None
start_time: float = time.time()

_LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)

# Configure loguru
logger.remove()
logger.add(sys.stderr, format=_LOG_FORMAT, level=settings.log_level)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None]:
    """Lifespan context manager for model loading/cleanup."""
    global finbert_model  # noqa: PLW0603

    logger.info("Starting FinBERT service")
    try:
        finbert_model = FinBERTInference(device=settings.device)
        logger.info(f"Model loaded on {finbert_model.device}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    yield

    logger.info("Shutting down FinBERT service")
    finbert_model = None


app = FastAPI(
    title="FinBERT Sentiment Service",
    description="Financial sentiment analysis using ProsusAI/finbert",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint for Docker healthcheck."""
    uptime = time.time() - start_time
    return HealthResponse(
        status="healthy" if finbert_model is not None else "unhealthy",
        model_loaded=finbert_model is not None,
        device=finbert_model.device if finbert_model else "unknown",
        uptime_seconds=uptime,
    )


@app.post("/analyze", response_model=BatchResponse)
async def analyze_batch(request: BatchRequest) -> BatchResponse:
    """Batch sentiment analysis endpoint.

    Args:
        request: BatchRequest with texts to analyze (1-100)

    Returns:
        BatchResponse with sentiment scores and metadata

    Raises:
        HTTPException: 503 if model not loaded, 422 if invalid request
    """
    if finbert_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        start = time.perf_counter()
        scores = await finbert_model.analyze_batch_async(request.texts)
        inference_time_ms = (time.perf_counter() - start) * 1000

        return BatchResponse(
            scores=scores,
            inference_time_ms=inference_time_ms,
            batch_size=len(request.texts),
        )
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}") from e
