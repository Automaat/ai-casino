# Multi-stage build for CPU-optimized FinBERT service
FROM python:3.12-slim AS builder

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --user -r /tmp/requirements.txt

# Runtime stage
FROM python:3.12-slim

# Install curl for healthcheck
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Create app directory
WORKDIR /app

# Copy application code
COPY src /app/src

# Environment defaults
ENV FINBERT_DEVICE=cpu \
    FINBERT_LOG_LEVEL=INFO \
    FINBERT_PORT=8485

# Expose port
EXPOSE 8485

# Healthcheck
HEALTHCHECK --interval=5s --timeout=3s --retries=5 --start-period=30s \
    CMD curl -f http://localhost:8485/health || exit 1

# Run service
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8485"]
