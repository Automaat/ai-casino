# Build stage - install build tools and compile dependencies
FROM python:3.14.3-slim AS builder

# Install system dependencies needed for building Python packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    cmake \
    make \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv (pinned version for reproducible builds)
COPY --from=ghcr.io/astral-sh/uv:0.10.3 /uv /usr/local/bin/uv

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock README.md ./
COPY src ./src

# Install dependencies (builds wheels)
RUN uv sync --frozen --no-dev --extra profiling

# Runtime stage - minimal image with only Python and compiled wheels
FROM python:3.14.3-slim

# Install only runtime dependencies (curl for healthcheck)
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv (pinned version for reproducible builds)
COPY --from=ghcr.io/astral-sh/uv:0.10.3 /uv /usr/local/bin/uv

# Set working directory
WORKDIR /app

# Copy installed dependencies and source from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY --from=builder /app/pyproject.toml /app/uv.lock /app/README.md /app/

# Create directory for daemon state
RUN mkdir -p /root/.ai-casino

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=INFO
ENV PATH="/app/.venv/bin:$PATH"

# Default command (can be overridden in docker-compose)
CMD ["python", "-m", "src.main", "daemon", "-c", "/root/.ai-casino/daemon-production.yaml"]
