# syntax=docker/dockerfile:1
# Build stage - install build tools and compile dependencies
FROM python:3.14.3-slim AS builder

# Install system dependencies needed for building Python packages
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    cmake \
    make \
    git

# Install uv (pinned version for reproducible builds)
COPY --from=ghcr.io/astral-sh/uv:0.11.0 /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency manifests only — layer cached until lock file changes
COPY pyproject.toml uv.lock README.md ./

# Install deps without the project itself — src changes won't bust this layer
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra profiling --no-install-project

# Copy source after deps are installed
COPY src ./src

# Runtime stage - minimal image with only Python and compiled wheels
FROM python:3.14.3-slim

# Install runtime dependencies (curl for healthcheck)
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    curl

# Install uv (pinned version for reproducible builds)
COPY --from=ghcr.io/astral-sh/uv:0.11.0 /uv /usr/local/bin/uv

WORKDIR /app

# Copy installed dependencies from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/pyproject.toml /app/uv.lock /app/README.md /app/

# Create directory for daemon state
RUN mkdir -p /root/.ai-casino

# Install Playwright before copying src — layer cached until venv changes, not src
RUN /app/.venv/bin/playwright install --with-deps chromium

# Copy source last — changes here don't rebuild playwright or deps
COPY --from=builder /app/src /app/src

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=INFO
ENV PATH="/app/.venv/bin:$PATH"

# Default command (can be overridden in docker-compose)
CMD ["python", "-m", "src.main", "daemon", "-c", "/root/.ai-casino/daemon-production.yaml"]
