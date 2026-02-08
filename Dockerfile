FROM python:3.14-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    cmake \
    make \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock README.md ./
COPY src ./src

# Install dependencies
RUN uv sync --frozen --no-dev

# Create directory for daemon state
RUN mkdir -p /root/.ai-casino

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=INFO

# Default command (can be overridden in docker-compose)
CMD ["uv", "run", "python", "-m", "src.main", "daemon", "-c", "/root/.ai-casino/daemon-production.yaml"]
