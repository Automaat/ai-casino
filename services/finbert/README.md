# FinBERT Sentiment Service

FastAPI microservice for financial sentiment analysis using ProsusAI/finbert.

## Overview

Extracts FinBERT (440MB transformers model) to independent microservice for:

- **Horizontal scaling** - Run multiple instances with load balancer
- **GPU acceleration** - Dedicated GPU container (5-10x faster than CPU)
- **Isolation** - ML inference doesn't block main daemon

## API Endpoints

### GET /health

Health check for Docker healthcheck.

**Response:**

```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu",
  "uptime_seconds": 123.45
}
```

### POST /analyze

Batch sentiment analysis (1-100 texts).

**Request:**

```json
{
  "texts": [
    "Strong earnings beat expectations",
    "Weak guidance disappoints investors"
  ]
}
```

**Response:**

```json
{
  "scores": [
    {"positive": 0.8, "negative": 0.1, "neutral": 0.1},
    {"positive": 0.2, "negative": 0.7, "neutral": 0.1}
  ],
  "inference_time_ms": 234.5,
  "batch_size": 2
}
```

**Errors:**

- `422` - Validation error (empty batch or >100 texts)
- `503` - Model not loaded
- `500` - Inference failed

## Local Development

### Install dependencies

```bash
cd services/finbert
pip install -r requirements.txt
```

### Run service

```bash
uvicorn src.main:app --reload --port 8485
```

### Test manually

```bash
# Health check
curl http://localhost:8485/health

# Analyze batch
curl -X POST http://localhost:8485/analyze \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Strong earnings", "Weak guidance"]}'
```

## Docker Deployment

### CPU (Default)

```bash
docker build -t ai-casino-finbert .
docker run -p 8485:8485 ai-casino-finbert
```

### GPU

```bash
docker build -f Dockerfile.gpu -t ai-casino-finbert-gpu .
docker run --gpus all -p 8485:8485 \
  -e FINBERT_DEVICE=cuda \
  ai-casino-finbert-gpu
```

### docker-compose (Production)

```bash
# CPU mode (default)
docker-compose up -d finbert

# GPU mode (edit docker-compose.yml to use Dockerfile.gpu and uncomment deploy section)
docker-compose up -d finbert
```

## Configuration

Environment variables:

- `FINBERT_DEVICE` - `cpu` or `cuda` (default: `cpu`)
- `FINBERT_LOG_LEVEL` - `DEBUG`, `INFO`, `WARNING`, `ERROR` (default: `INFO`)
- `FINBERT_PORT` - Service port (default: `8485`)

## Scaling

### Horizontal Scaling with nginx

**nginx.conf:**

```nginx
upstream finbert {
    server finbert-1:8485;
    server finbert-2:8485;
    server finbert-3:8485;
}

server {
    listen 8485;
    location / {
        proxy_pass http://finbert;
    }
}
```

**docker-compose.yml:**

```yaml
services:
  finbert-1:
    build: ./services/finbert
    environment:
      - FINBERT_DEVICE=cpu

  finbert-2:
    build: ./services/finbert
    environment:
      - FINBERT_DEVICE=cpu

  finbert-3:
    build: ./services/finbert
    environment:
      - FINBERT_DEVICE=cpu

  nginx:
    image: nginx:alpine
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    ports:
      - "8485:8485"
    depends_on:
      - finbert-1
      - finbert-2
      - finbert-3
```

## Performance

**CPU (single instance):**

- Batch size 50: ~200-300ms
- Memory: ~2GB

**GPU (single instance):**

- Batch size 50: ~30-50ms (5-10x faster)
- Memory: ~4GB

**Overhead vs local:**

- HTTP overhead: ~10-20ms
- Target: <25% overhead

Run benchmark: `python scripts/benchmark_finbert.py`

## Troubleshooting

### Service won't start

- Check model download completed (440MB)
- Verify port 8485 not in use
- Check Docker logs: `docker logs ai-casino-finbert`

### 503 Model not loaded

- Wait 30s for model initialization
- Check healthcheck: `curl http://localhost:8485/health`

### GPU not detected

- Verify NVIDIA drivers installed
- Check Docker GPU support: `docker run --gpus all nvidia/cuda:12.1.0-base nvidia-smi`
- Uncomment `deploy.resources` in docker-compose.yml
