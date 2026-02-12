# FinBERT Microservice Deployment Guide

Comprehensive guide for deploying and operating the FinBERT sentiment analysis microservice.

## Quick Start

Build and run service:

```bash
cd services/finbert
docker build -t ai-casino-finbert .
docker run -d --name finbert -p 8485:8485 ai-casino-finbert
```

Verify health:

```bash
curl http://localhost:8485/health
```

Configure daemon (~/.ai-casino/daemon.yaml):

```yaml
daemon:
  finbert:
    mode: remote
    service_url: http://localhost:8485
```

## Configuration

Environment variables:

- `FINBERT_DEVICE` - cpu or cuda (default: cpu)
- `FINBERT_LOG_LEVEL` - DEBUG, INFO, WARNING, ERROR (default: INFO)
- `FINBERT_PORT` - Service port (default: 8485)

## Performance

Memory usage:

- Local: 2.5GB (daemon + model)
- Remote CPU: 2GB (service only)
- Remote GPU: 4GB (service + CUDA)

Latency (50 texts):

- Local: 200-300ms
- Remote: 220-320ms
- Overhead: ~20ms (10%)

Benchmark: `python scripts/benchmark_finbert.py`

## Deployment

See `services/finbert/README.md` for complete deployment guide including:

- Docker deployment (CPU and GPU)
- Horizontal scaling with nginx
- Monitoring and troubleshooting
- Migration from local mode
