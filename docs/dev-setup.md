# Development Setup

## Docker-based Development (Recommended)

Run full stack with live reload using production config:

```bash
# Start all services (daemon, postgres, finbert, frontend with HMR)
mise dev

# View logs
mise dev:logs

# Restart daemon after code changes
mise dev:restart

# Stop all services
mise dev:down
```

**Features:**

- Uses `~/.ai-casino/daemon-production.yaml` config
- Frontend: Vite dev server with HMR at http://localhost:5173
- Backend: Source code mounted (restart with `mise dev:restart` after changes)
- All services in Docker (postgres, finbert, daemon, frontend)
- Debug logging enabled

**Ports:**

- Frontend: http://localhost:5173 (Vite dev server)
- Daemon API: http://localhost:8484
- FinBERT: http://localhost:8485
- PostgreSQL: localhost:5432

## Local Development

For debugging or when you need full control:

```bash
# Daemon only (uses daemon-dev.yaml with ollama)
mise dev:local

# Daemon + Svelte dev server (uses daemon-dev.yaml with ollama)
mise dev:svelte
```

## Production

```bash
# Start production environment
mise prod

# View logs
mise prod:logs

# Stop
mise prod:down
```

**Production uses:**

- Nginx for frontend (port 8050)
- Production-optimized builds
- `~/.ai-casino/daemon-production.yaml` config
