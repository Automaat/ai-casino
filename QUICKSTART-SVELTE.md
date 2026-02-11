# Quick Start: Svelte Frontend

Get the new Svelte trading dashboard running in 2 minutes.

## Prerequisites

- Python 3.12+
- Node.js 18+ (for frontend)
- mise installed (`curl https://mise.run | sh`)
- API keys configured (see main README)

## Option 1: Development Mode (Recommended for First Run)

**Hot reload, instant updates, dev tools:**

```bash
# Install dependencies (one-time)
cd frontend && npm install && cd ..

# Start everything with one command
mise dev:svelte
```

This starts:
- ✅ Daemon with dev config (Ollama for LLM)
- ✅ Svelte dev server with HMR
- 🌐 http://localhost:5173

**Press Ctrl+C to stop both services**

## Option 2: Production Mode

**Optimized build, production config:**

```bash
# Start everything with one command
mise prod:svelte
```

This starts:
- ✅ Daemon with production config (~/.ai-casino/daemon-production.yaml)
- ✅ Svelte optimized build + preview server
- 🌐 http://localhost:4173

## Configuration

### Daemon API (Required)

Add to your config file:

**For dev:** `daemon-dev.yaml`
```yaml
api:
  enabled: true
  port: 8484
  cors_origins:
    - "http://localhost:5173"  # Svelte dev server
```

**For prod:** `~/.ai-casino/daemon-production.yaml`
```yaml
api:
  enabled: true
  port: 8484
  cors_origins:
    - "http://localhost:4173"  # Svelte preview server
    - "https://yourdomain.com"  # Production domain
```

### Environment Variables (.env)

Optional - defaults work out of box:
```bash
# Frontend (in frontend/.env)
VITE_API_URL=http://localhost:8484  # Daemon API URL (default)

# Backend (in root .env)
LLM_PROVIDER=ollama                 # or anthropic/openai
LLM_MODEL=qwen3:14b                 # or claude-sonnet-4-5/gpt-4o
```

## Dashboard Pages

Once running, explore:

| Page | URL | Features |
|------|-----|----------|
| **Overview** | http://localhost:5173 | Health metrics, confidence trends, recent analyses |
| **Portfolio** | http://localhost:5173/portfolio | Positions, equity curve, allocation treemap |
| **Signals** | http://localhost:5173/signals | Filterable trade signals, stats |
| **Risk** | http://localhost:5173/risk | Sharpe ratio, volatility, correlation heatmap |

## Troubleshooting

### "Failed to fetch" errors

**Problem:** Frontend can't connect to daemon API

**Solution:**
1. Check daemon is running: `curl http://localhost:8484/health`
2. Verify CORS config includes frontend URL
3. Check browser console for errors

### Empty dashboard

**Problem:** No data showing in charts/tables

**Solution:**
- Wait for first trading cycle to complete
- Check daemon logs: `tail -f ~/.ai-casino/logs/daemon.log`
- Verify watchlist is configured in daemon config

### Type errors after API changes

**Problem:** TypeScript errors after updating backend

**Solution:**
```bash
cd frontend
npm run generate-api  # Regenerate API client from OpenAPI spec
npm run check          # Verify types
```

### Port conflicts

**Problem:** "Port already in use"

**Solution:**
```bash
# Kill existing processes
pkill -f "vite dev"
pkill -f "src.main daemon"

# Or use different ports
cd frontend
npm run dev -- --port 5174
```

## Manual Setup (Without mise)

**Terminal 1 - Backend:**
```bash
uv run python -m src.main daemon --config daemon-dev.yaml
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

## Next Steps

- **Customize:** Edit `frontend/src/routes/+page.svelte` for overview page
- **Add Charts:** Use components in `frontend/src/lib/components/charts/`
- **New Pages:** Add to `frontend/src/routes/yourpage/+page.svelte`
- **API Client:** Auto-generated from FastAPI: `npm run generate-api`

## Comparison: Dash vs Svelte

| Feature | Old Dash | New Svelte |
|---------|----------|------------|
| Command | `mise dev` | `mise dev:svelte` |
| Port | 8050 | 5173 (dev) / 4173 (prod) |
| Hot Reload | Slow | Instant (HMR) |
| Load Time | 2-3s | <500ms |
| Bundle Size | ~500KB | ~100KB |
| Customization | Limited | Full control |

---

**Ready to build!** The Svelte dashboard auto-refreshes every 5 seconds and uses beautiful ECharts + Lightweight Charts visualizations.
