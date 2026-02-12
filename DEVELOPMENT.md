# Development Setup

## Two Dev Modes

### 1. Cloud LLM Mode (Recommended)

Fast and reliable - uses Anthropic/OpenAI APIs.

```bash
mise dev
```

**Config:** Uses `~/.ai-casino/daemon-production.yaml`

- Set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`
- Configure provider in `daemon.llm.provider`

**Pros:**

- ✅ Fast inference
- ✅ High quality responses
- ✅ No local GPU needed

**Cons:**

- ❌ Requires API keys
- ❌ Costs money per request

### 2. Ollama Mode (Free, Local)

Local LLM inference - free but slower on CPU.

```bash
mise dev:ollama
```

**Config:** Uses `~/.ai-casino/daemon-dev.yaml`

- Runs Ollama container locally
- Default model: `qwen3:14b` (~8GB)

**First time setup:**

```bash
# Pull model after first start
docker exec ai-casino-ollama-dev ollama pull qwen3:14b
```

**Pros:**

- ✅ Free
- ✅ Private (no external API calls)
- ✅ Works offline

**Cons:**

- ❌ Very slow on CPU (60s+ per analysis)
- ❌ Large model download (~8GB)
- ❌ High memory usage

**GPU Support (much faster):**

Uncomment GPU section in `docker-compose.dev-ollama.yml`:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

## Stopping Dev Environment

```bash
# Stop cloud mode
mise dev:down

# Stop Ollama mode
mise dev:ollama:down
```

## Services

Both modes include:

- **Frontend:** http://localhost:5173 (Vite HMR)
- **API:** http://localhost:8484
- **Postgres:** localhost:5432
- **FinBERT:** http://localhost:8485

Ollama mode adds:

- **Ollama:** http://localhost:11434
