# Migration Guide: Environment Variables → YAML-Only Configuration

**BREAKING CHANGE:** AI Casino no longer supports environment variables for configuration. All config must be in `daemon.yaml`.

## Overview

**Before (v0.x):** Dual configuration - YAML primary, env var fallback

**After (v1.x):** YAML-only configuration - no env var fallback

**Exceptions:**

- OS-provided env vars (`TERM_PROGRAM`, `COLORFGBG`) are still used for terminal detection.
- Security-sensitive API key env vars (see list in the "Environment Variable Mapping" section below) are still read as a *fallback only* when the corresponding value is omitted from `daemon.yaml`. If both YAML and env vars are set, **YAML takes precedence**. This fallback is for backward compatibility with existing `.env` setups and is deprecated; you should migrate all API keys into `daemon.yaml`.

## Migration Steps

### 1. Identify Current Env Vars

Check your `.env` file or shell environment for AI Casino configuration:

```bash
# Common env vars that need migration
env | grep -E "ALPHA_VANTAGE|MARKETAUX|ALPACA|ANTHROPIC|OPENAI|FINNHUB|REDDIT|LLM_|LOG_LEVEL|AI_CASINO"
```

### 2. Create YAML Config

Copy the example config and fill in your values:

```bash
cp docs/daemon.yaml.example ~/.ai-casino/daemon.yaml
vim ~/.ai-casino/daemon.yaml
```

### 3. Map Env Vars to YAML

#### API Keys

**Environment Variables:**

```bash
ALPHA_VANTAGE_API_KEY=abc123
MARKETAUX_API_KEY=def456
ALPACA_API_KEY=ghi789
ALPACA_SECRET_KEY=jkl012
ALPACA_PAPER_API_KEY=mno345
ALPACA_PAPER_SECRET_KEY=pqr678
FINNHUB_API_KEY=stu901
REDDIT_CLIENT_ID=vwx234
REDDIT_CLIENT_SECRET=yza567
REDDIT_USER_AGENT=bcd890
```

**YAML Config:**

```yaml
daemon:
  api_keys:
    alpha_vantage_api_key: "abc123"
    marketaux_api_key: "def456"
    alpaca_api_key: "ghi789"
    alpaca_secret_key: "jkl012"
    alpaca_paper_api_key: "mno345"
    alpaca_paper_secret_key: "pqr678"
    finnhub_api_key: "stu901"
    reddit_client_id: "vwx234"
    reddit_client_secret: "yza567"
    reddit_user_agent: "bcd890"
```

#### LLM Configuration

**Environment Variables:**

```bash
LLM_PROVIDER=anthropic
LLM_MODEL=claude-sonnet-4-20250514
ANTHROPIC_API_KEY=sk-ant-...
LLM_MAX_CONCURRENT=10
OLLAMA_BASE_URL=http://localhost:11434
```

**YAML Config:**

```yaml
daemon:
  llm:
    provider: "anthropic"
    model: "claude-sonnet-4-20250514"
    max_concurrent: 10
    ollama_base_url: "http://localhost:11434"

  api_keys:
    anthropic_api_key: "sk-ant-..."
```

#### Logging

**Environment Variable:**

```bash
LOG_LEVEL=DEBUG
```

**YAML Config:**

```yaml
daemon:
  logging:
    log_level: "DEBUG"
```

#### Metrics

**Environment Variables:**

```bash
RISK_FREE_RATE=0.03
EXECUTION_METRICS_ENABLED=false
PORTFOLIO_SNAPSHOT_ON_TRADE=true
```

**YAML Config:**

```yaml
daemon:
  metrics:
    risk_free_rate: 0.03
    execution_metrics_enabled: false
    portfolio_snapshot_on_trade: true
```

#### UI/Dashboard

**Environment Variables:**

```bash
AI_CASINO_THEME=nord-dark
DASHBOARD_HOST=0.0.0.0
DASHBOARD_PORT=8080
DAEMON_API_URL=http://192.168.1.100:8484
DASHBOARD_REFRESH_INTERVAL=10000
```

**YAML Config:**

```yaml
daemon:
  ui:
    theme: "nord-dark"
    dashboard_host: "0.0.0.0"
    dashboard_port: 8080
    daemon_api_url: "http://192.168.1.100:8484"
    dashboard_refresh_interval: 10000
```

#### Database

**Environment Variable:**

```bash
DATABASE_URL=<your_postgres_connection_string>
```

**YAML Config:**

```yaml
daemon:
  database:
    database_url: "<your_postgres_connection_string>"
```

### 4. Verify Configuration

Test your new YAML config:

```bash
# Start daemon with config
python -m src.cli.daemon --config ~/.ai-casino/daemon.yaml

# If config is missing or invalid, you'll see clear error:
# "Config error: Field required: alpha_vantage_api_key"
```

### 5. Remove Old Env Vars

Once verified, clean up:

```bash
# Remove .env file (no longer used)
rm .env

# Remove env vars from shell profile (~/.bashrc, ~/.zshrc)
# Search for lines like: export ALPHA_VANTAGE_API_KEY=...
```

## Complete Mapping Reference

| Environment Variable          | YAML Path                                    | Notes                           |
|:------------------------------|:---------------------------------------------|:--------------------------------|
| `ALPHA_VANTAGE_API_KEY`       | `daemon.api_keys.alpha_vantage_api_key`      | Required                        |
| `MARKETAUX_API_KEY`           | `daemon.api_keys.marketaux_api_key`          | Optional                        |
| `ALPACA_API_KEY`              | `daemon.api_keys.alpaca_api_key`             | Live trading                    |
| `ALPACA_SECRET_KEY`           | `daemon.api_keys.alpaca_secret_key`          | Live trading                    |
| `ALPACA_PAPER_API_KEY`        | `daemon.api_keys.alpaca_paper_api_key`       | Paper trading                   |
| `ALPACA_PAPER_SECRET_KEY`     | `daemon.api_keys.alpaca_paper_secret_key`    | Paper trading                   |
| `FINNHUB_API_KEY`             | `daemon.api_keys.finnhub_api_key`            | Optional                        |
| `REDDIT_CLIENT_ID`            | `daemon.api_keys.reddit_client_id`           | Optional                        |
| `REDDIT_CLIENT_SECRET`        | `daemon.api_keys.reddit_client_secret`       | Optional                        |
| `REDDIT_USER_AGENT`           | `daemon.api_keys.reddit_user_agent`          | Optional                        |
| `ANTHROPIC_API_KEY`           | `daemon.api_keys.anthropic_api_key`          | Claude LLM                      |
| `OPENAI_API_KEY`              | `daemon.api_keys.openai_api_key`             | GPT LLM                         |
| `OPENAI_API_BASE`             | `daemon.api_keys.openai_api_base`            | Custom endpoint                 |
| `LLM_PROVIDER`                | `daemon.llm.provider`                        | ollama/anthropic/openai         |
| `LLM_MODEL`                   | `daemon.llm.model`                           | Model ID                        |
| `LLM_MAX_CONCURRENT`          | `daemon.llm.max_concurrent`                  | 1-20, default 5                 |
| `OLLAMA_BASE_URL`             | `daemon.llm.ollama_base_url`                 | Default: http://localhost:11434 |
| `LOG_LEVEL`                   | `daemon.logging.log_level`                   | DEBUG/INFO/WARNING/ERROR        |
| `RISK_FREE_RATE`              | `daemon.metrics.risk_free_rate`              | Default: 0.02                   |
| `EXECUTION_METRICS`           | `daemon.metrics.execution_metrics_enabled`   | Default: true                   |
| `PORTFOLIO_SNAPSHOT_ON_TRADE` | `daemon.metrics.portfolio_snapshot_on_trade` | Default: false                  |
| `AI_CASINO_THEME`             | `daemon.ui.theme`                            | nord-dark/nord-light/null       |
| `DASHBOARD_HOST`              | `daemon.ui.dashboard_host`                   | Default: 127.0.0.1              |
| `DASHBOARD_PORT`              | `daemon.ui.dashboard_port`                   | Default: 8050                   |
| `DAEMON_API_URL`              | `daemon.ui.daemon_api_url`                   | Default: http://localhost:8484  |
| `DASHBOARD_REFRESH_INTERVAL`  | `daemon.ui.dashboard_refresh_interval`       | Default: 5000ms                 |
| `DATABASE_URL`                | `daemon.database.database_url`               | PostgreSQL connection           |

## Troubleshooting

### Error: Config file not found

```bash
Error: Config file not found: ~/.ai-casino/daemon.yaml
```

**Fix:** Create config file from example:

```bash
cp docs/daemon.yaml.example ~/.ai-casino/daemon.yaml
```

### Error: Field required

```bash
Config error: Field required: alpha_vantage_api_key
```

**Fix:** Add required field to `daemon.yaml`:

```yaml
daemon:
  api_keys:
    alpha_vantage_api_key: "YOUR_KEY_HERE"
```

### Error: Invalid log level

```bash
Config error: Invalid log_level: 'debug' (must be uppercase)
```

**Fix:** Use uppercase log level:

```yaml
daemon:
  logging:
    log_level: "DEBUG"  # Not "debug"
```

### CLI tools fail to start

**Issue:** Some CLI tools may not honor `daemon.yaml` settings.

**Fix:** Most CLI tools load defaults internally. For tools that support config loading, ensure the default config path exists:

```bash
# Create default config location
mkdir -p ~/.ai-casino
cp docs/daemon.yaml.example ~/.ai-casino/daemon.yaml
```

Note: CLI tools that support YAML config will automatically load from `~/.ai-casino/daemon.yaml` if it exists. Check individual tool documentation for config support.

### Daemon ignores my settings

**Issue:** You have both `.env` file and `daemon.yaml` with conflicting values.

**Fix:** YAML config takes full priority now. Remove `.env` file:

```bash
rm .env
```

## Rollback

If you need to rollback to environment variable support:

```bash
# Checkout previous version
git checkout v0.x.x

# Restore .env file
git restore .env

# Reinstall dependencies
uv sync
```

**Note:** Future versions (v1.x+) will not support env var fallback.

## Benefits of YAML-Only

- **Single source of truth**: No confusion about config priority
- **Type validation**: Pydantic validates all config fields
- **Better defaults**: Clear default values in config models
- **Discoverability**: `daemon.yaml.example` shows all options
- **Version control**: Config structure in git, secrets in file
- **IDE support**: YAML schema validation in editors

## Questions?

- Check `docs/daemon.yaml.example` for comprehensive config documentation
- See `docs/daemon-operations.md` for operational guide
- Open issue: https://github.com/skalski-ai-casino/ai-casino/issues
