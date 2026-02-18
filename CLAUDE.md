# AI Casino

Multi-agent stock trading system: technical analysis, sentiment (FinBERT), news (LLM) → trading decisions (BUY/SELL/HOLD).

**Stack:** Python 3.12, pandas-ta, transformers (FinBERT), yfinance, Anthropic/OpenAI SDKs
**Status:** MVP complete (44% - 11/25 features)

---

## Project Structure

```
src/
├── coordinator/     # PRIMARY: TradingCoordinator (LLM tool-calling loop), tools/, memory
├── agents/          # TraderAgent, TradingSupervisor, GamePlanAgent, RiskManagementAgent, MetaAgent, CriticAgent, TradeJournalAgent, EventTriageAgent
├── workers/         # Stateless LLM execution units: technical, sentiment, news, fundamental, comparative, web_research, social, trump, thesis_research
├── workflows/       # Per-symbol pipeline (called via analyze_symbol tool): stages/, orchestrator
├── data/            # market.py (Alpha Vantage + yfinance), news.py (Marketaux)
├── models/          # llm.py (facade), providers/ (Anthropic/OpenAI/Ollama), sentiment.py (FinBERT)
├── strategies/      # momentum.py (RSI + MACD)
├── daemon/          # Runner, cycle orchestrator, task runner, scheduled tasks, API
└── main.py
tests/               # Mirror of src/, conftest.py has shared fixtures
```

---

## Development Workflow

1. ASK clarifying questions → research existing patterns → plan → implement incrementally
2. `mise check` before every commit (format + lint + typecheck + test). Never skip.
3. `mise audit` for CVEs (optional locally, enforced in CI)

**Git:** Conventional commits with `-s -S` flags. Branches: `feat/`, `fix/`, `chore/`

**PRs:** Must link issue (`Fix: #123`). Format: `## Motivation` / `## Implementation information` / `## Supporting documentation`

### Configuration (YAML-only)

- **ALWAYS** `~/.ai-casino/daemon-production.yaml` — **NEVER** env vars for config
- Env vars only as fallbacks in DI providers (`resolve_config_or_env`)
- Adding config: model in `src/daemon/config/` → `DaemonConfig` field → `from_yaml()` → **update `docs/daemon.yaml.example`** → DI provider

---

## Python Conventions

**Tools:** ruff (format+lint), pyrefly (typecheck) | **Line:** 110 | **Quotes:** Double | **Docstrings:** Google style
**Limits:** 400 lines/file, 60 lines/method | **Type hints:** Mandatory on all functions
**Errors:** Fix properly, NEVER use skip/disable directives. Exception: missing stubs, complex generics pyrefly can't infer.

**Imports:** stdlib → third-party (alphabetical) → local. `TYPE_CHECKING` only for expensive imports (torch, transformers).

**Types:** Python 3.10+ syntax (`list[str]`, `int | None`). No `Any` unless truly dynamic. `collections.abc` for params. `Final` for constants.

### Error Handling

- Try-except + `logger.error()` + re-raise. No bare excepts.
- **Critical** (propagate): data fetchers, LLM, broker, DB, user-facing
- **Non-critical** (may swallow): batch processing, cache, metrics — use `logger.opt(exception=True).warning()` when swallowing
- Specific exceptions first, then general

### Pydantic Models

All classes: 1-line docstring, `__repr__()`. Use `Field()` for validation, `| None` not `Optional`, `StrEnum` for fixed strings, `@property` for computed fields. Name: `{Component}{Purpose}`.

### Testing

Fixtures: `sample_ohlcv_data`, `sample_news_articles`, `mock_llm_client`, `mock_finbert`
Markers: `@pytest.mark.unit`, `.integration`, `.slow`. Mock all external APIs.

### Async

- All I/O methods `async`. `asyncio.run()` only at CLI entry points.
- Blocking I/O → `await asyncio.to_thread(fn)`. CPU-heavy → `run_in_executor`.
- `asyncio.Semaphore` for rate limiting, `asyncio.Lock` for async state (never `threading.Lock` in async)
- `asyncio.gather(*tasks, return_exceptions=True)` — handle exceptions, use semaphore for backpressure
- Reuse HTTP clients. Never: `nest_asyncio`, `asyncio.run()` in async, blocking calls in async

---

## Simplicity Principles

❌ **NEVER:** TODOs, placeholders, obvious comments, over-engineering, >400 line files, >60 line methods, print(), bare excepts, commented code, globals, singletons, dicts for structured data, inheritance for code reuse

✅ **ALWAYS:** Simplest solution, reuse patterns, typed classes over dicts, composition over inheritance

### Typed Classes over Dicts

**Use Pydantic/dataclasses for:** return values, parameters, state, config, API schemas

**Dicts acceptable only for:** framework requirements (LLM tool `**kwargs`, decorators), dynamic key-value stores (caches, registries), template interpolation (`PromptLoader.load(**vars)`), transient external API parsing

**Migration:** Create model → update signature → replace dict construction → update consumers → `mise typecheck`

---

## Architecture Patterns

### Dependency Injection (MANDATORY)

All deps via `__init__`. Use `dependency-injector` container (`src/di/container.py`). Singletons for stateful services, Factories for per-request.

**Constructor-only injection:** All dependencies pass through `__init__` as non-null typed params. Initialize all instance attributes in `__init__` with defaults (never leave uninitialized). NEVER use setter methods or `| None` for required dependencies.

```python
# ❌ Setter injection or nullable deps
class Service:
    def __init__(self):
        self.dep: Dependency | None = None
    def set_dep(self, dep: Dependency): ...

# ✅ Constructor injection with non-null types
class Service:
    def __init__(self, dep: Dependency):
        self.dep = dep
        self._state = 0  # Initialize all attrs
```

**CRITICAL:** `providers.Self()` doesn't work with Factory — always pass container explicitly:

```python
# ❌ container=providers.Self() → None
# ✅ container=container when calling factory
workflow = container.workflow_meta(broker=broker, container=container)
```

**Adding providers:** Singleton/Factory in container → provider function in `src/di/providers/` → use `resolve_config_or_env` for API keys

Never create service instances directly — always use container.

### Composition over Inheritance (MANDATORY)

Extract shared behavior into composable components injected via DI. Each abstraction: single-responsibility, independently testable.

Inheritance OK only for: Protocol/ABC, Pydantic BaseModel, framework-required, Enum.

### Database Sessions (MANDATORY)

Sessions per-request, engines singleton. `expire_on_commit=False`, `autoflush=False`.

- **API endpoints:** `session: AsyncSession = Depends(get_db_session)`
- **Background tasks:** `async with get_db_engine().session() as session:`
- **Repositories:** receive session in `__init__`, never create sessions
- **State managers:** API methods accept session param, background methods create inline
- **Never:** store sessions in long-lived objects, scoped sessions, sessions in repos

### LLM Abstraction

`BaseLLMProvider` → `AnthropicProvider`, `OpenAIProvider`, `OllamaProvider`. `LLMClient` facade with factory pattern.

Dev: Ollama qwen3:14b | Prod: Claude sonnet-4 | Alt: OpenAI gpt-4o

### Agent Pattern

Structured output with `{Agent}LLMResponse` models + fallback to `acomplete`. All prompts in `src/prompts/{agent}/` via `PromptLoader` — never hardcode.

### Workflow: Coordinator Mode (Primary)

Coordinator mode is the **primary and preferred mode**. All new features must be added here.

`TradingCoordinator` runs an LLM-driven agentic loop — the LLM autonomously decides which tools to call, in what order, for which symbols:

```
TradingCoordinator.run_cycle()
  └─ llm.acomplete_with_tools(prompt, tools, max_tool_calls=N)
       └─ LLM autonomously calls tools:
            generate_game_plan → market_overview → screen_stocks
            → analyze_symbol (triggers full supervisor+workers pipeline)
            → execute_trade → save_observation → reflect_on_decision
```

**Adding new capabilities to coordinator:**
- New tool → implement in `src/coordinator/tools/` or `src/tools/`
- Register in `build_coordinator_registry()` (`src/coordinator/tools/__init__.py`)
- New agent → wire as tool or inject into existing tool

**Legacy pipeline** (`fetch data → technical → sentiment → news → decision`) still exists and is called by `analyze_symbol` tool internally. Do not add new features there — integrate at coordinator level instead.

---

## Domain Rules

- **Signals:** Always `Signal` enum (BUY/SELL/HOLD), never strings
- **Confidence:** 0.0-1.0. Risk: LOW (≥0.75), MEDIUM (0.5-0.75), HIGH (<0.5)
- **Temperature:** Technical 0.3, Trading 0.5, General 0.7
- **Data:** Alpha Vantage → yfinance fallback → raise on empty
- **Indicators:** RSI(14) oversold <30 overbought >70, MACD histogram >0 bullish (needs ~35 points)
- **Sessions:** REGULAR (9:30-16:00 ET), PRE_MARKET (4:00-9:30 ET, config-enabled)

### UI/UX: Empty States

Distinguish "disabled" vs "no data yet" with status fields (`enabled`, `database_enabled`, `has_data`).

---

## Commands

```bash
uv sync --frozen --all-extras          # Install deps
python -m src.main AAPL                # Run analysis
mise check                             # All checks (format+lint+typecheck+test)
mise format|lint|typecheck|test        # Individual checks
mise test:cov                          # Coverage
mise audit                             # CVE check
mise ollama:start|stop|status          # Local LLM
```

**Config:** `~/.ai-casino/daemon-production.yaml` (see `docs/daemon.yaml.example`)
**Env fallbacks:** `ALPHA_VANTAGE_API_KEY`, `MARKETAUX_API_KEY`, `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `LOG_LEVEL`, `TOOL_EXECUTION_MAX_CONCURRENT`
**Logs:** `~/.ai-casino/worker.log` (debug), `~/.ai-casino/chat-history.json`

## Gotchas

- Alpha Vantage: 5 req/min free tier, cache in data/cache/
- FinBERT: 440MB first download
- Ollama: must run locally (`mise ollama:start`)
- Empty news: warning, not error
- Transformers logging: suppress BEFORE import cascade — env vars at `src/cli/app.py` + `hf_logging.set_verbosity_error()`
- OpenAI structured output: requires `additionalProperties: false` recursively

### Database Debugging

**Local development always uses PostgreSQL via Docker:**
- Container: `ai-casino-postgres` (postgres:18-alpine)
- Connection: `postgresql+asyncpg://ai_casino:ai_casino_secure_pass_2025@localhost:5432/ai_casino`
- Configured in `~/.ai-casino/daemon-production.yaml`

**Quick checks:**
```bash
docker ps | grep postgres                           # Check container running
docker exec ai-casino-postgres psql -U ai_casino -d ai_casino -c "\dt"  # List tables
docker exec ai-casino-postgres psql -U ai_casino -d ai_casino -c "SELECT COUNT(*) FROM discovery_history;"  # Query
```

**Common issues:**
- Container not running → `docker start ai-casino-postgres`
- Empty tables despite daemon running → check if data written to correct DB (not SQLite fallback)

## Resources

- Plan: ./implem-plan.md | Research: ./agentic-stock-trading-system-research.md
- Config: pyproject.toml, ruff.toml | CI: .github/workflows/ci.yml
