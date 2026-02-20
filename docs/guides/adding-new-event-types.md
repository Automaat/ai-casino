# Adding New Event Types

All market events flow through: **source watcher → `EventTriageAgent` → `MarketEventQueue` → `EventQueueConsumer` → `TradingCoordinator`**.

---

## Checklist

### 1. Define the event model — `src/daemon/events.py`

```python
class MyEvent(BaseModel):
    """One-line docstring."""

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: Literal["my_event"] = "my_event"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    source: str = "my_source"
    # ... event-specific fields

    def __repr__(self) -> str:
        """Return string representation."""
        return f"MyEvent(...)"

    def to_prompt_text(self) -> str:
        """Format for LLM triage prompt."""
        return f"MY EVENT: ..."
```

Rules:
- `event_type` must be `Literal["my_event"]` — matches the string stored in the DB
- `event_id` must be stable across retries (URL, UUID, accession number, etc.)
- `to_prompt_text()` is shown to the triage LLM — keep it information-dense

### 2. Register the prompt template — `src/coordinator/event_prompt.py`

Add the event type string to `_EVENT_TYPE_TEMPLATES`:

```python
_EVENT_TYPE_TEMPLATES = frozenset({..., "my_event"})
```

If the event has fields beyond the common ones (`source`, `symbol`, `article`, `post`, `mention_count`, `spike_ratio`, `anomaly_types`), add rendering in `_format_event_details` or extract a helper like `_append_signal_fields`:

```python
def _append_my_event_fields(event_data: dict, lines: list[str]) -> None:
    """Append my_event-specific fields."""
    if my_field := event_data.get("my_field"):
        lines.append(f"My Field: {my_field}")
```

Call it at the end of `_format_event_details`:

```python
_append_my_event_fields(event_data, lines)
```

> Keep `_format_event_details` complexity ≤ 10 (C901). Extract helpers for event-specific field blocks.

### 3. Create the coordinator prompt template — `src/prompts/coordinator/events/my_event.txt`

```
### My Event

{event_details}

**Triage:** urgency={urgency}, sentiment={sentiment}, confidence={confidence:.0%}
**Reasoning:** {reasoning}

**Instructions:**
1. ...specific coordinator instructions for this event type...
```

Required template variables (always present): `event_details`, `urgency`, `sentiment`, `confidence`, `reasoning`.

One extra variable is always passed but only used when explicitly referenced: `game_plan` (today's game plan text, populated only when at least one event in the batch is a `signal` type). Add `{game_plan}` to your template if the event type benefits from cross-referencing the day's trading plan.

If `event_type` is not in `_EVENT_TYPE_TEMPLATES`, the prompt builder falls back to `news.txt`.

### 3a. Pass runtime context via `EventCycleContext`

`EventCyclePromptBuilder.build()` accepts an `EventCycleContext` dataclass instead of individual positional args:

```python
@dataclass
class EventCycleContext:
    positions_summary: str
    session: TradingSession
    market_open: bool
    game_plan: str = ""  # populated by run_event_cycle for signal events
```

When adding a new event type that needs additional runtime context in its prompt (beyond the five standard triage variables), extend `EventCycleContext` with a new field and populate it in `TradingCoordinator.run_event_cycle()` — the same pattern used for `game_plan`.

### 4. Implement the watcher / emitter

Events enter the queue from two paths:

**a) Watcher-based** (polls external source, triages via `EventTriageAgent`):
- Subclass `PeriodicWatcher` in `src/v1/watchers/`
- Inject `EventTriagePipeline` via constructor
- Override `_tick()` to call `_fetch_events()` then `await self._pipeline.process(events)`
- See `src/v1/watchers/news_watcher.py` as a reference implementation

**b) Orchestrator-based** (generated from analysis results, e.g. `SignalEvent`):
- Construct `TriageResult` manually (skip LLM triage)
- Call `market_event_queue.enqueue(event, triage, ttl_hours=..., process_after=...)`
- Use `process_after` to defer processing (e.g. signal deferred to market open)

### 5. Wire watcher into `DaemonRunner` / `DaemonComponents` (watcher path only)

1. Add to `DaemonComponents` dataclass (`src/daemon/factory.py`)
2. Create provider in `src/di/providers/watchers.py` (use `_build_pipeline()` helper)
3. Register singleton in `src/di/container.py`
4. Start/stop in `DaemonLifecycle` (`src/daemon/lifecycle.py`)

---

## TTL and `process_after` guidance

| Use case | `ttl_hours` | `process_after` |
|---|---|---|
| Breaking news (act now) | 4 | `None` |
| Pre-market signal (act at open) | `max(8, hours_until_open + 2)` | `scheduler.next_regular_open()` |
| Filing (act within the day) | 12 | `None` |
| Economic event (act at specific time) | 24 | scheduled event time |

---

## Existing event types

| `event_type` | Class | Source |
|---|---|---|
| `news` | `NewsEvent` | Marketaux / DuckDuckGo |
| `social` | `SocialEvent` | Reddit / Finnhub |
| `filing` | `FilingEvent` | SEC EDGAR |
| `trump` | `TrumpEvent` | Truth Social |
| `anomaly` | `AnomalyEvent` | Market data watcher |
| `news_trending` | `NewsTrendingEvent` | News trending watcher |
| `signal` | `SignalEvent` | `AnalysisOrchestrator` (pre-market) |

---

## Key files

| File | Purpose |
|---|---|
| `src/daemon/events.py` | All event + triage models |
| `src/coordinator/event_prompt.py` | Prompt rendering for coordinator |
| `src/prompts/coordinator/events/` | Per-type prompt templates |
| `src/event_queue/service.py` | `MarketEventQueue.enqueue()` |
| `src/event_queue/consumer.py` | `EventQueueConsumer` — dequeues + triggers coordinator |
| `src/agents/event_triage.py` | `EventTriageAgent` — LLM triage |
