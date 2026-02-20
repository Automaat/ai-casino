# Adding New Watchers

Watchers live in `src/v1/watchers/`. Two kinds:

- **Event-driven** — poll external sources (news, social, options), triage via LLM, emit to `MarketEventQueue`
- **Signal-provider** — compute a signal (sentiment, flow direction) consumed by other components

---

## Checklist

### 1. Implement the watcher class — `src/v1/watchers/<name>_watcher.py`

**Event-driven watcher** (emits to queue):

```python
class MyWatcher(PeriodicWatcher):
    """One-line docstring."""

    def __init__(
        self,
        pipeline: EventTriagePipeline,
        config: MyWatcherConfig,
        # ... source-specific dependencies
    ) -> None:
        super().__init__(poll_interval=config.poll_interval)
        self._pipeline = pipeline
        # ...

    @property
    def name(self) -> str:
        return "MyWatcher"

    async def _tick(self) -> None:
        events = await self._fetch_events()
        if events:
            await self._pipeline.process(events)

    async def _fetch_events(self) -> list[BaseEvent]:
        # ... poll source, return list of BaseEvent subclasses
```

**Signal-provider watcher** (no queue, no pipeline):

```python
class MySignalWatcher(PeriodicWatcher):
    """One-line docstring."""

    def __init__(self, config: MySignalWatcherConfig, fetcher: MyFetcher) -> None:
        super().__init__(poll_interval=config.poll_interval)
        self._fetcher = fetcher
        self._signals: dict[str, MySignal] = {}

    @property
    def name(self) -> str:
        return "MySignalWatcher"

    async def _tick(self) -> None:
        # ... fetch + compute signals, store in self._signals

    def get_signal(self, symbol: str) -> MySignal | None:
        return self._signals.get(symbol)
```

### 2. Define the config class — `src/daemon/config/`

```python
class MyWatcherConfig(BaseModel):
    """Config for MyWatcher."""
    enabled: bool = False
    poll_interval_minutes: int = Field(default=15, ge=1, le=60)
    # ... watcher-specific fields
```

Add to `DaemonConfig` in `src/daemon/config/__init__.py`.

Update `docs/daemon.yaml.example` with the new config section.

### 3. Wire DI provider — `src/di/providers/watchers.py`

**Event-driven:**

```python
def create_my_watcher(
    historical_cache: HistoricalCache,
    daemon_config: DaemonConfig,
    container: AppContainer | None = None,
    state: DaemonState | None = None,
) -> MyWatcher | None:
    config = daemon_config.my_watcher
    if not config.enabled:
        return None
    if container is None:
        from src.di.container import create_container
        container = create_container()
    pipeline = _build_pipeline(daemon_config, container, state)
    # ... create fetcher, build config, return MyWatcher(pipeline=pipeline, ...)
```

**Signal-provider:**

```python
def create_my_signal_watcher(
    daemon_config: DaemonConfig,
) -> MySignalWatcher | None:
    config = daemon_config.my_signal_watcher
    if not config.enabled:
        return None
    # ... create deps, return MySignalWatcher(...)
```

### 4. Register in container — `src/di/container.py`

```python
my_watcher = providers.Singleton(
    watcher_providers.create_my_watcher,
    historical_cache=historical_cache,
    daemon_config=daemon_config,
)
```

### 5. Register in `DaemonComponents` + `DaemonFactory` + `DaemonLifecycle`

- `DaemonComponents` (`src/daemon/factory.py`): add `my_watcher: MyWatcher | None = None`
- `DaemonFactory._create_components()`: create and assign `components.my_watcher`
- `DaemonLifecycle._start_watchers()` / `_stop_watchers()`: call `watcher.run()` / `watcher.stop()`

### 6. Export from `src/v1/watchers/__init__.py`

```python
from src.v1.watchers.my_watcher import MyWatcher, MyWatcherConfig
__all__ = [..., "MyWatcher", "MyWatcherConfig"]
```

### 7. For event-driven watchers: define the event model

See [adding-new-event-types.md](./adding-new-event-types.md) for the event model + triage prompt checklist.

---

## Key files

| File | Purpose |
|---|---|
| `src/v1/watchers/base.py` | `Watcher` ABC + `PeriodicWatcher` (1s-granularity sleep loop) |
| `src/v1/watchers/pipeline.py` | `EventTriagePipeline` — triage + enqueue to `MarketEventQueue` |
| `src/v1/watchers/__init__.py` | Public API exports |
| `src/di/providers/watchers.py` | Factory functions; `_build_pipeline()` helper |
| `src/daemon/factory.py` | `DaemonComponents` dataclass + `DaemonFactory` |
| `src/daemon/lifecycle.py` | Watcher start/stop orchestration |
