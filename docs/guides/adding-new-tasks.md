# Adding New Tasks (v1 Framework)

Tasks live in `src/v1/tasks/implementations/`. Each task is a self-contained unit with its own schedule, execute logic, and dedup.

---

## Checklist

### 1. Implement `Task` subclass — `src/v1/tasks/implementations/<name>.py`

```python
class MyTask(Task):
    """One-line docstring."""

    def __init__(self, dep_a: DepA, config: MyConfig) -> None:
        self._dep_a = dep_a
        self._config = config

    @property
    def name(self) -> str:
        return "my_task"

    @property
    def schedule(self) -> TaskSchedule:
        return TaskSchedule(
            time=self._config.run_time,        # "HH:MM"
            days=WEEKDAYS,                      # or custom [DayOfWeek.MON, ...]
            enabled=self._config.enabled,
            dedup=DedupStrategy.DAILY,          # DAILY | INTERVAL | NONE
            window_minutes=5,                   # trigger window
        )

    async def execute(self) -> TaskResult:
        start = time.monotonic()
        # ... do work ...
        duration = time.monotonic() - start
        return TaskResult(
            task_name=self.name,
            success=True,
            duration_seconds=duration,
            message="summary of what happened",
        )

    async def last_run_at(self) -> datetime | None:
        return await self._state.get_last_my_task()
```

### 2. Define schedule via `TaskSchedule`

- `time`: `"HH:MM"` in daemon timezone
- `days`: `WEEKDAYS` or custom list of `DayOfWeek`
- `dedup`: `DAILY` (skip if ran today), `INTERVAL` (skip if ran within N min), `NONE` (always run)
- `window_minutes`: how long after scheduled time the task can still trigger (default 5)

### 3. Wire in `DaemonRunner._build_v1_task_runner()`

```python
# In src/daemon/runner.py, _build_v1_task_runner()
my_config = self.config.my_task
if my_config.enabled:
    tasks.append(MyTask(
        dep_a=self._container.dep_a(),
        config=my_config,
    ))
```

### 4. Add config (if new config fields)

- Add config model in `src/daemon/config/`
- Add field to `DaemonConfig`
- Update `docs/daemon.yaml.example`

### 5. Write tests — `tests/test_v1/test_tasks/test_<name>.py`

Test the task's `execute()` with mocked dependencies. Test schedule properties.

### 6. Remove from old TASKS registry (if migrating)

Remove the `ScheduledTask(...)` entry from `ScheduledTaskRunner.TASKS` in `src/daemon/task_runner.py`.

---

## Reference implementation

`GamePlanTask` (`src/v1/tasks/implementations/game_plan.py`) — daily pre-market game plan generation.

---

## Key files

| File | Purpose |
|---|---|
| `src/v1/tasks/interface.py` | `Task` ABC |
| `src/v1/tasks/models.py` | `TaskSchedule`, `TaskResult`, `DayOfWeek`, `DedupStrategy` |
| `src/v1/tasks/runner.py` | `TaskRunner` — evaluates schedules, runs due tasks |
| `src/v1/tasks/scheduling.py` | `in_schedule_window()`, `should_skip()` helpers |
| `src/v1/tasks/implementations/` | Task implementations |
| `src/daemon/runner.py` | `_build_v1_task_runner()` — DI wiring |
