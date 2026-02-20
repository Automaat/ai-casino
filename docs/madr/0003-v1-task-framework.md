# MADR 0003 — V1 Task Framework

**Status:** Accepted
**Date:** 2026-02-20

---

## Context

Current task infrastructure has 14 scheduled tasks across 3 layers of indirection (`TASKS registry` → `DaemonTaskService.run_X()` → `TaskExecutor.run()`). Problems:

1. **God object scheduler** — 990-line `MarketScheduler` with 15 near-identical `is_*_time()` methods
2. **String-based dispatch** — `ScheduledTask.check_method` → `getattr(scheduler, method_name)()` → fragile, no type safety
3. **Inconsistent enable checks** — some tasks check `enabled` in TASKS registry, others in `TaskExecutor`, some in both
4. **Dead/unwired tasks** — 5 tasks in TASKS registry have no corresponding `DaemonTaskService` method
5. **No retry** — tasks fail silently, no structured result tracking
6. **Fragile trigger window** — 1-minute scheduler resolution means tasks can miss their window

Adding a new task requires touching 4+ files: scheduler method, TASKS entry, task service method, TaskExecutor subclass.

---

## Decisions

### 1. New `src/v1/tasks/` package with `Task` ABC

**Decision:** Create `Task` ABC where each task is self-contained — schedule as data (not methods), execute logic, dedup via `last_run_at()`.

**Rationale:** Schedule-as-data eliminates the need for `is_*_time()` methods. Self-contained tasks reduce the surface area for adding new tasks from 4+ files to 1 file + DI wiring.

### 2. `TaskRunner` replaces `ScheduledTaskRunner` for new tasks

**Decision:** `TaskRunner` iterates registered `Task` instances, evaluates their `TaskSchedule`, runs due tasks, returns `TaskResult` list. No string dispatch, no mediator.

**Alternatives considered:**
- Keep `TaskExecutor` + fix scheduler — rejected: fundamental architecture (string dispatch, mediator pattern) remains fragile
- Watcher-based approach (like `PeriodicWatcher`) — rejected: watchers are long-running loops, tasks are one-shot executions

### 3. `TaskSchedule` model with `DedupStrategy`

**Decision:** Schedule is a Pydantic model with `time`, `days`, `dedup` strategy, and configurable `window_minutes` (default 5, vs current 1-min resolution).

**Rationale:** Configurable window prevents missed triggers. `DedupStrategy` (DAILY, INTERVAL, NONE) replaces ad-hoc dedup checks scattered across task implementations.

### 4. Task implementations in `src/v1/tasks/implementations/`

**Decision:** Each task is a separate file in `implementations/`. `GamePlanTask` is the reference implementation.

### 5. Incremental migration — old + new runners coexist

**Decision:** `DaemonCycleOrchestrator` calls both `task_runner.run_scheduled_tasks()` (old) and `v1_task_runner.tick()` (new). Tasks migrate one by one: remove from old TASKS → add as v1 Task.

**Rationale:** Big-bang migration is risky with 14 tasks. Dual-runner allows incremental migration with rollback safety.

---

## Consequences

- `game_plan` removed from old `ScheduledTaskRunner.TASKS` registry
- `GamePlanTask` now runs via v1 `TaskRunner` with agentic tool-calling agent
- `generate_game_plan` tool removed from coordinator (game plan is a pre-market task, not coordinator responsibility)
- Coordinator still reads today's game plan from memory (no behavior change)
- Future tasks should be added via v1 framework, not old `ScheduledTaskRunner`
