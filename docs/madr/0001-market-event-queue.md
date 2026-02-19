# Market Event Queue

## Status

Accepted

## Context and Problem Statement

EventWatchers poll real-time market signals (news, social, anomalies, Trump posts) and triage them
as IMMEDIATE/WATCHLIST/IGNORE. The coordinator runs on a fixed interval (every N minutes) and has
no visibility into events that arrived since the last cycle. High-urgency signals are delayed by up
to the full cycle interval before the coordinator can act on them.

## Decision Drivers

- Events must survive daemon restarts (coordinator restart must not lose queued signals)
- Deduplication required — same event must not be enqueued twice
- Simple consumer interface: coordinator calls a tool, gets next events
- Events should auto-expire (stale signals are worthless after a few hours)
- Minimal coupling: queue is infrastructure, watcher/coordinator wiring comes later

## Considered Options

1. In-memory asyncio.Queue
2. Interrupt-driven early wake (signal daemon sleep)
3. Extend existing discovery batch pipeline
4. PostgreSQL-backed FIFO queue (chosen)

## Decision Outcome

Chosen: **PostgreSQL-backed FIFO queue** in `src/queue/` package.

Payload stored as JSONB — event-specific fields are not modeled in schema, deserialized in code
based on `event_type`. `event_id` unique constraint provides idempotent enqueue. `consumed_at`
timestamp marks consumed rows (tombstone pattern, no deletes on read). `expires_at` enables TTL.

### Positive Consequences

- Survives restarts: events queued before crash are available after restart
- Idempotent enqueue: safe to retry, no duplicate processing
- Generic schema: adding new event types requires no migration
- FIFO guaranteed by `ORDER BY enqueued_at ASC` on pending index

### Negative Consequences

- Requires PostgreSQL (no in-memory fallback)
- Slightly higher latency than in-memory queue (DB round-trip per dequeue)

## Pros and Cons of the Options

### In-memory asyncio.Queue

- Good: zero latency, no infra dependency
- Bad: lost on restart, no dedup, no TTL without extra code

### Interrupt-driven early wake

- Good: most responsive (sub-minute reaction time)
- Bad: risk of cycle storms on burst events, complex cancellation logic, overkill for current use

### Extend discovery batch pipeline

- Good: already exists, minimal new code
- Bad: 15-minute batch lag, no FIFO guarantee, discovery is a different concern
