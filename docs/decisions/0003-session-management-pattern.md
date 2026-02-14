# ADR 0003: SQLAlchemy Async Session Management Pattern

## Status

Accepted

## Context

SQLAlchemy async sessions are mutable, stateful objects representing single database transactions. They are NOT thread-safe or task-safe for concurrent operations.

Key characteristics:

- Sessions maintain connection state and transaction context
- Sessions should not be shared across concurrent requests/tasks
- Connection pools are managed by the engine (singleton), not sessions
- Async sessions require `expire_on_commit=False` to prevent lazy-loading issues
- Sessions must be properly closed to return connections to the pool

## Decision

We will use two distinct patterns for session management:

### Pattern 1: Session-per-request (FastAPI endpoints)

All API endpoints receive sessions via dependency injection using FastAPI's `Depends()`:

```python
@app.get("/positions")
async def get_positions(session: AsyncSession = Depends(get_db_session)):
    repo = PositionRecordRepository(session)
    return await repo.get_all_active()
```

**Rationale:** FastAPI's dependency injection system automatically creates and closes sessions per request, preventing session sharing across concurrent requests.

### Pattern 2: Inline session creation (background tasks)

Background tasks and daemon lifecycle methods create sessions inline using context managers:

```python
async def scheduled_task():
    async with get_db_engine().session() as session:
        repo = AnalysisRepository(session)
        await repo.create(analysis)
```

**Rationale:** Background tasks don't have request-scoped lifecycle, so they must manage sessions explicitly with context managers.

### Supporting patterns

**Repository Pattern:**

- Repositories receive sessions via `__init__` parameter
- Repository instances can be reused, but sessions cannot
- Never create sessions inside repositories

**State Manager Pattern:**

- API-facing methods accept `session: AsyncSession` parameter
- Background task methods create sessions inline
- Never store sessions or engines as instance variables

## Consequences

### Positive

- **Prevents concurrency errors:** Each request/task gets isolated session
- **Follows SQLAlchemy best practices:** Official documentation recommends session-per-request
- **Clear separation of concerns:** API pattern vs background task pattern explicit
- **Testability:** Easy to mock sessions in unit tests
- **Connection pooling:** Engine singleton manages pool efficiently

### Negative

- **Slightly more verbose:** Explicit session passing requires additional parameter
- **Learning curve:** Developers must understand when to use each pattern
- **Boilerplate:** Dependency injection setup requires initial configuration

### Neutral

- **Migration effort:** Existing code using stored sessions must be refactored
- **Documentation overhead:** Team must maintain clear guidelines

## Alternatives Considered

### Alternative 1: Scoped sessions

**Rejected:** `scoped_session()` uses thread-local storage, which doesn't work reliably with asyncio tasks. Not recommended for async contexts.

### Alternative 2: Singleton session

**Rejected:** Would cause "concurrent operations not permitted" errors when handling parallel requests.

### Alternative 3: Session middleware

**Rejected:** While possible, FastAPI's dependency injection is more explicit and type-safe.

## References

- [SQLAlchemy Async Documentation](https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html)
- [FastAPI Dependencies with Yield](https://fastapi.tiangolo.com/tutorial/dependencies/dependencies-with-yield/)
- [Database Session Management Best Practices](https://deepwiki.com/fastapi-practices/fastapi_best_architecture/7.6-database-session-management)
- Related PRs: #620, #621

## Date

2026-02-14
