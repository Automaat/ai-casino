"""Queue observability endpoints."""

from fastapi import APIRouter, Request

from src.daemon.api.models import QueueEventItem, QueueEventsResponse, QueueStatsResponse, QueueTypeBreakdown
from src.daemon.api.routers.shared import get_components
from src.event_queue.service import MarketEventQueue

router = APIRouter(tags=["queue"])


@router.get("/queue/stats", response_model=QueueStatsResponse)
async def get_queue_stats(request: Request) -> QueueStatsResponse:
    """Get queue statistics (pending, stale, consumed 24h, total, by type)."""
    components = get_components(request)
    queue: MarketEventQueue = components.container.market_event_queue()
    queue_stats = await queue.stats()
    return QueueStatsResponse(
        pending_count=queue_stats.pending_count,
        stale_count=queue_stats.stale_count,
        consumed_count_24h=queue_stats.consumed_count_24h,
        total_in_db=queue_stats.total_in_db,
        by_type=[QueueTypeBreakdown(event_type=et, count=cnt) for et, cnt in queue_stats.by_type.items()],
    )


@router.get("/queue/events", response_model=QueueEventsResponse)
async def get_queue_events(request: Request, limit: int = 100, status: str = "all") -> QueueEventsResponse:
    """List queue events filtered by status (all/pending/consumed/expired)."""
    components = get_components(request)
    queue: MarketEventQueue = components.container.market_event_queue()
    events = await queue.list_events(limit=limit, status=status)
    return QueueEventsResponse(
        events=[
            QueueEventItem(
                event_id=e.event_id,
                event_type=e.event_type,
                status=e.status,
                enqueued_at=e.enqueued_at,
                expires_at=e.expires_at,
                consumed_at=e.consumed_at,
                symbols=e.symbols,
                urgency=e.urgency,
                sentiment=e.sentiment,
                confidence=e.confidence,
                reasoning=e.reasoning,
                ttl_remaining_seconds=e.ttl_remaining_seconds,
            )
            for e in events
        ],
        returned_count=len(events),
    )
