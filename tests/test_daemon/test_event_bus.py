"""Tests for EventBus real-time event streaming."""

import asyncio

import pytest

from src.daemon.event_bus import DashboardEvent, EventBus, EventType


async def test_eventbus_subscribe_unsubscribe():
    """Test subscriber lifecycle and count tracking."""
    bus = EventBus(history_size=50, queue_size=10)

    assert bus.get_subscriber_count() == 0

    sub1_id, sub1_queue = await bus.subscribe()
    assert bus.get_subscriber_count() == 1
    assert isinstance(sub1_id, str)
    assert isinstance(sub1_queue, asyncio.Queue)

    sub2_id, _sub2_queue = await bus.subscribe()
    assert bus.get_subscriber_count() == 2
    assert sub1_id != sub2_id

    await bus.unsubscribe(sub1_id)
    assert bus.get_subscriber_count() == 1

    await bus.unsubscribe(sub2_id)
    assert bus.get_subscriber_count() == 0

    await bus.unsubscribe("nonexistent")
    assert bus.get_subscriber_count() == 0


async def test_eventbus_publish_fanout():
    """Test event reaches all subscribers."""
    bus = EventBus(history_size=50, queue_size=10)

    sub1_id, sub1_queue = await bus.subscribe()
    sub2_id, sub2_queue = await bus.subscribe()
    sub3_id, sub3_queue = await bus.subscribe()

    event = DashboardEvent(event_type=EventType.CYCLE_START, data={"watchlist_size": 5})

    await bus.publish(event)

    event1 = await asyncio.wait_for(sub1_queue.get(), timeout=1.0)
    event2 = await asyncio.wait_for(sub2_queue.get(), timeout=1.0)
    event3 = await asyncio.wait_for(sub3_queue.get(), timeout=1.0)

    assert event1.event_type == EventType.CYCLE_START
    assert event2.event_type == EventType.CYCLE_START
    assert event3.event_type == EventType.CYCLE_START

    assert event1.event_id == event.event_id
    assert event2.event_id == event.event_id
    assert event3.event_id == event.event_id

    assert event1.data == {"watchlist_size": 5}
    assert event2.data == {"watchlist_size": 5}
    assert event3.data == {"watchlist_size": 5}

    await bus.unsubscribe(sub1_id)
    await bus.unsubscribe(sub2_id)
    await bus.unsubscribe(sub3_id)


async def test_eventbus_queue_full_drops():
    """Test drop behavior when subscriber queue is full."""
    bus = EventBus(history_size=50, queue_size=5)

    sub_id, sub_queue = await bus.subscribe()

    for i in range(10):
        event = DashboardEvent(event_type=EventType.ANALYSIS_START, data={"symbol": f"TICK{i}"})
        await bus.publish(event)

    received_count = 0
    try:
        while True:
            await asyncio.wait_for(sub_queue.get(), timeout=0.1)
            received_count += 1
    except TimeoutError:
        pass

    assert received_count == 5

    history = bus.get_history()
    assert len(history) == 10

    await bus.unsubscribe(sub_id)


async def test_eventbus_history():
    """Test history deque maxlen retention."""
    bus = EventBus(history_size=10, queue_size=10)

    for i in range(15):
        event = DashboardEvent(event_type=EventType.CYCLE_START, data={"iteration": i})
        await bus.publish(event)

    history = bus.get_history()
    assert len(history) == 10

    assert history[0].data == {"iteration": 14}
    assert history[9].data == {"iteration": 5}


async def test_eventbus_history_limit():
    """Test get_history with limit parameter."""
    bus = EventBus(history_size=50, queue_size=10)

    for i in range(20):
        event = DashboardEvent(event_type=EventType.ANALYSIS_COMPLETE, data={"index": i})
        await bus.publish(event)

    history_all = bus.get_history()
    assert len(history_all) == 20

    history_5 = bus.get_history(limit=5)
    assert len(history_5) == 5
    assert history_5[0].data == {"index": 19}
    assert history_5[4].data == {"index": 15}

    history_100 = bus.get_history(limit=100)
    assert len(history_100) == 20

    history_0 = bus.get_history(limit=0)
    assert len(history_0) == 0


async def test_eventbus_concurrent_subscribers():
    """Test multiple concurrent subscribers receiving events."""
    bus = EventBus(history_size=50, queue_size=20)

    subscriber_count = 5
    subscribers = []

    for _ in range(subscriber_count):
        sub_id, sub_queue = await bus.subscribe()
        subscribers.append((sub_id, sub_queue))

    event_count = 10
    for i in range(event_count):
        event = DashboardEvent(event_type=EventType.HEALTH_CHECK, data={"check": i})
        await bus.publish(event)

    for _sub_id, sub_queue in subscribers:
        received = []
        for _ in range(event_count):
            event = await asyncio.wait_for(sub_queue.get(), timeout=1.0)
            received.append(event)

        assert len(received) == event_count
        assert all(e.event_type == EventType.HEALTH_CHECK for e in received)

    for sub_id, _ in subscribers:
        await bus.unsubscribe(sub_id)

    assert bus.get_subscriber_count() == 0


async def test_eventbus_different_event_types():
    """Test publishing different event types."""
    bus = EventBus(history_size=50, queue_size=10)

    sub_id, sub_queue = await bus.subscribe()

    events = [
        DashboardEvent(event_type=EventType.CYCLE_START, data={"watchlist_size": 3}),
        DashboardEvent(event_type=EventType.ANALYSIS_START, data={"symbol": "AAPL"}),
        DashboardEvent(event_type=EventType.ANALYSIS_COMPLETE, data={"signal": "BUY"}),
        DashboardEvent(event_type=EventType.TRADE_EXECUTED, data={"order_id": "123"}),
        DashboardEvent(event_type=EventType.DEGRADATION, data={"tier": "DEGRADED"}),
    ]

    for event in events:
        await bus.publish(event)

    received = []
    for _ in range(len(events)):
        event = await asyncio.wait_for(sub_queue.get(), timeout=1.0)
        received.append(event)

    assert len(received) == len(events)
    assert [e.event_type for e in received] == [e.event_type for e in events]

    await bus.unsubscribe(sub_id)


async def test_eventbus_publish_exception_handling():
    """Test that publish() never raises exceptions."""
    bus = EventBus(history_size=50, queue_size=10)

    event = DashboardEvent(event_type=EventType.STATE_UPDATE, data={"key": "value"})

    await bus.publish(event)

    history = bus.get_history()
    assert len(history) == 1
    assert history[0].event_type == EventType.STATE_UPDATE


async def test_eventbus_empty_history():
    """Test empty history returns empty list."""
    bus = EventBus(history_size=50, queue_size=10)

    history = bus.get_history()
    assert history == []

    history_limited = bus.get_history(limit=10)
    assert history_limited == []


async def test_eventbus_repr():
    """Test string representation."""
    bus = EventBus(history_size=100, queue_size=20)

    assert "EventBus" in repr(bus)
    assert "subscribers=0" in repr(bus)
    assert "history_size=0/100" in repr(bus)

    sub_id, _ = await bus.subscribe()

    assert "subscribers=1" in repr(bus)

    event = DashboardEvent(event_type=EventType.CYCLE_START, data={})
    await bus.publish(event)

    assert "history_size=1/100" in repr(bus)

    await bus.unsubscribe(sub_id)
