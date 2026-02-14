"""WebSocket event streaming endpoint."""

import asyncio
import queue

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger

router = APIRouter(tags=["websocket"])


@router.websocket("/ws/events")
async def websocket_events(websocket: WebSocket) -> None:
    """Stream real-time events to dashboard."""
    # Access components via websocket.app.state
    components = websocket.app.state.components

    # Accept connection first (FastAPI CORSMiddleware doesn't handle WebSocket properly)
    await websocket.accept()

    # Validate origin after accepting
    origin = websocket.headers.get("origin")
    allowed_origins = components.config.api.cors_origins

    if not components.event_bus:
        logger.warning("WebSocket rejected - EventBus not available")
        await websocket.close(code=1011, reason="EventBus not available")
        return

    # Explicitly check for None origin (security monitoring)
    if origin is None:
        logger.warning("WebSocket rejected - null origin (potential security issue)")
        await websocket.close(code=1008, reason="Invalid origin")
        return

    if origin not in allowed_origins:
        logger.warning(f"WebSocket rejected - invalid origin: {origin}")
        await websocket.close(code=1008, reason="Invalid origin")
        return

    logger.info(f"WebSocket connected from {origin}")

    subscriber_id, event_queue = await components.event_bus.subscribe()

    try:
        while True:
            try:
                event = await asyncio.to_thread(event_queue.get, block=True, timeout=30.0)
                event_dict = event.model_dump(mode="json")
                await websocket.send_json(event_dict)
            except queue.Empty:
                # Ping client to detect disconnect
                try:
                    await websocket.send_json({"type": "ping"})
                except WebSocketDisconnect:
                    logger.info("Client disconnected during ping")
                    break
                except Exception as e:
                    logger.opt(exception=True).error(f"Unexpected error during ping: {e}")
                    break

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {websocket.client}")
    except Exception as e:
        logger.opt(exception=True).error(f"WebSocket error: {e}")
    finally:
        await components.event_bus.unsubscribe(subscriber_id)
        logger.info(f"Unsubscribed: {subscriber_id}")
