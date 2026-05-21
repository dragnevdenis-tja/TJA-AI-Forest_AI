import asyncio
import logging
import threading
from typing import List, Set, Any
from fastapi import WebSocket

logger = logging.getLogger("forest_audio.event_bus")

class EventBus:
    """
    Simple pub/sub in-memory event bus for WebSockets.
    """
    def __init__(self):
        self._subscribers: Set[WebSocket] = set()
        self._lock = threading.Lock()

    def subscribe(self, websocket: WebSocket):
        with self._lock:
            self._subscribers.add(websocket)
            logger.info(f"WebSocket subscribed. Total: {len(self._subscribers)}")

    def unsubscribe(self, websocket: WebSocket):
        with self._lock:
            if websocket in self._subscribers:
                self._subscribers.remove(websocket)
                logger.info(f"WebSocket unsubscribed. Total: {len(self._subscribers)}")

    async def publish(self, event: dict):
        """
        Sends event to all WebSocket subscribers.
        """
        if not self._subscribers:
            return

        # Use a copy of subscribers to avoid issues with removal during iteration
        with self._lock:
            active_subscribers = list(self._subscribers)

        # Broadcast to all
        tasks = []
        for ws in active_subscribers:
            tasks.append(self._send_event(ws, event))
        
        if tasks:
            await asyncio.gather(*tasks)

    async def _send_event(self, websocket: WebSocket, event: dict):
        try:
            await websocket.send_json(event)
        except Exception as e:
            logger.error(f"Error sending to WebSocket: {e}")
            self.unsubscribe(websocket)

# Global instance
event_bus = EventBus()
