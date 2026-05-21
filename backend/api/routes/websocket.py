import json
import os
import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from backend.services.risk_service.event_bus import event_bus

router = APIRouter()

@router.websocket("/live")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # 1. Send initial status
    try:
        registry_path = os.path.join(os.path.dirname(__file__), "..", "sensor_registry.json")
        if os.path.exists(registry_path):
            with open(registry_path, "r") as f:
                sensors = json.load(f)
            await websocket.send_json({
                "event": "initial_status",
                "sensors": sensors
            })
    except Exception as e:
        print(f"Error sending initial status: {e}")

    # 2. Subscribe to event bus
    event_bus.subscribe(websocket)
    
    try:
        # Keep connection open and wait for messages (though we mostly push)
        while True:
            # We can receive ping/pong or config from client if needed
            data = await websocket.receive_text()
            # For now, just keep alive
    except WebSocketDisconnect:
        event_bus.unsubscribe(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        event_bus.unsubscribe(websocket)
