import time
import base64
from fastapi import APIRouter, HTTPException, Request, Body
from pydantic import BaseModel
from typing import Dict, Any, Optional
from datetime import datetime

from backend.services.risk_service.rolling_buffer import rolling_buffer
from backend.services.risk_service.event_bus import event_bus

router = APIRouter()

class PipelineRequest(BaseModel):
    audio_base64: str
    temperature: float
    humidity: float
    wind_speed: float
    rainfall: float
    sensor_id: str
    region: str
    latitude: float
    longitude: float

class PipelineResponse(BaseModel):
    audio_predictions: Dict[str, float]
    risk_assessment: Dict[str, Any]
    alert_triggered: bool
    latency_ms: float

@router.post("/analyze", response_model=PipelineResponse)
async def analyze_pipeline(request: Request, body: PipelineRequest):
    """
    Primary endpoint for IoT sensor integration.
    calls audio inference → extracts confidences → builds structured feature vector → calls risk model
    """
    start_time = time.time()
    
    # 1. Audio Inference
    try:
        audio_bytes = base64.b64decode(body.audio_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid audio base64")

    audio_model = request.app.state.audio_model
    if not audio_model:
        raise HTTPException(status_code=503, detail="Audio model not loaded")

    audio_results = audio_model.predict(audio_bytes)
    audio_preds = {label: 0.0 for label in audio_model.labels}
    for res in audio_results:
        audio_preds[res["label"]] = res["confidence"]

    # 2. Update Rolling Buffer
    rolling_buffer.add_observation(body.sensor_id, audio_preds)
    
    # 3. Build Structured Features
    now = datetime.now()
    
    # Get rolling averages
    avg_5m = rolling_buffer.get_averages(body.sensor_id, 5)
    avg_10m = rolling_buffer.get_averages(body.sensor_id, 10)
    avg_30m = rolling_buffer.get_averages(body.sensor_id, 30)
    
    features = {
        "temperature": body.temperature,
        "humidity": body.humidity,
        "wind_speed": body.wind_speed,
        "rainfall": body.rainfall,
        "hour_of_day": now.hour,
        "day_of_week": now.weekday(),
        "season": get_season(now.month),
        "region": body.region,
        "latitude": body.latitude,
        "longitude": body.longitude,
        "chainsaw_confidence": audio_preds.get("chainsaw", 0.0),
        "fire_confidence": audio_preds.get("fire", 0.0),
        "gunshot_confidence": audio_preds.get("gunshot", 0.0),
        "wildlife_activity_score": audio_preds.get("wildlife", 0.0),
        "rolling_chainsaw_5m": avg_5m["chainsaw"],
        "rolling_chainsaw_10m": avg_10m["chainsaw"],
        "rolling_chainsaw_30m": avg_30m["chainsaw"],
        "rolling_fire_5m": avg_5m["fire"],
        "rolling_fire_10m": avg_10m["fire"],
        "rolling_fire_30m": avg_30m["fire"],
        "detection_frequency_1h": 1, # Placeholder
        "anomaly_score": max(audio_preds.get("fire", 0), audio_preds.get("chainsaw", 0), audio_preds.get("gunshot", 0))
    }
    
    # 4. Risk Assessment
    risk_model = request.app.state.risk_model
    if not risk_model:
        raise HTTPException(status_code=503, detail="Risk model not loaded")

    risk_result = risk_model.predict(features)
    
    latency_ms = (time.time() - start_time) * 1000
    alert_triggered = risk_result.get("risk_level") in ["MEDIUM", "HIGH"]
    
    # 5. Broadcast results via WebSocket
    timestamp = now.isoformat()
    broadcast_data = {
        "event": "inference_result",
        "sensor_id": body.sensor_id,
        "region": body.region,
        "timestamp": timestamp,
        "audio_predictions": audio_preds,
        "risk_level": risk_result.get("risk_level"),
        "probabilities": risk_result.get("probabilities"),
        "alert_triggered": alert_triggered,
        "lat": body.latitude,
        "lon": body.longitude
    }
    await event_bus.publish(broadcast_data)
    
    if alert_triggered:
        await event_bus.publish({
            "event": "alert",
            "sensor_id": body.sensor_id,
            "risk_level": risk_result.get("risk_level"),
            "timestamp": timestamp
        })

    return {
        "audio_predictions": audio_preds,
        "risk_assessment": risk_result,
        "alert_triggered": alert_triggered,
        "latency_ms": round(latency_ms, 2)
    }

def get_season(month: int) -> str:
    if month in [12, 1, 2]: return "Winter"
    if month in [3, 4, 5]: return "Spring"
    if month in [6, 7, 8]: return "Summer"
    return "Autumn"
