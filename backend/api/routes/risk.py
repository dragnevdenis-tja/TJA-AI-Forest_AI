import time
from fastapi import APIRouter, HTTPException, Request, Query
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

router = APIRouter()

# Simple in-memory history for demo purposes
# In production, this would be a database
risk_history: Dict[str, List[Dict[str, Any]]] = {}

class RiskAssessmentRequest(BaseModel):
    temperature: float
    humidity: float
    wind_speed: float
    rainfall: float
    hour_of_day: int
    day_of_week: int
    season: str
    region: str
    latitude: float
    longitude: float
    chainsaw_confidence: float
    fire_confidence: float
    gunshot_confidence: float
    wildlife_activity_score: float
    rolling_chainsaw_5m: float
    rolling_chainsaw_10m: float
    rolling_chainsaw_30m: float
    rolling_fire_5m: float
    rolling_fire_10m: float
    rolling_fire_30m: float
    detection_frequency_1h: int
    anomaly_score: float
    sensor_id: str

class RiskAssessmentResponse(BaseModel):
    risk_level: str
    probabilities: Dict[str, float]
    timestamp: str

@router.post("/assess", response_model=RiskAssessmentResponse)
async def assess_risk(request: Request, body: RiskAssessmentRequest):
    """
    Input: full structured feature dict
    Output: {risk_level, probabilities, timestamp}
    """
    risk_model = request.app.state.risk_model
    if not risk_model:
        raise HTTPException(status_code=503, detail="Risk model not loaded")

    try:
        features = body.dict()
        sensor_id = features.pop("sensor_id")
        
        # Predict
        result = risk_model.predict(features)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
            
        result["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        
        # Store in history
        if sensor_id not in risk_history:
            risk_history[sensor_id] = []
        risk_history[sensor_id].append(result)
        # Keep only last 100
        if len(risk_history[sensor_id]) > 100:
            risk_history[sensor_id].pop(0)
            
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@router.get("/history")
async def get_risk_history(
    sensor_id: str,
    limit: int = Query(50, ge=1, le=100)
):
    """
    Last N assessments for a sensor
    """
    if sensor_id not in risk_history:
        return []
        
    return risk_history[sensor_id][-limit:]
