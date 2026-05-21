import base64
import io
import time
from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

router = APIRouter()

class AudioAnalysisResponse(BaseModel):
    predictions: Dict[str, float]
    anomaly_score: float
    timestamp: str

@router.post("/analyze", response_model=AudioAnalysisResponse)
async def analyze_audio(
    request: Request,
    file: Optional[UploadFile] = File(None),
    data: Optional[Dict[str, str]] = None
):
    """
    Input: base64-encoded audio chunk or file upload
    Output: {predictions: {fire, chainsaw, gunshot, rain, wildlife, human}, anomaly_score, timestamp}
    """
    audio_bytes = None
    
    if file:
        audio_bytes = await file.read()
    elif data and "audio" in data:
        try:
            audio_bytes = base64.b64decode(data["audio"])
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 encoding")
    
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="No audio data provided")

    # Get model from app state
    audio_model = request.app.state.audio_model
    if not audio_model:
        raise HTTPException(status_code=503, detail="Audio model not loaded")

    try:
        # CRNN inference
        raw_results = audio_model.predict(audio_bytes)
        
        # Format results as requested: {fire, chainsaw, gunshot, rain, wildlife, human}
        # Based on trainer.py, labels are: ['fire', 'chainsaw', 'gunshot', 'rain', 'wildlife', 'human']
        # Ensure we have all classes even if 0.0
        predictions = {label: 0.0 for label in audio_model.labels}
        for res in raw_results:
            predictions[res["label"]] = res["confidence"]
            
        # Simplified anomaly score based on high-threat detections
        anomaly_score = max(predictions.get("fire", 0), predictions.get("chainsaw", 0), predictions.get("gunshot", 0))
        
        return {
            "predictions": predictions,
            "anomaly_score": anomaly_score,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
