import base64
import os
import pytest
from fastapi.testclient import TestClient
from backend.main import app

def test_websocket_broadcast():
    # Use with block to trigger lifespan events
    with TestClient(app) as client:
        # 1. Connect to WebSocket
        with client.websocket_connect("/live") as websocket:
            # 2. Check initial status message
            data = websocket.receive_json()
            assert data["event"] == "initial_status"
            assert "sensors" in data
            
            # 3. Trigger analysis via REST API
            sample_path = "audio_samples/AuenwaldWasser.mp3"
            if os.path.exists(sample_path):
                with open(sample_path, "rb") as f:
                    audio_b64 = base64.b64encode(f.read()).decode()
            else:
                audio_b64 = base64.b64encode(b"dummy audio data").decode()

            payload = {
                "audio_base64": audio_b64,
                "temperature": 25.0,
                "humidity": 30.0,
                "wind_speed": 5.0,
                "rainfall": 0.0,
                "sensor_id": "SN-CHI-101",
                "region": "Chisinau",
                "latitude": 47.0105,
                "longitude": 28.8638
            }
            
            response = client.post("/api/pipeline/analyze", json=payload)
            assert response.status_code == 200
            
            # 4. Assert WebSocket receives the inference_result event
            ws_data = websocket.receive_json()
            assert ws_data["event"] == "inference_result"
            assert ws_data["sensor_id"] == "SN-CHI-101"
            assert "risk_level" in ws_data
            assert "audio_predictions" in ws_data

if __name__ == "__main__":
    pytest.main([__file__])
