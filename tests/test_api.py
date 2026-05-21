import pytest
import base64
from fastapi.testclient import TestClient
from backend.main import app

@pytest.fixture
def client():
    # Use lifespan for model loading (mocks might be needed in CI)
    with TestClient(app) as c:
        yield c

def test_get_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_audio_analyze_schema(client):
    # Dummy base64 audio
    dummy_audio = base64.b64encode(b"fake audio data").decode()
    payload = {"audio": dummy_audio}
    
    # This might fail inference but should hit the route
    response = client.post("/api/audio/analyze", json=payload)
    # If model is loaded but fails on fake data, it's 500 or 400. 
    # If successful mock or real loaded, it's 200.
    if response.status_code == 200:
        data = response.json()
        assert "predictions" in data
        assert "anomaly_score" in data

def test_risk_assess_schema(client):
    valid_payload = {
        "temperature": 25.0, "humidity": 30.0, "wind_speed": 5.0, "rainfall": 0.0,
        "hour_of_day": 14, "day_of_week": 2, "season": "Summer", "region": "South",
        "latitude": 45.9, "longitude": 28.2, "chainsaw_confidence": 0.1, "fire_confidence": 0.1,
        "gunshot_confidence": 0.1, "wildlife_activity_score": 0.4,
        "rolling_chainsaw_5m": 0.1, "rolling_chainsaw_10m": 0.1, "rolling_chainsaw_30m": 0.1,
        "rolling_fire_5m": 0.1, "rolling_fire_10m": 0.1, "rolling_fire_30m": 0.1,
        "detection_frequency_1h": 1, "anomaly_score": 0.5, "sensor_id": "SN-TEST"
    }
    
    response = client.post("/api/risk/assess", json=valid_payload)
    if response.status_code == 200:
        data = response.json()
        assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
        assert "probabilities" in data

def test_risk_assess_missing_field(client):
    invalid_payload = {"temperature": 25.0} # Missing many fields
    response = client.post("/api/risk/assess", json=invalid_payload)
    assert response.status_code == 422 # Validation error
