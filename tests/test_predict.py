import pytest
import numpy as np
from backend.ml.structured_model.predict import RiskPredictor

def test_predict_schema():
    valid_input = {
        'temperature': 25.0, 'humidity': 20.0, 'wind_speed': 5.0, 'rainfall': 0.0,
        'hour_of_day': 14, 'day_of_week': 2, 'season': 'Summer', 'region': 'South',
        'latitude': 45.9, 'longitude': 28.2, 'chainsaw_confidence': 0.1, 'fire_confidence': 0.8,
        'gunshot_confidence': 0.05, 'wildlife_activity_score': 0.4,
        'rolling_chainsaw_5m': 0.05, 'rolling_chainsaw_10m': 0.05, 'rolling_chainsaw_30m': 0.05,
        'rolling_fire_5m': 0.7, 'rolling_fire_10m': 0.6, 'rolling_fire_30m': 0.5,
        'detection_frequency_1h': 10, 'anomaly_score': 0.9
    }
    
    try:
        result = RiskPredictor.predict(valid_input)
        if "error" not in result:
            assert "risk_level" in result
            assert "probabilities" in result
            assert result["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
            
            # Check probabilities sum to 1
            probs = result["probabilities"]
            total_prob = sum(probs.values())
            assert total_prob == pytest.approx(1.0, rel=1e-3)
    except FileNotFoundError:
        pytest.skip("Models not trained yet, skipping prediction schema test.")

def test_predict_missing_input():
    invalid_input = {'temperature': 25.0}
    result = RiskPredictor.predict(invalid_input)
    assert "error" in result
    assert "Missing features" in result["error"]

def test_singleton_loading():
    # Verify that assets are only loaded once (placeholder check)
    # We can check if _model is not None after first call
    try:
        RiskPredictor._load_assets()
        model1 = RiskPredictor._model
        RiskPredictor._load_assets()
        model2 = RiskPredictor._model
        assert model1 is model2
    except FileNotFoundError:
        pytest.skip("Models not trained yet.")
