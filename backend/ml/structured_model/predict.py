import joblib
import os
import pandas as pd
import numpy as np
from typing import Dict, Any

from backend.ml.structured_model.feature_engineering import engineer_features

class RiskPredictor:
    _model = None
    _preprocessor = None
    _label_map = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}
    
    @classmethod
    def _load_assets(cls):
        if cls._model is None:
            model_path = "models/risk_model.joblib"
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found at {model_path}. Run training first.")
            cls._model = joblib.load(model_path)
            
        if cls._preprocessor is None:
            preprocessor_path = "models/risk_preprocessor.joblib"
            if not os.path.exists(preprocessor_path):
                raise FileNotFoundError(f"Preprocessor not found at {preprocessor_path}. Run training first.")
            cls._preprocessor = joblib.load(preprocessor_path)

    @classmethod
    def predict(cls, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        predict(features: dict) → {"risk_level": "HIGH", "probabilities": {"LOW": 0.1, "MEDIUM": 0.2, "HIGH": 0.7}}
        """
        cls._load_assets()
        
        # 1. Validation
        required_features = [
            'temperature', 'humidity', 'wind_speed', 'rainfall',
            'hour_of_day', 'day_of_week', 'season', 'region',
            'latitude', 'longitude', 'chainsaw_confidence', 'fire_confidence',
            'gunshot_confidence', 'wildlife_activity_score',
            'rolling_chainsaw_5m', 'rolling_chainsaw_10m', 'rolling_chainsaw_30m',
            'rolling_fire_5m', 'rolling_fire_10m', 'rolling_fire_30m',
            'detection_frequency_1h', 'anomaly_score'
        ]
        
        missing = [f for f in required_features if f not in features]
        if missing:
            return {"error": f"Missing features: {', '.join(missing)}"}
            
        # 2. Convert to DataFrame
        df = pd.DataFrame([features])
        
        # 3. Feature Engineering
        df = engineer_features(df)
        
        # 4. Preprocessing
        # We need to drop any columns that weren't used in training (like sensor_id if it was passed)
        # The preprocessor.transform will handle the column selection based on numeric_features and categorical_features
        try:
            X_proc = cls._preprocessor.transform(df)
        except Exception as e:
            return {"error": f"Preprocessing error: {str(e)}"}
            
        # 5. Prediction
        risk_idx = cls._model.predict(X_proc)[0]
        risk_probs = cls._model.predict_proba(X_proc)[0]
        
        return {
            "risk_level": cls._label_map[int(risk_idx)],
            "probabilities": {
                "LOW": round(float(risk_probs[0]), 4),
                "MEDIUM": round(float(risk_probs[1]), 4),
                "HIGH": round(float(risk_probs[2]), 4)
            }
        }

def predict(features: Dict[str, Any]) -> Dict[str, Any]:
    return RiskPredictor.predict(features)

if __name__ == "__main__":
    # Test prediction
    test_input = {
        'temperature': 25.0, 'humidity': 20.0, 'wind_speed': 5.0, 'rainfall': 0.0,
        'hour_of_day': 14, 'day_of_week': 2, 'season': 'Summer', 'region': 'South',
        'latitude': 45.9, 'longitude': 28.2, 'chainsaw_confidence': 0.1, 'fire_confidence': 0.8,
        'gunshot_confidence': 0.05, 'wildlife_activity_score': 0.4,
        'rolling_chainsaw_5m': 0.05, 'rolling_chainsaw_10m': 0.05, 'rolling_chainsaw_30m': 0.05,
        'rolling_fire_5m': 0.7, 'rolling_fire_10m': 0.6, 'rolling_fire_30m': 0.5,
        'detection_frequency_1h': 10, 'anomaly_score': 0.9
    }
    
    try:
        print("Testing prediction...")
        result = predict(test_input)
        print(result)
    except Exception as e:
        print(f"Error during test prediction: {e}")
