import joblib
import os
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def get_preprocessor():
    """
    Returns an sklearn Pipeline with ColumnTransformer for scaling and encoding.
    """
    numeric_features = [
        'temperature', 'humidity', 'wind_speed', 'rainfall',
        'latitude', 'longitude', 'chainsaw_confidence', 'fire_confidence',
        'gunshot_confidence', 'wildlife_activity_score',
        'rolling_chainsaw_5m', 'rolling_chainsaw_10m', 'rolling_chainsaw_30m',
        'rolling_fire_5m', 'rolling_fire_10m', 'rolling_fire_30m',
        'detection_frequency_1h', 'anomaly_score',
        'time_sin', 'time_cos', 'audio_composite_score', 'rolling_trend'
    ]
    
    categorical_features = ['season', 'region', 'is_night', 'day_of_week']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )
    
    return preprocessor

def save_preprocessor(preprocessor, path='models/risk_preprocessor.joblib'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(preprocessor, path)
    print(f"✅ Preprocessor saved to {path}")

def load_preprocessor(path='models/risk_preprocessor.joblib'):
    if os.path.exists(path):
        return joblib.load(path)
    return None
