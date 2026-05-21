import pytest
import pandas as pd
import numpy as np
from backend.ml.structured_model.feature_engineering import engineer_features
from backend.ml.structured_model.preprocessor import get_preprocessor

def test_engineer_features():
    df = pd.DataFrame([{
        'hour_of_day': 12,
        'fire_confidence': 0.5,
        'chainsaw_confidence': 0.2,
        'gunshot_confidence': 0.1,
        'rolling_fire_30m': 0.4,
        'rolling_fire_5m': 0.1
    }])
    
    aug_df = engineer_features(df)
    
    # Cyclical checks
    assert -1 <= aug_df['time_sin'].iloc[0] <= 1
    assert -1 <= aug_df['time_cos'].iloc[0] <= 1
    
    # Composite score: 0.5*0.5 + 0.3*0.2 + 0.2*0.1 = 0.25 + 0.06 + 0.02 = 0.33
    assert aug_df['audio_composite_score'].iloc[0] == pytest.approx(0.33)
    
    # Rolling trend: 0.4 - 0.1 = 0.3
    assert aug_df['rolling_trend'].iloc[0] == pytest.approx(0.3)
    
    # is_night check (12 is day)
    assert aug_df['is_night'].iloc[0] == 0

def test_preprocessor_unseen_data():
    # Setup dummy data matching generator schema
    data = {
        'temperature': [20.0], 'humidity': [50.0], 'wind_speed': [5.0], 'rainfall': [0.0],
        'latitude': [47.0], 'longitude': [28.0], 'chainsaw_confidence': [0.1], 
        'fire_confidence': [0.1], 'gunshot_confidence': [0.1], 'wildlife_activity_score': [0.5],
        'rolling_chainsaw_5m': [0.1], 'rolling_chainsaw_10m': [0.1], 'rolling_chainsaw_30m': [0.1],
        'rolling_fire_5m': [0.1], 'rolling_fire_10m': [0.1], 'rolling_fire_30m': [0.1],
        'detection_frequency_1h': [1], 'anomaly_score': [0.5],
        'time_sin': [0.0], 'time_cos': [1.0], 'audio_composite_score': [0.1], 'rolling_trend': [0.0],
        'season': ['Summer'], 'region': ['Codrii'], 'is_night': [0], 'day_of_week': [1]
    }
    df = pd.DataFrame(data)
    
    preprocessor = get_preprocessor()
    preprocessor.fit(df)
    
    # Test on unseen category
    unseen_df = df.copy()
    unseen_df['season'] = 'UnknownSeason'
    
    try:
        X_proc = preprocessor.transform(unseen_df)
        assert X_proc.shape[0] == 1
    except Exception as e:
        pytest.fail(f"Preprocessor crashed on unseen category: {e}")
