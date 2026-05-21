import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds derived features before the preprocessor:
    - time_sin / time_cos (cyclical encoding of hour_of_day)
    - audio_composite_score = 0.5*fire_confidence + 0.3*chainsaw_confidence + 0.2*gunshot_confidence
    - is_night (hour < 6 or hour > 21)
    - rolling_trend = rolling_fire_30m - rolling_fire_5m
    """
    df = df.copy()
    
    # 1. Cyclical encoding of hour_of_day
    df['time_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
    df['time_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
    
    # 2. Audio composite score
    df['audio_composite_score'] = (
        0.5 * df['fire_confidence'] + 
        0.3 * df['chainsaw_confidence'] + 
        0.2 * df['gunshot_confidence']
    )
    
    # 3. is_night (hour < 6 or hour > 21)
    df['is_night'] = ((df['hour_of_day'] < 6) | (df['hour_of_day'] > 21)).astype(int)
    
    # 4. rolling_trend = rolling_fire_30m - rolling_fire_5m
    df['rolling_trend'] = df['rolling_fire_30m'] - df['rolling_fire_5m']
    
    return df
