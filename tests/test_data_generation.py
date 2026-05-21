import pytest
import pandas as pd
import numpy as np
from backend.pipelines.data_generation.generate_dataset import DataGenerator
from backend.pipelines.data_generation.labeling import assign_risk_label

def test_data_generator_output_schema():
    gen = DataGenerator(seed=42)
    row = gen.generate_row()
    
    expected_cols = [
        'temperature', 'humidity', 'wind_speed', 'rainfall',
        'hour_of_day', 'day_of_week', 'season', 'sensor_id',
        'region', 'latitude', 'longitude', 'chainsaw_confidence',
        'fire_confidence', 'gunshot_confidence', 'wildlife_activity_score',
        'rolling_chainsaw_5m', 'rolling_chainsaw_10m', 'rolling_chainsaw_30m',
        'rolling_fire_5m', 'rolling_fire_10m', 'rolling_fire_30m',
        'detection_frequency_1h', 'anomaly_score'
    ]
    
    for col in expected_cols:
        assert col in row, f"Missing column: {col}"
    assert len(row) == len(expected_cols)

def test_data_generator_ranges():
    gen = DataGenerator(seed=42)
    for _ in range(100):
        row = gen.generate_row()
        assert -20 <= row['temperature'] <= 45
        assert 0 <= row['humidity'] <= 100
        assert 0 <= row['wind_speed'] <= 50
        assert 0 <= row['rainfall'] <= 100
        assert 0 <= row['chainsaw_confidence'] <= 1
        assert 0 <= row['fire_confidence'] <= 1
        assert 0 <= row['gunshot_confidence'] <= 1
        assert 0 <= row['anomaly_score'] <= 1

def test_data_generator_determinism():
    gen1 = DataGenerator(seed=42)
    gen2 = DataGenerator(seed=42)
    
    row1 = gen1.generate_row()
    row2 = gen2.generate_row()
    
    assert row1 == row2

def test_class_distribution():
    gen = DataGenerator(seed=42)
    labels = []
    n_samples = 500
    for _ in range(n_samples):
        row = gen.generate_row()
        labels.append(assign_risk_label(row))
    
    counts = pd.Series(labels).value_counts(normalize=True)
    
    # Target: ~60% LOW, ~25% MEDIUM, ~15% HIGH
    # ±15% allowance for smaller sample size in tests
    assert 0.45 <= counts.get('LOW', 0) <= 0.75
    assert 0.10 <= counts.get('MEDIUM', 0) <= 0.40
    assert 0.05 <= counts.get('HIGH', 0) <= 0.25
