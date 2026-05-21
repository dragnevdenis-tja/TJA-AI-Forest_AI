import os
import joblib
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from backend.utils.config import RANDOM_SEED
from backend.ml.structured_model.feature_engineering import engineer_features

def run_ablation():
    print("--- Risk Model Ablation Study ---")
    
    # 1. Load Data
    data_path = "datasets/structured/v1.csv"
    if not os.path.exists(data_path):
        print(f"❌ Data not found at {data_path}.")
        return

    df = pd.read_csv(data_path)
    df = engineer_features(df)
    
    X = df.drop(columns=['risk_label', 'sensor_id'])
    y = df['risk_label'].map({"LOW": 0, "MEDIUM": 1, "HIGH": 2})
    
    # Feature Groups
    feature_groups = {
        "Audio Features": ['chainsaw_confidence', 'fire_confidence', 'gunshot_confidence', 'audio_composite_score'],
        "Rolling Averages": ['rolling_chainsaw_5m', 'rolling_chainsaw_10m', 'rolling_chainsaw_30m', 
                             'rolling_fire_5m', 'rolling_fire_10m', 'rolling_fire_30m', 'rolling_trend'],
        "Weather Features": ['temperature', 'humidity', 'wind_speed', 'rainfall'],
        "Temporal Features": ['hour_of_day', 'day_of_week', 'season', 'time_sin', 'time_cos', 'is_night']
    }

    results = {}
    
    def train_and_eval(X_subset, y):
        # Local preprocessor for ablation (since feature names change)
        numeric_cols = [c for c in X_subset.select_dtypes(include=[np.number]).columns]
        categorical_cols = [c for c in X_subset.select_dtypes(exclude=[np.number]).columns]
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
            ]
        )
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_subset, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
        )
        
        X_train_proc = preprocessor.fit_transform(X_train)
        X_test_proc = preprocessor.transform(X_test)
        
        model = XGBClassifier(n_estimators=50, random_state=RANDOM_SEED, eval_metric='mlogloss')
        model.fit(X_train_proc, y_train)
        
        y_prob = model.predict_proba(X_test_proc)
        auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')
        return auc

    # Baseline (All features)
    print("🚀 Training baseline model...")
    baseline_auc = train_and_eval(X, y)
    results["Full Model"] = round(baseline_auc, 4)
    print(f"   Baseline ROC-AUC: {baseline_auc:.4f}")

    # Ablation
    for group_name, features in feature_groups.items():
        print(f"🔥 Removing {group_name}...")
        X_reduced = X.drop(columns=[f for f in features if f in X.columns])
        reduced_auc = train_and_eval(X_reduced, y)
        drop = baseline_auc - reduced_auc
        results[f"No {group_name}"] = {
            "roc_auc": round(reduced_auc, 4),
            "drop": round(drop, 4)
        }
        print(f"   ROC-AUC: {reduced_auc:.4f} (Drop: {drop:.4f})")

    # Save results
    os.makedirs("experiments", exist_ok=True)
    with open("experiments/ablation_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print(f"\n✅ Ablation results saved to experiments/ablation_results.json")
    
    # Identify most important group
    most_important = max(feature_groups.keys(), key=lambda g: results[f"No {g}"]["drop"])
    print(f"🏆 Most contributing feature group: {most_important}")

if __name__ == "__main__":
    run_ablation()
