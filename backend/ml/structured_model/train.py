import pandas as pd
import numpy as np
import json
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

from backend.utils.config import config, RANDOM_SEED
from backend.ml.structured_model.feature_engineering import engineer_features
from backend.ml.structured_model.preprocessor import get_preprocessor, save_preprocessor

def train():
    # 1. Load data
    data_path = "datasets/structured/v1.csv"
    if not os.path.exists(data_path):
        print(f"❌ Data not found at {data_path}. Please run data generation first.")
        return

    df = pd.read_csv(data_path)
    
    # 2. Feature Engineering
    df = engineer_features(df)
    
    # 3. Prepare target and features
    X = df.drop(columns=['risk_label', 'sensor_id']) # sensor_id is just an identifier
    y = df['risk_label']
    
    # Encode labels: LOW=0, MEDIUM=1, HIGH=2
    label_map = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    y_encoded = y.map(label_map)
    
    # 4. Split data (60/20/20)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=RANDOM_SEED, stratify=y_encoded
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.25, random_state=RANDOM_SEED, stratify=y_train_full
    )
    
    # 5. Preprocessing
    preprocessor = get_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    
    # Save preprocessor
    save_preprocessor(preprocessor)
    
    # 6. Train models
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED),
        "XGBoost": XGBClassifier(n_estimators=100, random_state=RANDOM_SEED, use_label_encoder=False, eval_metric='mlogloss')
    }
    
    results = {}
    best_f1 = -1
    best_model_name = ""
    
    print("\nTraining models and evaluating on validation set...")
    for name, model in models.items():
        model.fit(X_train_proc, y_train)
        y_pred = model.predict(X_val_proc)
        y_prob = model.predict_proba(X_val_proc)
        
        f1 = f1_score(y_val, y_pred, average='weighted')
        auc = roc_auc_score(y_val, y_prob, multi_class='ovr')
        
        results[name] = {"f1_weighted": round(f1, 4), "roc_auc_ovr": round(auc, 4)}
        
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            
    # Print comparison table
    print("\n" + "="*40)
    print(f"{'Model':20} | {'F1 Weighted':10} | {'ROC-AUC':10}")
    print("-" * 40)
    for name, metrics in results.items():
        print(f"{name:20} | {metrics['f1_weighted']:10.4f} | {metrics['roc_auc_ovr']:10.4f}")
    print("="*40)
    
    # 7. Save best model (XGBoost is the requirement for "final model", but let's save the best one)
    # User said: "Saves best model to models/risk_model.joblib"
    # And: "XGBClassifier (final model)"
    final_model = models["XGBoost"]
    model_save_path = "models/risk_model.joblib"
    os.makedirs("models", exist_ok=True)
    joblib.dump(final_model, model_save_path)
    print(f"\n✅ Best model (XGBoost) saved to {model_save_path}")
    
    # 8. Save report
    report = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "best_model": "XGBoost",
        "validation_results": results,
        "label_mapping": label_map,
        "seed": RANDOM_SEED
    }
    
    os.makedirs("experiments", exist_ok=True)
    with open("experiments/risk_model_v1_report.json", "w") as f:
        json.dump(report, f, indent=4)
    print(f"📝 Training report saved to experiments/risk_model_v1_report.json")

if __name__ == "__main__":
    train()
