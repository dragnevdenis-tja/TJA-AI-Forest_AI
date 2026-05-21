import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

from backend.utils.config import RANDOM_SEED
from backend.ml.structured_model.feature_engineering import engineer_features
from backend.ml.structured_model.preprocessor import load_preprocessor

def evaluate():
    # 1. Load data
    data_path = "datasets/structured/v1.csv"
    if not os.path.exists(data_path):
        print(f"❌ Data not found at {data_path}.")
        return

    df = pd.read_csv(data_path)
    df = engineer_features(df)
    
    X = df.drop(columns=['risk_label', 'sensor_id'])
    y = df['risk_label']
    label_map = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    inv_label_map = {v: k for k, v in label_map.items()}
    y_encoded = y.map(label_map)
    
    # Replicate split from train.py
    _, X_test, _, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=RANDOM_SEED, stratify=y_encoded
    )
    
    # 2. Load model and preprocessor
    model_path = "models/risk_model.joblib"
    preprocessor_path = "models/risk_preprocessor.joblib"
    
    if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
        print("❌ Model or preprocessor not found. Please run training first.")
        return
        
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    
    # 3. Transform and Predict
    X_test_proc = preprocessor.transform(X_test)
    y_pred = model.predict(X_test_proc)
    y_prob = model.predict_proba(X_test_proc)
    
    # 4. Classification Report
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["LOW", "MEDIUM", "HIGH"]))
    
    # 5. ROC-AUC
    roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
    print(f"📈 ROC-AUC (One-vs-Rest): {roc_auc:.4f}")
    
    # 6. Confusion Matrix
    print("\n🧩 Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=["Actual LOW", "Actual MEDIUM", "Actual HIGH"], 
                         columns=["Pred LOW", "Pred MEDIUM", "Pred HIGH"])
    print(cm_df)
    
    # 7. Feature Importance Plot
    print("\n📊 Generating feature importance plot...")
    # Get feature names from preprocessor
    # ColumnTransformer .get_feature_names_out()
    feature_names = preprocessor.get_feature_names_out()
    importances = model.feature_importances_
    
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(15)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feat_imp.values, y=feat_imp.index)
    plt.title("Top 15 Feature Importances (XGBoost)")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    
    os.makedirs("experiments", exist_ok=True)
    plt.savefig("experiments/feature_importance.png")
    print(f"✅ Feature importance plot saved to experiments/feature_importance.png")
    
    # 8. PASS/FAIL line
    target_auc = 0.90
    status = "TARGET MET" if roc_auc >= target_auc else "BELOW TARGET (≥0.90 required)"
    print(f"\n📢 ROC-AUC = {roc_auc:.4f} — {status}")

if __name__ == "__main__":
    evaluate()
