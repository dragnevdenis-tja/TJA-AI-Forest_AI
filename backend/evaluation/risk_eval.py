import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    roc_auc_score, 
    confusion_matrix, 
    classification_report
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from backend.utils.config import RANDOM_SEED
from backend.ml.structured_model.feature_engineering import engineer_features

def evaluate_risk():
    print("--- Risk Model Evaluation ---")
    
    # 1. Load Data
    data_path = "datasets/structured/v1.csv"
    if not os.path.exists(data_path):
        print(f"❌ Data not found at {data_path}.")
        return

    df = pd.read_csv(data_path)
    df = engineer_features(df)
    
    X = df.drop(columns=['risk_label', 'sensor_id'])
    y = df['risk_label']
    label_map = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    y_encoded = y.map(label_map)
    
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=RANDOM_SEED, stratify=y_encoded
    )
    
    # 2. Load Model and Preprocessor
    model_path = "models/risk_model.joblib"
    preprocessor_path = "models/risk_preprocessor.joblib"
    
    if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
        print("❌ Model or preprocessor not found. Please run training first.")
        return
        
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    
    # 3. Baseline Comparison
    print("\n🚀 Baseline Comparison:")
    X_test_proc = preprocessor.transform(X_test)
    X_train_proc = preprocessor.transform(X_train_full)
    
    # Baseline 1: Always predict LOW
    y_pred_low = np.zeros_like(y_test)
    f1_low = f1_score(y_test, y_pred_low, average='weighted')
    
    # Baseline 2: Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    lr.fit(X_train_proc, y_train_full)
    y_pred_lr = lr.predict(X_test_proc)
    f1_lr = f1_score(y_test, y_pred_lr, average='weighted')
    
    # Final Model
    y_pred_final = model.predict(X_test_proc)
    y_prob_final = model.predict_proba(X_test_proc)
    f1_final = f1_score(y_test, y_pred_final, average='weighted')
    
    baselines = pd.DataFrame({
        'Model': ['Always-LOW', 'Logistic Regression', 'Final Model (XGBoost)'],
        'F1 Weighted': [f1_low, f1_lr, f1_final]
    })
    print(baselines.to_string(index=False))

    # 4. Detailed Metrics
    acc = accuracy_score(y_test, y_pred_final)
    f1_weighted = f1_score(y_test, y_pred_final, average='weighted')
    f1_macro = f1_score(y_test, y_pred_final, average='macro')
    roc_auc = roc_auc_score(y_test, y_prob_final, multi_class='ovr', average='macro')
    
    print("\n📊 Model Performance:")
    print(f"   Accuracy:    {acc:.4f}")
    print(f"   Weighted F1: {f1_weighted:.4f}")
    print(f"   Macro F1:    {f1_macro:.4f}")
    print(f"   ROC-AUC:     {roc_auc:.4f}")

    # 5. Plots
    os.makedirs("experiments", exist_ok=True)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_final)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=label_map.keys(), yticklabels=label_map.keys(), cmap='Greens')
    plt.title('Risk Model: Confusion Matrix')
    plt.savefig("experiments/risk_confusion_matrix.png")
    plt.close()
    
    # Calibration Plot (Binary check for 'HIGH' risk)
    # y_test == 2 (HIGH)
    prob_high = y_prob_final[:, 2]
    prob_true, prob_pred = calibration_curve(y_test == 2, prob_high, n_bins=10)
    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, marker='o', label='XGBoost (HIGH class)')
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
    plt.xlabel('Predicted Probability')
    plt.ylabel('True Probability')
    plt.title('Reliability Diagram: HIGH Risk Class')
    plt.legend()
    plt.savefig("experiments/risk_calibration.png")
    plt.close()

    # 6. Success Criterion
    auc_pass = roc_auc >= 0.90
    print(f"\n📢 ROC-AUC = {roc_auc:.4f} — {'TARGET MET' if auc_pass else 'BELOW TARGET (>=0.90 required)'}")

if __name__ == "__main__":
    evaluate_risk()
