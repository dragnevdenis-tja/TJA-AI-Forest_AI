import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_recall_fscore_support, 
    average_precision_score, 
    confusion_matrix, 
    roc_curve, 
    auc,
    precision_recall_curve
)
import torch
import time

# Mock labels if real model not loaded
DEFAULT_LABELS = ['fire', 'chainsaw', 'gunshot', 'rain', 'wildlife', 'human']

def evaluate_audio(model=None, test_data=None):
    """
    Evaluates the CRNN audio classifier.
    If model/test_data are missing, it uses simulated predictions.
    """
    print("--- Audio Model Evaluation ---")
    
    # 1. Setup Labels and Data
    labels = model.labels if model else DEFAULT_LABELS
    n_classes = len(labels)
    
    is_simulated = False
    if test_data is None:
        print("⚠️ Real test data unavailable. Using simulated predictions for evaluation.")
        is_simulated = True
        n_samples = 500
        # Simulate ground truth
        y_true = np.random.randint(0, 2, (n_samples, n_classes))
        # Simulate predictions with some accuracy
        y_prob = np.clip(y_true * 0.7 + np.random.normal(0, 0.2, (n_samples, n_classes)), 0, 1)
    else:
        # Placeholder for real data evaluation logic
        y_true, y_prob = test_data

    # 2. Per-class metrics
    y_pred = (y_prob > 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
    aps = [average_precision_score(y_true[:, i], y_prob[:, i]) for i in range(n_classes)]
    mAP = np.mean(aps)

    metrics_df = pd.DataFrame({
        'Class': labels,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'AP': aps
    })
    print("\n📋 Per-Class Metrics:")
    print(metrics_df.to_string(index=False))
    print(f"\n📈 mAP: {mAP:.4f}")

    # 3. Threshold Sweep for Fire class
    fire_idx = labels.index('fire') if 'fire' in labels else 0
    p, r, thresholds = precision_recall_curve(y_true[:, fire_idx], y_prob[:, fire_idx])
    f1_scores = 2 * (p * r) / (p + r + 1e-9)
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    max_fire_f1 = f1_scores[best_idx]
    
    print(f"🔥 Fire Class Threshold Optimization:")
    print(f"   Max F1: {max_fire_f1:.4f} at Threshold: {best_thresh:.4f}")

    # 4. Plots
    os.makedirs("experiments", exist_ok=True)
    
    # ROC Curves
    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_prob[:, i])
        plt.plot(fpr, tpr, label=f'{labels[i]} (AUC = {auc(fpr, tpr):.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Audio Model: ROC Curves per Class')
    plt.legend()
    plt.savefig("experiments/audio_roc_curves.png")
    plt.close()

    # Confusion Matrix (aggregated)
    # Since multi-label, we'll take the max class for a standard CM visual
    y_true_single = np.argmax(y_true, axis=1)
    y_pred_single = np.argmax(y_prob, axis=1)
    cm = confusion_matrix(y_true_single, y_pred_single)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
    plt.title('Audio Model: Confusion Matrix (Aggregated)')
    plt.savefig("experiments/audio_confusion_matrix.png")
    plt.close()

    # 5. Success Criteria
    map_pass = mAP >= 0.85
    fire_recall_pass = recall[fire_idx] >= 0.90
    
    print("\n✅ Success Criteria:")
    print(f"   mAP >= 0.85:        {'PASS' if map_pass else 'FAIL'} ({mAP:.4f})")
    print(f"   Fire Recall >= 0.90: {'PASS' if fire_recall_pass else 'FAIL'} ({recall[fire_idx]:.4f})")

    return {
        "is_simulated": is_simulated,
        "mAP": float(mAP),
        "fire_recall": float(recall[fire_idx]),
        "map_pass": map_pass,
        "fire_recall_pass": fire_recall_pass
    }

if __name__ == "__main__":
    evaluate_audio()
