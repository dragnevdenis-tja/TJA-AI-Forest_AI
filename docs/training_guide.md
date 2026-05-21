# Forest Audio AI: Training Guide

This guide details the process of refreshing the risk prediction model with new synthetic or harvested data.

## 1. Data Generation
Generate a large-scale synthetic dataset for training and validation. The generator uses deterministic seeds and seasonally coherent weather patterns for Moldova.

```bash
# Generate 10,000 samples
$env:PYTHONPATH = "."; python backend/pipelines/data_generation/run_generation.py --n_samples 10000 --seed 42 --output datasets/structured/v1.csv
```

## 2. Model Training
The training script fits the preprocessor (scaling and encoding), executes feature engineering, and performs a stratified train/val/test split. It compares Logistic Regression, Random Forest, and XGBoost models.

```bash
# Run the multi-model training pipeline
$env:PYTHONPATH = "."; python backend/ml/structured_model/train.py
```
- **Output**: The best model is saved to `models/risk_model.joblib`.
- **Preprocessor**: Saved to `models/risk_preprocessor.joblib`.
- **Report**: A performance summary is saved to `experiments/risk_model_v1_report.json`.

## 3. Rigorous Evaluation
Execute the full evaluation suite to generate diagnostic plots and verify success criteria.

```bash
# Run all evaluation scripts
$env:PYTHONPATH = "."; python backend/evaluation/audio_eval.py
$env:PYTHONPATH = "."; python backend/evaluation/risk_eval.py
$env:PYTHONPATH = "."; python backend/evaluation/ablation.py
$env:PYTHONPATH = "."; python backend/evaluation/leakage_check.py
```

### Key Verification Tasks:
- **Ablation**: Check `experiments/ablation_results.json` to see feature contributions.
- **Leakage**: Verify the PASS status for data integrity.
- **Diagnostics**: Inspect `experiments/audio_roc_curves.png` and `experiments/risk_calibration.png`.

## 4. Maintenance
It is recommended to run the data generation and training pipeline quarterly to account for shifts in environmental patterns or updated audio classification labels.
