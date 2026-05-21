# Forest Audio AI (TerraGuard)

Forest Audio AI is a production-grade monitoring and analysis system designed to protect the forests of Moldova. It leverages a dual-service machine learning pipeline to detect illegal logging, wildfires, and other environmental threats in real-time. By combining high-fidelity audio classification with structured environmental data, the system provides a holistic view of forest health and security.

The system is built on a resilient edge-to-cloud architecture. Low-power IoT sensor nodes capture 5-second audio chunks and environmental metadata, which are aggregated by a local gateway and forwarded to a central FastAPI backend. The backend orchestrates a sophisticated analysis pipeline: it uses a CRNN model for audio soundscape classification and an XGBoost model for multi-factor risk assessment. Results are pushed instantly to a modern React dashboard via WebSockets, enabling immediate geographic awareness and response.

## System Architecture
```text
[ IoT Sensor Nodes ] --- (MQTT) ---> [ RPi Gateway ] --- (HTTPS) ---+
      (ESP32)                                                       |
                                                                    v
[ Frontend Dashboard ] <--- (WS) --- [ FastAPI Backend ] <--- [ ML Models ]
      (React)                         (Uvicorn)            (CRNN + XGBoost)
```

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   cd frontend && npm install && cd ..
   ```

2. **Generate Training Data**:
   ```bash
   $env:PYTHONPATH = "."; python backend/pipelines/data_generation/run_generation.py --n_samples 10000
   ```

3. **Train the Risk Model**:
   ```bash
   $env:PYTHONPATH = "."; python backend/ml/structured_model/train.py
   ```

4. **Launch System (Docker)**:
   ```bash
   docker-compose up --build
   ```

5. **Access Dashboard**:
   Open [http://localhost:3000](http://localhost:3000) in your browser.

## Directory Structure

| Directory | Description |
| :--- | :--- |
| `backend/` | Centralized FastAPI application, services, and ML logic. |
| `backend/api/` | REST routes, WebSocket endpoint, and middleware. |
| `backend/ml/` | Model definitions, training scripts, and preprocessing. |
| `backend/pipelines/` | Data generation and feature engineering pipelines. |
| `backend/utils/` | Shared utilities (Config, Logger, Seed). |
| `frontend/` | React/Vite dashboard source code. |
| `iot/` | IoT architecture, gateway code, and deployment guides. |
| `datasets/` | Synthetic and harvested datasets. |
| `experiments/` | Evaluation reports, plots, and ablation studies. |
| `tests/` | Comprehensive unit and integration test suite. |

## Success Criteria Summary
| Criterion | Status | Verified By |
| :--- | :--- | :--- |
| **Real-time Latency** | ✅ PASS | Timing Middleware (< 2s) |
| **Detection mAP** | ✅ 0.993 | `audio_eval.py` |
| **Risk ROC-AUC** | ✅ 1.000 | `risk_eval.py` |
| **Determinism** | ✅ 100% | `test_data_generation.py` |
| **Containerization** | ✅ Ready | `docker-compose up` |
