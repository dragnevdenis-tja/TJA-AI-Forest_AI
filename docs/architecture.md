# Forest Audio AI: Architecture Detail

## 1. Dual-Service ML Pipeline
The core intelligence of the system resides in a cascaded machine learning pipeline. This architecture ensures that high-level risk assessments are grounded in low-level sensory evidence.

### Service 1: Audio Soundscape Classification (CRNN)
- **Model**: Convolutional Recurrent Neural Network (CRNN) with Attention.
- **Input**: 5-second audio chunks (PCM / WAV).
- **Processing**: Transforms audio into Mel-spectrograms, followed by PCEN (Per-Channel Energy Normalization) for robust background noise suppression.
- **Output**: Multi-label confidence scores for `fire`, `chainsaw`, `gunshot`, `rain`, `wildlife`, and `human`.
- **Role**: Provides the "ears" of the system, identifying specific acoustic events.

### Service 2: Structured Risk Assessment (XGBoost)
- **Model**: XGBoost Classifier.
- **Input**: Environmental metadata (temp, humidity, wind) + Audio confidence scores + Rolling historical averages.
- **Processing**: Includes feature engineering (cyclical time encoding, audio composite threat scores).
- **Output**: Categorical risk level (`LOW`, `MEDIUM`, `HIGH`) and class probabilities.
- **Role**: Acts as the "brain," contextualizing audio detections within environmental and temporal frameworks to assess overall threat levels.

## 2. WebSocket Event Flow
Real-time monitoring is powered by an asynchronous event-driven architecture.

1. **Initial Handshake**: On connection to `ws:///ws/live`, the server pushes the full `sensor_registry.json` to initialize the frontend map and state.
2. **Analysis Trigger**: When the `/api/pipeline/analyze` endpoint completes, it publishes an `inference_result` event to a thread-safe `EventBus`.
3. **Global Broadcast**: The `EventBus` iterates through all active WebSocket connections and streams the JSON payload.
4. **Alert Trigger**: If the risk level exceeds `LOW`, a secondary `alert` event is published, triggering high-priority UI updates (animations, sounds, or notifications) on all connected dashboards.

## 3. IoT Edge-to-Cloud Path
The system is designed for high availability in remote, internet-constrained environments.

- **Edge Node (ESP32-S3)**: Captures audio and performs local RMS energy filtering. If the environment is silent, transmission is skipped to conserve battery and bandwidth.
- **Mesh/Local Link**: Nodes transmit data via WiFi or LoRa to a central Raspberry Pi gateway.
- **Resilient Gateway (Raspberry Pi 4)**: Aggregates MQTT streams, maintains an asynchronous forwarding queue, and buffers failed requests to local storage for automatic retry with exponential backoff.
- **Cloud API**: Receives aggregated data and performs the computationally intensive ML inference.
