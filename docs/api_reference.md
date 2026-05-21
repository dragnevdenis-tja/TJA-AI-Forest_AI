# Forest Audio AI: API Reference

## 1. REST Endpoints

### `POST /api/pipeline/analyze`
The primary endpoint for integrated analysis.
- **Request Body**:
  ```json
  {
    "audio_base64": "...",
    "temperature": 25.0,
    "humidity": 35.0,
    "wind_speed": 5.0,
    "rainfall": 0.0,
    "sensor_id": "SN-COD-202",
    "region": "Codrii",
    "latitude": 47.1167,
    "longitude": 28.3333
  }
  ```
- **Response**:
  ```json
  {
    "audio_predictions": { "fire": 0.8, "chainsaw": 0.1, ... },
    "risk_assessment": { "risk_level": "HIGH", "probabilities": { "LOW": 0.1, ... } },
    "alert_triggered": true,
    "latency_ms": 145.2
  }
  ```

### `POST /api/audio/analyze`
Independent audio classification. Supports multi-part file upload or JSON with base64.
- **Request Body**: `{ "audio": "base64_string" }`
- **Response**: `{ "predictions": { "fire": 0.7, ... }, "anomaly_score": 0.7, "timestamp": "..." }`

### `POST /api/risk/assess`
Independent structured risk evaluation.
- **Request Body**: Full feature dictionary matching training schema.
- **Response**: `{ "risk_level": "MEDIUM", "probabilities": { ... }, "timestamp": "..." }`

### `GET /api/risk/history`
Retrieves last $N$ assessments for a sensor.
- **Query Params**: `sensor_id=X`, `limit=50`
- **Response**: List of assessment objects.

### `GET /health`
System status.
- **Response**: `{ "status": "healthy", "audio_model_loaded": true, "risk_model_loaded": true, "uptime_seconds": 3600.0 }`

## 2. WebSocket Interface

### `ws:///ws/live`
Real-time event stream.

#### Outgoing Events (Server to Client)

**`initial_status`**:
Sent immediately upon connection.
```json
{
  "event": "initial_status",
  "sensors": [ { "id": "...", "lat": 0.0, "lng": 0.0, "status": "active" }, ... ]
}
```

**`inference_result`**:
Broadcasted whenever a pipeline analysis completes.
```json
{
  "event": "inference_result",
  "sensor_id": "SN-COD-202",
  "region": "Codrii",
  "timestamp": "2026-05-21T14:30:00Z",
  "audio_predictions": { "fire": 0.9, ... },
  "risk_level": "HIGH",
  "probabilities": { "HIGH": 0.95, ... },
  "alert_triggered": true,
  "lat": 47.1167,
  "lon": 28.3333
}
```

**`alert`**:
Broadcasted only for `MEDIUM` or `HIGH` risk detections.
```json
{
  "event": "alert",
  "sensor_id": "SN-COD-202",
  "risk_level": "HIGH",
  "timestamp": "..."
}
```
