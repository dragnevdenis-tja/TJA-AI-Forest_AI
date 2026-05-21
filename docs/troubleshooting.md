# Forest Audio AI: Troubleshooting Manual

This guide covers common failure modes and provides diagnostic steps and fixes for the system.

## 1. Backend & ML Models

| Symptom | Diagnosis | Fix |
| :--- | :--- | :--- |
| `KeyError: 'audio_model'` in tests or API. | Lifespan events not triggered (models didn't load). | Ensure `TestClient` or the server is used in a `with` block or via `uvicorn` which triggers startup. |
| API returns `503 Service Unavailable`. | One or more models failed to load at startup. | Check `backend/ml/audio_model/best_checkpoint.pth` exists. Run `make train` to ensure risk model exists. |
| Predictions seem random or "flat". | Determinism issue or model trained on insufficient data. | Verify `backend/utils/seed.py` is being used. Re-generate 10,000+ samples and retrain. |

## 2. WebSockets & Live Data

| Symptom | Diagnosis | Fix |
| :--- | :--- | :--- |
| Dashboard shows "Reconnecting..." indefinitely. | WebSocket server not running or wrong URL. | Verify `VITE_WS_URL` in `.env` matches the backend host. Check backend logs for crash. |
| No `inference_result` events received. | `/api/pipeline/analyze` is not being called or event bus is failing. | Check gateway logs to ensure it's successfully posting to the API. |
| WebSocket disconnects frequently. | Network instability or proxy timeout. | Increase Nginx/Gateway timeout settings. Check for circular imports in `event_bus.py`. |

## 3. Frontend Dashboard

| Symptom | Diagnosis | Fix |
| :--- | :--- | :--- |
| Map is blank or showing only grey tiles. | No internet (cannot load tiles) or Leaflet CSS missing. | Ensure dashboard has access to `cartocdn.com`. Verify `leaflet.css` is imported in `SensorMap.tsx`. |
| Markers not appearing on the map. | `sensor_registry.json` not loaded or coordinate mismatch. | Check the `initial_status` event in browser dev tools (Network tab > WS). |
| Charts not updating in real-time. | `useWebSocket` hook not updating state correctly. | Verify `sensor_id` in incoming events matches the ID in the registry exactly. |

## 4. Docker & Networking

| Symptom | Diagnosis | Fix |
| :--- | :--- | :--- |
| `frontend` cannot reach `api`. | Docker internal networking issue. | In `docker-compose.yml`, ensure frontend uses `http://api:8000` or that ports are correctly exposed to localhost. |
| Container build fails on `npm install`. | Node.js version mismatch or network error. | Check `frontend/Dockerfile` uses `node:18-alpine`. Clear `node_modules` and try again. |
| Volume mounts not showing updated data. | Windows/Docker Desktop file sharing issue. | Restart Docker Desktop. Ensure paths in `docker-compose.yml` are relative to the project root. |
