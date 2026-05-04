# Forest Audio AI Web Interface

This is a 3D web interface for the Forest Audio AI project. It allows you to place virtual sensors (nodes) on a globe and monitor environmental sounds in real-time.

## Prerequisites

- Node.js and npm
- Python 3.9+ with dependencies installed in the root `.venv`

## Setup

1. **Backend**:
   The backend is a FastAPI server that handles node persistence and audio processing using the CRNN model.
   
   To run:
   ```bash
   # From the project root
   .venv\Scripts\python -m uvicorn forest-audio-web.app.main:app --reload --port 8000
   ```

2. **Frontend**:
   The frontend is a React application using `react-globe.gl` and Three.js.
   
   To run:
   ```bash
   cd forest-audio-web/frontend
   npm run dev
   ```

## Features

- **3D World Map**: Interact with a high-resolution globe.
- **Node Management**: Add sensors anywhere in the world by clicking 'Add Node' and then clicking on the map.
- **Real-time Processing**: Click a node, then 'Start Monitoring' to record 5 seconds of audio from your microphone. The audio is sent to the AI and classified instantly.
- **Aesthetic UI**: Modern, dark-themed interface with smooth animations and progress bars for detection confidence.
