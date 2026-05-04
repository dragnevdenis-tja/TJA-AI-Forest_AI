# Forest Audio AI: Automated Forest Monitoring System

Forest Audio AI is a comprehensive deep learning system designed for real-time environmental surveillance. It identifies environmental and anthropogenic sounds (e.g., wildfires, chainsaws, wildlife, rain) using a hybrid Convolutional Recurrent Neural Network (CRNN) architecture.

## 🚀 Quick Start

### 1. Prerequisites
- **Python**: 3.9+ (Tested on 3.13/3.14)
- **Node.js & npm**: For the 3D web interface
- **FFmpeg**: Required for audio processing (handled automatically via `static-ffmpeg`)

### 2. Backend Setup
Clone the repository and install the Python dependencies:
```bash
# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Or .venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Web Interface Setup
The project includes a modern 3D globe interface.
```bash
# Frontend setup
cd forest-audio-web/frontend
npm install
npm run dev
```

In a separate terminal, start the FastAPI backend:
```bash
# From the project root
python -m uvicorn forest-audio-web.app.main:app --reload --port 8000
```

---

## 🏗️ Architecture & Pipeline

### Core Model (CRNN)
The system uses a **Convolutional Recurrent Neural Network**:
- **CNN Layers**: Extract spatial features from Mel-spectrograms (audio "images").
- **RNN (GRU) Layers**: Capture temporal sequences and sound durations.
- **PCEN (Trainable)**: Per-Channel Energy Normalization for robust feature extraction in noisy environments.

### Data Pipeline
1.  **Automated Harvesting**: `trainer.py` includes built-in logic to download and label audio samples from YouTube using `yt-dlp`.
2.  **Dataset Integration**: Supports multiple sources including FSC22, ESC50, and custom wildfire datasets.
3.  **Adversarial Hardening**: The `validator.py` script trains a secondary model to detect potential misclassifications by the primary model, creating a self-improving loop.

---

## 🛠️ Key Components & Scripts

### `scripts/trainer.py`
The main training engine. 
- **Features**: Automatic manifest building, YouTube data integration, multi-GPU support, and extensive augmentation.
- **Usage**: `python scripts/trainer.py --data_root ./Data --epochs 50`

### `scripts/runnner.py`
High-performance inference and batch evaluation script.
- **Features**: PCEN implementation, frame-level prediction, and robust audio I/O via `soundfile`.

### `scripts/validator.py`
Implements the "Validator" architecture for model hardening. It focuses on identifying edge cases where the primary classifier might fail.

### `forest-audio-web/`
A production-ready web application:
- **FastAPI Backend**: Handles real-time audio processing and node management.
- **React Frontend**: A Three.js based 3D globe where users can place virtual sensors and monitor real-time microphone input.

---

## 📦 Deployment

The project is designed to be **deploy-ready**:
- **Cross-Platform**: Uses `soundfile` and `static-ffmpeg` to ensure audio compatibility across Windows and Linux.
- **Requirements**: Pinned versions in `requirements.txt` ensure consistent environments.
- **Checkpoints**: The best model weights are saved in `models/best_checkpoint.pth`.

---

## 📝 License
This project is intended for conservation and early warning systems. See specific dataset licenses (e.g., ESC50) for data usage restrictions.
