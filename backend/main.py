import os
import time
import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from backend.utils.config import config, AUDIO_MODEL_PATH
from backend.services.audio_service.model_wrapper import ForestAudioAI
from backend.ml.structured_model.predict import RiskPredictor
from backend.api.routes import audio, risk, pipeline, websocket
from backend.api.middleware.timing import TimingMiddleware

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("forest_audio")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load models once
    logger.info("🚀 Starting Forest Audio AI Backend...")
    
    # 1. Load Audio Model
    try:
        model_path = os.path.join(os.getcwd(), AUDIO_MODEL_PATH)
        logger.info(f"Loading Audio Model from {model_path}...")
        app.state.audio_model = ForestAudioAI(model_path)
        app.state.audio_model_loaded = True
        logger.info("✅ Audio Model loaded successfully.")
    except Exception as e:
        logger.error(f"❌ Failed to load Audio Model: {e}")
        app.state.audio_model = None
        app.state.audio_model_loaded = False

    # 2. Load Risk Model
    try:
        logger.info("Loading Risk Model and Preprocessor...")
        # RiskPredictor handles its own lazy loading, but let's trigger it now
        # so it's ready for the first request
        from backend.ml.structured_model.predict import RiskPredictor
        RiskPredictor._load_assets()
        app.state.risk_model = RiskPredictor
        app.state.risk_model_loaded = True
        logger.info("✅ Risk Model loaded successfully.")
    except Exception as e:
        logger.error(f"❌ Failed to load Risk Model: {e}")
        app.state.risk_model = None
        app.state.risk_model_loaded = False

    app.state.start_time = time.time()
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Forest Audio AI Backend...")

app = FastAPI(title="Forest Audio AI", lifespan=lifespan)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(TimingMiddleware)

# Routes
app.include_router(audio.router, prefix="/api/audio", tags=["Audio"])
app.include_router(risk.router, prefix="/api/risk", tags=["Risk"])
app.include_router(pipeline.router, prefix="/api/pipeline", tags=["Pipeline"])
app.include_router(websocket.router, tags=["WebSocket"])

@app.get("/health")
async def health_check():
    uptime = time.time() - getattr(app.state, "start_time", time.time())
    return {
        "status": "healthy",
        "audio_model_loaded": getattr(app.state, "audio_model_loaded", False),
        "risk_model_loaded": getattr(app.state, "risk_model_loaded", False),
        "uptime_seconds": round(uptime, 2)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
