from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Dict
import json
import os
import asyncio
import io
import numpy as np
import soundfile as sf
from .model_wrapper import ForestAudioAI

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "best_checkpoint.pth")
model = ForestAudioAI(MODEL_PATH)

NODES_FILE = os.path.join(os.path.dirname(__file__), "nodes.json")
SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "audio_samples")

# Mount samples directory
if os.path.exists(SAMPLES_DIR):
    app.mount("/samples", StaticFiles(directory=SAMPLES_DIR), name="samples")

class Node(BaseModel):
    id: str
    lat: float
    lng: float
    name: str
    status: str = "active"

def load_nodes():
    if os.path.exists(NODES_FILE):
        with open(NODES_FILE, "r") as f:
            return json.load(f)
    return []

def save_nodes(nodes):
    with open(NODES_FILE, "w") as f:
        json.dump(nodes, f)

@app.get("/nodes")
async def get_nodes():
    return load_nodes()

@app.get("/available-samples")
async def get_samples():
    if not os.path.exists(SAMPLES_DIR):
        return []
    files = [f for f in os.listdir(SAMPLES_DIR) if f.endswith(('.mp3', '.wav', '.ogg'))]
    return files

@app.post("/nodes")
async def add_node(node: Node):
    nodes = load_nodes()
    nodes.append(node.dict())
    save_nodes(nodes)
    return node

@app.delete("/nodes/{node_id}")
async def delete_node(node_id: str):
    nodes = load_nodes()
    nodes = [n for n in nodes if n["id"] != node_id]
    save_nodes(nodes)
    return {"status": "deleted"}

@app.post("/process-audio/{node_id}")
async def process_audio(node_id: str, file: UploadFile = File(...)):
    audio_bytes = await file.read()
    try:
        predictions = model.predict(audio_bytes)
        return {
            "node_id": node_id,
            "predictions": predictions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Streaming state
class StreamSession:
    def __init__(self):
        self.buffer = np.array([], dtype=np.float32)
        self.sample_rate = 32000
        self.max_buffer_len = 32000 * 10 # 10 seconds max history

    def add_chunk(self, pcm_data: np.ndarray):
        self.buffer = np.concatenate([self.buffer, pcm_data])
        if len(self.buffer) > self.max_buffer_len:
            self.buffer = self.buffer[-self.max_buffer_len:]

    def get_last_n_seconds(self, seconds=5):
        num_samples = int(self.sample_rate * seconds)
        if len(self.buffer) < num_samples:
            # Pad with zeros if not enough data
            return np.pad(self.buffer, (num_samples - len(self.buffer), 0))
        return self.buffer[-num_samples:]

@app.websocket("/ws/stream/{node_id}")
async def websocket_endpoint(websocket: WebSocket, node_id: str):
    await websocket.accept()
    session = StreamSession()
    
    try:
        while True:
            # Receive audio chunk (expected as raw Float32 PCM)
            data = await websocket.receive_bytes()
            
            try:
                # Interpret the bytes as Float32 PCM
                audio = np.frombuffer(data, dtype=np.float32)
                
                if len(audio) > 0:
                    session.add_chunk(audio)

                    # Run prediction on the last 5 seconds
                    last_5s = session.get_last_n_seconds(5)
                    predictions = model.predict_pcm(last_5s, 32000)
                    
                    await websocket.send_json({
                        "node_id": node_id,
                        "predictions": predictions,
                        "timestamp": asyncio.get_event_loop().time()
                    })
                
            except Exception as e:
                print(f"Error processing stream chunk: {e}")
                await websocket.send_json({"error": str(e)})

    except WebSocketDisconnect:
        print(f"Client disconnected from node {node_id}")
