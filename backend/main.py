# backend/main.py

import os
import uuid
import logging
import platform
from typing import Dict, List

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from pipeline import TranscriptionSession, convert_pcm_to_mp3

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

# In-memory store for active sessions
sessions: Dict[str, TranscriptionSession] = {}


# --- API Routes ---
@app.get("/api/config")
def get_config():
    """Returns the available models and compute devices."""
    logger.info("Serving configuration")

    # Determine available compute devices
    devices: List[str] = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    # Check for Apple Silicon MPS
    if platform.system() == "Darwin" and torch.backends.mps.is_available():
        devices.append("mps")

    # Define available Whisper models
    models: List[str] = ["tiny", "base", "small", "medium", "turbo", "distil-large-v3", "large-v3"]

    return JSONResponse(content={"devices": devices, "models": models})


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    logger.info(f"WebSocket connection accepted for session_id: {session_id}")
    session = None # Ensure session is defined for the finally block
    try:
        config_data = await websocket.receive_json()    # First message is configuration
        logger.info(f"[{session_id}] Received configuration: {config_data}")

        session = TranscriptionSession(session_id, websocket, config_data)
        sessions[session_id] = session

        await session.run_pipeline()

    except WebSocketDisconnect:
        logger.info(f"[{session_id}] WebSocket disconnected.")
    except Exception as e:
        logger.error(f"[{session_id}] Error in WebSocket endpoint: {e}", exc_info=True)
    finally:
        if session:
            logger.info(f"[{session_id}] WebSocket closed, session is now available for download.")


@app.get("/download/{session_id}")
async def download_recording(session_id: str, background_tasks: BackgroundTasks):
    session = sessions.get(session_id)
    if not session or not os.path.exists(session.temp_file.name):
        return JSONResponse(content={"error": "Session not found or file is unavailable"}, status_code=404)

    pcm_path = session.temp_file.name
    
    try:
        mp3_path = convert_pcm_to_mp3(pcm_path)

        def cleanup_files():
            logger.info(f"[{session_id}] Cleaning up files: {pcm_path}, {mp3_path}")
            if os.path.exists(pcm_path):
                os.unlink(pcm_path)
            if os.path.exists(mp3_path):
                os.unlink(mp3_path)
            if session_id in sessions:
                del sessions[session_id]
                logger.info(f"[{session_id}] Session removed after download.")

        background_tasks.add_task(cleanup_files)
        
        return FileResponse(path=mp3_path, media_type='audio/mpeg', filename=f"recording_{session_id}.mp3")
    except Exception as e:
        logger.error(f"[{session_id}] Error during download conversion: {e}", exc_info=True)
        return JSONResponse(content={"error": "Failed to convert audio file"}, status_code=500)


app.mount("/", StaticFiles(directory="../frontend", html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
