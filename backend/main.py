# backend/main.py

import os
import uuid
import logging
import platform
from typing import Dict, List
import asyncio

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks, Request, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from pipeline import TranscriptionSession, convert_pcm_to_mp3
from config_manager import config_data, save_config
from faster_whisper import WhisperModel


# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

sessions: Dict[str, TranscriptionSession] = {}
transcription_jobs: Dict[str, Dict] = {}
model_cache: Dict[str, WhisperModel] = {}


# --- Model Loading ---
def get_whisper_model(model_size: str, device: str) -> WhisperModel:
    """Loads a Whisper model from cache or creates a new one."""
    model_key = f"{model_size}_{device}"
    if model_key not in model_cache:
        logger.info(f"Loading Whisper model '{model_size}' on '{device}'...")
        compute_type = "int8" if device == "cpu" else "float16"
        model_cache[model_key] = WhisperModel(model_size, device=device, compute_type=compute_type)
        logger.info(f"Whisper model '{model_key}' loaded and cached.")
    return model_cache[model_key]


# --- API Routes ---
@app.get("/api/settings")
def get_settings():
    """Returns the current application settings."""
    return JSONResponse(content=config_data)


@app.post("/api/settings")
async def set_settings(request: Request):
    """Updates the application settings."""
    new_settings = await request.json()
    config_data.update(new_settings)
    save_config(config_data)
    return JSONResponse(content={"message": "Settings updated successfully"})


@app.post("/api/transcribe")
async def transcribe_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model: str = Form("base"),
    language: str = Form("en"),
    device: str = Form("cpu")
):
    job_id = str(uuid.uuid4())
    transcription_jobs[job_id] = {"status": "processing", "result": ""}

    file_path = f"temp_{job_id}_{file.filename}"
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    background_tasks.add_task(
        run_file_transcription,
        job_id,
        file_path,
        model,
        language,
        device
    )

    return JSONResponse(content={"job_id": job_id})


@app.get("/api/transcribe/status/{job_id}")
async def get_transcription_status(job_id: str):
    job = transcription_jobs.get(job_id)
    if not job:
        return JSONResponse(content={"error": "Job not found"}, status_code=404)
    return JSONResponse(content=job)


def run_file_transcription(job_id: str, file_path: str, model_size: str, language: str, device: str):
    """Background task to transcribe an audio file."""
    try:
        logger.info(f"[{job_id}] Starting file transcription for {file_path}")
        model = get_whisper_model(model_size, device)

        segments, _ = model.transcribe(file_path, language=language, beam_size=5)
        
        result_text = " ".join([seg.text for seg in segments]).strip()
        
        transcription_jobs[job_id] = {"status": "completed", "result": result_text}
        logger.info(f"[{job_id}] File transcription completed successfully.")

    except Exception as e:
        logger.error(f"[{job_id}] Error during file transcription: {e}", exc_info=True)
        transcription_jobs[job_id] = {"status": "error", "result": str(e)}
    finally:
        if os.path.exists(file_path):
            os.unlink(file_path)


@app.get("/api/config")
def get_config():
    """Returns the available models and compute devices."""
    logger.info("Serving configuration")

    devices: List[str] = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if platform.system() == "Darwin" and torch.backends.mps.is_available():
        devices.append("mps")

    models: List[str] = ["tiny", "base", "small", "medium", "large-v3", "distil-large-v3"]

    languages = {
        "en": "English", "es": "Spanish", "fr": "French", "de": "German", 
        "it": "Italian", "pt": "Portuguese", "ru": "Russian", "zh": "Chinese", 
        "ja": "Japanese", "ko": "Korean"
    }

    return JSONResponse(content={"devices": devices, "models": models, "languages": languages})


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    logger.info(f"WebSocket connection accepted for session_id: {session_id}")
    session = None
    try:
        client_config = await websocket.receive_json()
        logger.info(f"[{session_id}] Received configuration: {client_config}")

        # Get model from cache
        model_size = client_config.get("model", "tiny")
        device = client_config.get("device", "cpu")
        whisper_model = get_whisper_model(model_size, device)

        session = TranscriptionSession(session_id, websocket, client_config, whisper_model)
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
