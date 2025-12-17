"""FastAPI application for EchoScribe transcription service."""

import asyncio
import logging
import os
import platform
import uuid
from contextlib import asynccontextmanager
from typing import Dict, List

import torch
from cleanup import (
    CleanupManager,
    JobInfo,
    SessionInfo,
    cleanup_orphaned_temp_files,
    cleanup_temp_directory,
    get_cleanup_config,
    get_temp_dir,
)
from config_manager import config_data, get_config, reload_config, save_config
from models import SettingsUpdate
from auth import api_key_auth
from rate_limiter import api_rate_limiter, upload_rate_limiter
from utils import (
    ALLOWED_AUDIO_EXTENSIONS,
    MAX_FILE_SIZE,
    VAD_CACHE_DIR,
    get_model_status,
    get_whisper_model,
    get_whisper_model_sync,
    is_model_cached,
    is_valid_audio_extension,
    sanitize_filename,
)


from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    HTTPException,
    Form,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel
from pipeline import TranscriptionSession, convert_pcm_to_mp3
from starlette.requests import ClientDisconnect

import starlette.formparsers as formparsers
formparsers.MultiPartParser.max_file_size = 10 * 1024 * 1024 * 1024


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

sessions: Dict[str, SessionInfo] = {}
transcription_jobs: Dict[str, JobInfo] = {}

active_websockets: Dict[str, WebSocket] = {}

cleanup_manager: CleanupManager = None
cleanup_task: asyncio.Task = None
shutdown_event: asyncio.Event = None


async def cleanup_background_task() -> None:
    """Periodically clean up expired sessions and jobs."""
    cleanup_config = get_cleanup_config()
    interval = cleanup_config["cleanup_interval_seconds"]

    while True:
        try:
            await asyncio.sleep(interval)
            if cleanup_manager:
                removed_sessions, removed_jobs = cleanup_manager.cleanup(
                    sessions, transcription_jobs
                )
                if removed_sessions or removed_jobs:
                    logger.info(
                        f"Cleanup: removed {len(removed_sessions)} sessions, "
                        f"{len(removed_jobs)} jobs"
                    )
        except asyncio.CancelledError:
            logger.info("Cleanup task cancelled")
            break
        except Exception as e:
            logger.error(f"Error in cleanup task: {e}", exc_info=True)


@asynccontextmanager
async def lifespan(app):
    """Application lifespan manager for startup and shutdown."""
    global cleanup_manager, cleanup_task, shutdown_event

    # Startup
    shutdown_event = asyncio.Event()

    cleanup_config = get_cleanup_config()
    cleanup_manager = CleanupManager(
        session_ttl_seconds=cleanup_config["session_ttl_minutes"] * 60,
        job_retention_seconds=cleanup_config["job_retention_minutes"] * 60,
    )

    # Initialize temp directory and cleanup orphaned files from previous runs
    temp_dir = get_temp_dir()
    orphaned_count = cleanup_orphaned_temp_files(temp_dir, max_age_seconds=3600)
    if orphaned_count > 0:
        logger.info(f"Cleaned up {orphaned_count} orphaned temp files from previous runs")

    # Preload Silero VAD model at startup
    logger.info("Checking Silero VAD model availability...")
    try:
        original_hub_dir = torch.hub.get_dir()
        torch.hub.set_dir(VAD_CACHE_DIR)
        logger.info(f"Loading Silero VAD model from cache dir: {VAD_CACHE_DIR}")
        torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=True,
        )
        torch.hub.set_dir(original_hub_dir)
        logger.info("Silero VAD model is ready")
    except Exception as e:
        logger.error(f"Failed to preload Silero VAD model: {e}")
        raise

    # Preload Whisper models if enabled
    config = get_config()
    preload_config = config.get("preload_models", {})
    if preload_config.get("enabled", False):
        models_to_preload = preload_config.get("models", [])
        device = preload_config.get("device", "cpu")
        if models_to_preload:
            logger.info(f"Preloading {len(models_to_preload)} Whisper models on {device}...")
            for model_name in models_to_preload:
                try:
                    logger.info(f"Preloading model: {model_name}")
                    await get_whisper_model(model_name, device)
                    logger.info(f"Model {model_name} preloaded successfully")
                except Exception as e:
                    logger.error(f"Failed to preload model {model_name}: {e}")

    cleanup_task = asyncio.create_task(cleanup_background_task())
    logger.info("Cleanup background task started")

    yield

    # Shutdown - signal all connections
    logger.info("Starting graceful shutdown...")
    shutdown_event.set()

    # Close all active WebSocket connections
    if active_websockets:
        logger.info(f"Closing {len(active_websockets)} active WebSocket connections...")
        close_tasks = []
        for ws_id, ws in list(active_websockets.items()):
            try:
                close_tasks.append(ws.close(code=1001, reason="Server shutting down"))
            except Exception as e:
                logger.warning(f"Error closing WebSocket {ws_id}: {e}")

        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)
        active_websockets.clear()

    # Cancel cleanup task
    if cleanup_task:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass

    # Clean up all sessions and their temp files
    for session_id, session_info in list(sessions.items()):
        try:
            session = session_info.session
            if hasattr(session, 'temp_file') and session.temp_file:
                temp_path = session.temp_file.name
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
        except Exception as e:
            logger.warning(f"Error cleaning session {session_id}: {e}")
    sessions.clear()
    transcription_jobs.clear()

    # Clean up remaining temp files on shutdown
    cleanup_temp_directory()
    logger.info("Graceful shutdown complete")


app = FastAPI(lifespan=lifespan)


@app.get("/api/settings")
def get_settings() -> JSONResponse:
    """Get current application settings.

    Returns:
        JSON response with current configuration.
    """
    return JSONResponse(content=get_config())


@app.post("/api/settings")
async def set_settings(
    settings: SettingsUpdate,
    _: str = Depends(api_key_auth)
) -> JSONResponse:
    """Update application settings with validation.

    Args:
        settings: Validated settings update from Pydantic model.

    Returns:
        JSON response confirming update.

    Raises:
        HTTPException: 401 if authentication fails.
    """
    current_config = get_config()

    update_dict = settings.model_dump(exclude_unset=True)
    for key, value in update_dict.items():
        if value is not None:
            if (key in current_config and
                isinstance(current_config[key], dict) and
                isinstance(value, dict)):
                current_config[key].update(value)
            else:
                current_config[key] = value

    save_config(current_config)
    return JSONResponse(content={"message": "Settings updated successfully"})


@app.post("/api/transcribe")
async def transcribe_file(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model: str = Form("base"),
    language: str = Form("en"),
    device: str = Form("cpu"),
    _: str = Depends(api_key_auth),
) -> JSONResponse:
    """Transcribe uploaded audio file.

    Args:
        request: Incoming FastAPI request for rate limiting.
        background_tasks: FastAPI background tasks manager.
        file: Audio file to transcribe.
        model: Whisper model size.
        language: Language code for transcription.
        device: Compute device (cpu, cuda, mps).

    Returns:
        JSON response with job_id for tracking.

    Raises:
        HTTPException: If file validation or rate limit fails.
    """
    upload_rate_limiter.check_rate_limit(request)

    filename = file.filename or "unnamed"
    ext = os.path.splitext(filename.lower())[1]
    if not is_valid_audio_extension(ext):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid file type. Allowed formats: "
                f"{', '.join(sorted(ALLOWED_AUDIO_EXTENSIONS))}"
            )
        )

    job_id = str(uuid.uuid4())
    transcription_jobs[job_id] = JobInfo(job_id=job_id, status="processing")
    safe_filename = sanitize_filename(filename)
    temp_dir = get_temp_dir()
    file_path = os.path.join(temp_dir, f"temp_{job_id}_{safe_filename}")

    file_size = 0
    chunk_size = 1024 * 1024
    try:
        with open(file_path, "wb") as buffer:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                buffer.write(chunk)
                file_size += len(chunk)
    except ClientDisconnect:
        if os.path.exists(file_path):
            os.unlink(file_path)
        del transcription_jobs[job_id]
        raise HTTPException(
            status_code=400,
            detail="Client disconnected during upload"
        )

    if MAX_FILE_SIZE > 0 and file_size > MAX_FILE_SIZE:
        os.unlink(file_path)
        del transcription_jobs[job_id]
        raise HTTPException(
            status_code=413,
            detail=(
                f"File too large. Maximum size: "
                f"{MAX_FILE_SIZE // (1024 * 1024)} MB"
            )
        )

    logger.info(
        f"[{job_id}] Received file: {filename} "
        f"({file_size / (1024*1024):.2f} MB)"
    )

    background_tasks.add_task(
        run_file_transcription, job_id, file_path, model, language, device
    )
    return JSONResponse(content={"job_id": job_id})


@app.get("/api/transcribe/status/{job_id}")
async def get_transcription_status(job_id: str) -> JSONResponse:
    """Get transcription job status.

    Args:
        job_id: Job identifier.

    Returns:
        JSON response with job status and result.
    """
    job = transcription_jobs.get(job_id)
    if not job:
        return JSONResponse(content={"error": "Job not found"}, status_code=404)
    return JSONResponse(content=job.to_dict())


@app.delete("/api/transcribe/{job_id}")
async def cancel_transcription(
    job_id: str,
    _: str = Depends(api_key_auth)
) -> JSONResponse:
    """Cancel transcription job.

    Args:
        job_id: Job identifier to cancel.

    Returns:
        JSON response confirming cancellation.

    Raises:
        HTTPException: 401 if authentication fails.
    """
    job = transcription_jobs.get(job_id)
    if not job:
        return JSONResponse(content={"error": "Job not found"}, status_code=404)

    if job.status != "processing":
        return JSONResponse(
            content={"error": "Job is not in processing state"},
            status_code=400
        )

    job.cancel()
    logger.info(f"[{job_id}] Cancellation requested")
    return JSONResponse(
        content={"message": "Cancellation requested", "job_id": job_id}
    )


def run_file_transcription(
    job_id: str,
    file_path: str,
    model_size: str,
    language: str,
    device: str
) -> None:
    """Background task to transcribe audio file.

    Args:
        job_id: Job identifier.
        file_path: Path to audio file.
        model_size: Whisper model size.
        language: Language code.
        device: Compute device.
    """
    try:
        job = transcription_jobs.get(job_id)

        if job and job.cancelled:
            logger.info(f"[{job_id}] Job was cancelled before starting")
            job.mark_completed("cancelled", "Transcription was cancelled")
            return

        logger.info(f"[{job_id}] Starting file transcription for {file_path}")
        model = get_whisper_model_sync(model_size, device)

        if job and job.cancelled:
            logger.info(f"[{job_id}] Job was cancelled before transcription")
            job.mark_completed("cancelled", "Transcription was cancelled")
            return

        segments, _ = model.transcribe(file_path, language=language, beam_size=5)

        result_parts = []
        for seg in segments:
            if job and job.cancelled:
                logger.info(f"[{job_id}] Job was cancelled during transcription")
                job.mark_completed("cancelled", "Transcription was cancelled")
                return
            result_parts.append(seg.text)

        result_text = " ".join(result_parts).strip()
        if job:
            job.mark_completed("completed", result_text)
        logger.info(f"[{job_id}] File transcription completed successfully.")
    except Exception as e:
        logger.error(
            f"[{job_id}] Error during file transcription: {e}",
            exc_info=True
        )
        job = transcription_jobs.get(job_id)
        if job and not job.cancelled:
            job.mark_completed("error", str(e))
    finally:
        if os.path.exists(file_path):
            os.unlink(file_path)


@app.get("/api/config")
def get_available_config() -> JSONResponse:
    """Get available models and compute devices.

    Returns:
        JSON response with devices, models, and languages.
    """
    logger.info("Serving configuration")
    devices: List[str] = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if platform.system() == "Darwin" and torch.backends.mps.is_available():
        devices.append("mps")
    models: List[str] = [
        "tiny",
        "base",
        "small",
        "medium",
        "distil-large-v3",
        "large-v3",
    ]
    languages = {
        "en": "English",
        "es": "Spanish",
        "fr": "French",
        "de": "German",
        "it": "Italian",
        "pt": "Portuguese",
        "ru": "Russian",
        "zh": "Chinese",
        "ja": "Japanese",
        "ko": "Korean",
    }
    return JSONResponse(
        content={"devices": devices, "models": models, "languages": languages}
    )


@app.get("/api/model/status")
def check_model_status(model: str, device: str = "cpu") -> JSONResponse:
    """Check if a model is cached and ready to use.

    Args:
        model: Whisper model name (e.g., 'tiny', 'base', 'large-v3').
        device: Compute device (cpu, cuda, mps).

    Returns:
        JSON response with model cache status.
    """
    status = get_model_status(model)
    status["device"] = device
    status["model"] = model
    return JSONResponse(content=status)


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str) -> None:
    """WebSocket endpoint for real-time transcription.

    Args:
        websocket: WebSocket connection.
        session_id: Session identifier.
    """
    await websocket.accept()
    logger.info(f"WebSocket connection accepted for session_id: {session_id}")

    active_websockets[session_id] = websocket

    session = None
    session_info = None

    # Create async-safe progress callback for WebSocket status updates
    async def send_status(status: str, message: str, progress: float):
        """Send status update over WebSocket."""
        try:
            await websocket.send_json({
                "type": "status",
                "status": status,
                "message": message,
                "progress": progress
            })
        except Exception as e:
            logger.warning(f"[{session_id}] Failed to send status: {e}")

    # Sync wrapper for progress callback (called from sync download code)
    def progress_callback(status: str, message: str, progress: float):
        """Sync wrapper to schedule async status send."""
        try:
            # Use asyncio.run_coroutine_threadsafe for thread-safe async call
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    send_status(status, message, progress),
                    loop
                )
        except Exception as e:
            logger.warning(f"[{session_id}] Progress callback error: {e}")

    try:
        client_config = await websocket.receive_json()
        logger.info(f"[{session_id}] Received configuration: {client_config}")
        model_size = client_config.get("model", "tiny")
        device = client_config.get("device", "cpu")

        # Check if model is cached and send appropriate initial status
        if is_model_cached(model_size):
            await send_status("loading", f"Loading {model_size} on {device}...", 0.3)
        else:
            await send_status("downloading", f"Model not found. Downloading {model_size}...", 0)

        # Load model with progress updates
        whisper_model = await get_whisper_model(model_size, device, progress_callback)

        # Send ready status before starting pipeline
        await send_status("ready", "Connected. Start speaking.", 1.0)

        session = TranscriptionSession(
            session_id, websocket, client_config, whisper_model
        )
        session_info = SessionInfo(session_id=session_id, session=session)
        sessions[session_id] = session_info
        await session.run_pipeline()
    except WebSocketDisconnect:
        logger.info(f"[{session_id}] WebSocket disconnected.")
    except Exception as e:
        logger.error(
            f"[{session_id}] Error in WebSocket endpoint: {e}",
            exc_info=True
        )
        # Try to send error status
        try:
            await websocket.send_json({
                "type": "status",
                "status": "error",
                "message": str(e),
                "progress": 0
            })
        except Exception:
            pass
    finally:
        active_websockets.pop(session_id, None)

        if session:
            if session_info:
                session_info.update_activity()
            logger.info(
                f"[{session_id}] WebSocket closed, "
                f"session is now available for download."
            )


@app.get("/download/{session_id}")
async def download_recording(
    session_id: str,
    background_tasks: BackgroundTasks
) -> FileResponse:
    """Download recording as MP3.

    Args:
        session_id: Session identifier.
        background_tasks: FastAPI background tasks manager.

    Returns:
        MP3 file response.
    """
    session_info = sessions.get(session_id)
    if not session_info:
        return JSONResponse(
            content={"error": "Session not found or file is unavailable"},
            status_code=404,
        )
    session = session_info.session
    if (not hasattr(session, 'temp_file') or
        not os.path.exists(session.temp_file.name)):
        return JSONResponse(
            content={"error": "Session not found or file is unavailable"},
            status_code=404,
        )
    pcm_path = session.temp_file.name
    try:
        mp3_path = convert_pcm_to_mp3(pcm_path)

        def cleanup_files() -> None:
            logger.info(
                f"[{session_id}] Cleaning up files: {pcm_path}, {mp3_path}"
            )
            if os.path.exists(pcm_path):
                os.unlink(pcm_path)
            if os.path.exists(mp3_path):
                os.unlink(mp3_path)
            if session_id in sessions:
                del sessions[session_id]
                logger.info(f"[{session_id}] Session removed after download.")

        background_tasks.add_task(cleanup_files)
        return FileResponse(
            path=mp3_path,
            media_type="audio/mpeg",
            filename=f"recording_{session_id}.mp3",
        )
    except Exception as e:
        logger.error(
            f"[{session_id}] Error during download conversion: {e}",
            exc_info=True
        )
        return JSONResponse(
            content={"error": "Failed to convert audio file"},
            status_code=500
        )


# Mount frontend static files using path relative to this script's location
# This ensures it works both locally and in Docker
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend")
if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
else:
    logger.warning(f"Frontend directory not found at {FRONTEND_DIR}, static files not mounted")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
