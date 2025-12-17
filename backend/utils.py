"""Utility functions for file validation, sanitization, and model caching."""

import asyncio
import logging
import os
import re
from pathlib import Path
from typing import Callable, Dict, Optional, Set

from faster_whisper import WhisperModel
from huggingface_hub import hf_hub_download, HfFileSystem
from tqdm import tqdm


logger = logging.getLogger(__name__)

# Type alias for progress callback
ProgressCallback = Callable[[str, str, float], None]  # (status, message, progress)

# Model cache directory configuration
MODELS_CACHE_DIR = os.getenv("MODELS_CACHE_DIR", os.path.join(os.getcwd(), "models_cache"))
WHISPER_CACHE_DIR = os.path.join(MODELS_CACHE_DIR, "whisper_models")
VAD_CACHE_DIR = os.path.join(MODELS_CACHE_DIR, "silero_vad")

# Ensure cache directories exist
Path(WHISPER_CACHE_DIR).mkdir(parents=True, exist_ok=True)
Path(VAD_CACHE_DIR).mkdir(parents=True, exist_ok=True)

# Mapping of model names to HuggingFace repo IDs
MODEL_REPO_MAP: Dict[str, str] = {
    "tiny": "Systran/faster-whisper-tiny",
    "tiny.en": "Systran/faster-whisper-tiny.en",
    "base": "Systran/faster-whisper-base",
    "base.en": "Systran/faster-whisper-base.en",
    "small": "Systran/faster-whisper-small",
    "small.en": "Systran/faster-whisper-small.en",
    "medium": "Systran/faster-whisper-medium",
    "medium.en": "Systran/faster-whisper-medium.en",
    "large-v1": "Systran/faster-whisper-large-v1",
    "large-v2": "Systran/faster-whisper-large-v2",
    "large-v3": "Systran/faster-whisper-large-v3",
    "large-v3-turbo": "Systran/faster-whisper-large-v3-turbo",
    "distil-large-v2": "Systran/faster-distil-whisper-large-v2",
    "distil-large-v3": "Systran/faster-distil-whisper-large-v3",
    "distil-medium.en": "Systran/faster-distil-whisper-medium.en",
    "distil-small.en": "Systran/faster-distil-whisper-small.en",
}

ALLOWED_AUDIO_EXTENSIONS: Set[str] = {
    ".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".wma", ".webm",
    ".mkv", ".mp4", ".avi", ".mov"
}
MAX_FILE_SIZE: int = 0

model_cache: Dict[str, WhisperModel] = {}
model_cache_lock = asyncio.Lock()

# Required files for faster-whisper models (only download what's needed)
REQUIRED_MODEL_FILES = [
    "config.json",
    "model.bin",
    "tokenizer.json",
    "vocabulary.json",
    "preprocessor_config.json",
]


def get_model_status(model_size: str, cache_dir: Optional[str] = None) -> Dict:
    """Get detailed model cache status.

    Args:
        model_size: Whisper model size.
        cache_dir: Cache directory path. Defaults to WHISPER_CACHE_DIR.

    Returns:
        Dict with cached status, missing files, and total size.
    """
    if cache_dir is None:
        cache_dir = WHISPER_CACHE_DIR

    repo_id = MODEL_REPO_MAP.get(model_size)
    if not repo_id:
        return {"cached": False, "error": "Unknown model"}

    cache_name = f"models--{repo_id.replace('/', '--')}"
    model_cache_path = Path(cache_dir) / cache_name

    if not model_cache_path.exists():
        return {"cached": False, "repo_id": repo_id, "missing_files": REQUIRED_MODEL_FILES}

    snapshots_dir = model_cache_path / "snapshots"
    if not snapshots_dir.exists():
        return {"cached": False, "repo_id": repo_id, "missing_files": REQUIRED_MODEL_FILES}

    # Check for most recent snapshot
    snapshot_dirs = sorted(snapshots_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)
    for snapshot_dir in snapshot_dirs:
        if snapshot_dir.is_dir():
            # Check which files exist
            existing_files = []
            missing_files = []
            total_size = 0

            for filename in REQUIRED_MODEL_FILES:
                file_path = snapshot_dir / filename
                if file_path.exists():
                    existing_files.append(filename)
                    total_size += file_path.stat().st_size
                else:
                    missing_files.append(filename)

            # Model is cached if model.bin exists (the main file)
            if "model.bin" in existing_files:
                return {
                    "cached": True,
                    "repo_id": repo_id,
                    "snapshot_dir": str(snapshot_dir),
                    "existing_files": existing_files,
                    "missing_files": missing_files,
                    "total_size": total_size,
                }

    return {"cached": False, "repo_id": repo_id, "missing_files": REQUIRED_MODEL_FILES}


def download_model_files(
    model_size: str,
    cache_dir: Optional[str] = None,
    progress_callback: Optional[ProgressCallback] = None
) -> str:
    """Download only required model files with progress tracking.

    Args:
        model_size: Whisper model size.
        cache_dir: Cache directory path. Defaults to WHISPER_CACHE_DIR.
        progress_callback: Optional callback for progress updates.

    Returns:
        Path to the downloaded model directory.
    """
    if cache_dir is None:
        cache_dir = WHISPER_CACHE_DIR

    repo_id = MODEL_REPO_MAP.get(model_size)
    if not repo_id:
        raise ValueError(f"Unknown model: {model_size}")

    # Get file sizes from HuggingFace
    fs = HfFileSystem()
    files_info = {}
    total_size = 0

    if progress_callback:
        progress_callback("checking", "Checking model files...", 0)

    for filename in REQUIRED_MODEL_FILES:
        try:
            file_info = fs.info(f"{repo_id}/{filename}")
            files_info[filename] = file_info.get("size", 0)
            total_size += files_info[filename]
        except Exception as e:
            logger.warning(f"Could not get info for {filename}: {e}")
            files_info[filename] = 0

    # Download each file with progress tracking
    downloaded_size = 0
    downloaded_path = None

    for i, filename in enumerate(REQUIRED_MODEL_FILES):
        file_size = files_info.get(filename, 0)
        file_size_mb = file_size / (1024 * 1024) if file_size else 0

        if progress_callback:
            overall_progress = downloaded_size / total_size if total_size > 0 else 0
            progress_callback(
                "downloading",
                f"Downloading {filename} ({file_size_mb:.1f} MB)...",
                overall_progress
            )

        try:
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=cache_dir,
                local_files_only=False,
            )
            downloaded_size += file_size
            logger.info(f"Downloaded {filename} ({file_size_mb:.1f} MB)")
        except Exception as e:
            logger.error(f"Failed to download {filename}: {e}")
            raise

    if progress_callback:
        progress_callback("downloading", "Download complete", 1.0)

    # Return the model directory (parent of the downloaded files)
    if downloaded_path:
        return str(Path(downloaded_path).parent)
    return cache_dir


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal attacks.

    Args:
        filename: Original filename.

    Returns:
        Sanitized filename safe for filesystem operations.
    """
    safe_name = re.sub(r'[/\\]', '_', filename)
    safe_name = re.sub(r'\.\.', '_', safe_name)
    safe_name = re.sub(r'[^a-zA-Z0-9._-]', '_', safe_name)
    safe_name = safe_name.strip('_.')
    return safe_name if safe_name else "unnamed_file"


def is_valid_audio_extension(extension: str) -> bool:
    """Check if file extension is valid audio format.

    Args:
        extension: File extension including dot.

    Returns:
        True if extension is valid audio format.
    """
    return extension.lower() in ALLOWED_AUDIO_EXTENSIONS


def is_model_cached(model_size: str, cache_dir: Optional[str] = None) -> bool:
    """Check if a Whisper model is already cached locally.

    Args:
        model_size: Whisper model size (e.g., 'tiny', 'base', 'distil-large-v3').
        cache_dir: Cache directory path. Defaults to WHISPER_CACHE_DIR.

    Returns:
        True if the model exists in cache with all required files.
    """
    if cache_dir is None:
        cache_dir = WHISPER_CACHE_DIR

    # Get the HuggingFace repo ID for this model
    repo_id = MODEL_REPO_MAP.get(model_size)
    if not repo_id:
        # If not in our map, assume it's already a repo ID or local path
        # In this case, we can't reliably detect cache, so return False
        # to let faster-whisper handle it
        logger.debug(f"Model '{model_size}' not in repo map, skipping cache check")
        return False

    # Convert repo ID to cache directory name (org--model)
    cache_name = f"models--{repo_id.replace('/', '--')}"
    model_cache_path = Path(cache_dir) / cache_name

    if not model_cache_path.exists():
        logger.debug(f"Cache directory not found: {model_cache_path}")
        return False

    # Check for snapshots directory with at least one snapshot
    snapshots_dir = model_cache_path / "snapshots"
    if not snapshots_dir.exists():
        logger.debug(f"Snapshots directory not found: {snapshots_dir}")
        return False

    # Check if there's at least one snapshot with the model.bin file
    snapshot_dirs = list(snapshots_dir.iterdir()) if snapshots_dir.exists() else []
    for snapshot_dir in snapshot_dirs:
        if snapshot_dir.is_dir():
            model_bin = snapshot_dir / "model.bin"
            if model_bin.exists():
                logger.debug(f"Model cache found at: {snapshot_dir}")
                return True

    logger.debug(f"No complete snapshot found in: {snapshots_dir}")
    return False


async def get_whisper_model(
    model_size: str,
    device: str,
    progress_callback: Optional[ProgressCallback] = None
) -> WhisperModel:
    """Load Whisper model from cache or create new instance.

    Uses double-check locking for thread safety.

    Args:
        model_size: Whisper model size (tiny, base, small, medium, large-v3).
        device: Compute device (cpu, cuda, mps).
        progress_callback: Optional callback for status/progress updates.

    Returns:
        Cached or newly loaded WhisperModel instance.
    """
    model_key = f"{model_size}_{device}"

    if model_key in model_cache:
        if progress_callback:
            progress_callback("ready", "Model already loaded", 1.0)
        return model_cache[model_key]

    async with model_cache_lock:
        if model_key in model_cache:
            if progress_callback:
                progress_callback("ready", "Model already loaded", 1.0)
            return model_cache[model_key]

        # Check if model is already cached to avoid network requests
        use_local_only = is_model_cached(model_size, WHISPER_CACHE_DIR)

        if use_local_only:
            logger.info(f"Loading Whisper model '{model_size}' on '{device}' from local cache (offline mode)")
            if progress_callback:
                progress_callback("loading", f"Loading {model_size} on {device}...", 0.5)
        else:
            logger.info(f"Downloading Whisper model '{model_size}' on '{device}' to cache dir: {WHISPER_CACHE_DIR}")
            # Download files with progress tracking
            if progress_callback:
                progress_callback("downloading", f"Downloading {model_size}...", 0)
            await asyncio.to_thread(
                download_model_files,
                model_size,
                WHISPER_CACHE_DIR,
                progress_callback
            )
            if progress_callback:
                progress_callback("loading", f"Loading {model_size} on {device}...", 0.9)

        compute_type = "int8" if device == "cpu" else "int8_float16"

        model = await asyncio.to_thread(
            WhisperModel,
            model_size,
            device=device,
            compute_type=compute_type,
            download_root=WHISPER_CACHE_DIR,
            local_files_only=True  # Files are now guaranteed to be downloaded
        )

        model_cache[model_key] = model
        if progress_callback:
            progress_callback("ready", "Model loaded successfully", 1.0)
        logger.info(f"Whisper model '{model_key}' loaded and cached.")
        return model


def get_whisper_model_sync(model_size: str, device: str) -> WhisperModel:
    """Synchronous version of get_whisper_model for background tasks.

    Args:
        model_size: Whisper model size.
        device: Compute device.

    Returns:
        Cached or newly loaded WhisperModel instance.
    """
    model_key = f"{model_size}_{device}"

    if model_key in model_cache:
        return model_cache[model_key]

    # Check if model is already cached to avoid network requests
    use_local_only = is_model_cached(model_size, WHISPER_CACHE_DIR)
    if use_local_only:
        logger.info(f"Loading Whisper model '{model_size}' on '{device}' from local cache (offline mode)")
    else:
        logger.info(f"Downloading Whisper model '{model_size}' on '{device}' to cache dir: {WHISPER_CACHE_DIR}")

    compute_type = "int8" if device == "cpu" else "int8_float16"
    model = WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
        download_root=WHISPER_CACHE_DIR,
        local_files_only=use_local_only
    )
    model_cache[model_key] = model
    logger.info(f"Whisper model '{model_key}' loaded and cached.")
    return model
