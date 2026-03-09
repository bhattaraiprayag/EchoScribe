"""Utility functions for file validation, sanitization, and model caching."""

import asyncio
import json
import logging
import os
import re
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Set

from faster_whisper import WhisperModel
from faster_whisper.utils import download_model as download_model_snapshot

logger = logging.getLogger(__name__)

# Type alias for progress callback
ProgressCallback = Callable[[str, str, float], None]  # (status, message, progress)

# Model cache directory configuration
# Use project root (parent of backend/) to ensure consistent location regardless of cwd
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_CACHE_DIR = os.getenv("MODELS_CACHE_DIR", str(_PROJECT_ROOT / "models_cache"))
WHISPER_CACHE_DIR = os.path.join(MODELS_CACHE_DIR, "whisper_models")
VAD_CACHE_DIR = os.path.join(MODELS_CACHE_DIR, "silero_vad")


def _get_whisper_cache_dir(cache_dir: Optional[str] = None) -> str:
    """Return the configured Whisper cache directory."""
    return cache_dir or WHISPER_CACHE_DIR


def _ensure_whisper_cache_dir(cache_dir: Optional[str] = None) -> str:
    """Ensure the Whisper cache directory exists before writes occur."""
    resolved_dir = Path(_get_whisper_cache_dir(cache_dir))
    try:
        resolved_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError as exc:
        raise PermissionError(
            f"Cannot write to Whisper cache directory '{resolved_dir}'. "
            "Ensure the models cache path is writable by the application user "
            "or set MODELS_CACHE_DIR to a writable location."
        ) from exc
    return str(resolved_dir)


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
    ".mp3",
    ".wav",
    ".m4a",
    ".ogg",
    ".flac",
    ".aac",
    ".wma",
    ".webm",
    ".mkv",
    ".mp4",
    ".avi",
    ".mov",
}
DEFAULT_MAX_FILE_SIZE_MB = 100

model_cache: Dict[str, WhisperModel] = {}
model_cache_lock = asyncio.Lock()
sync_model_cache_lock = threading.Lock()  # For synchronous access in background tasks

REQUIRED_MODEL_PATTERNS = [
    "config.json",
    "model.bin",
    "tokenizer.json",
    "vocabulary.*",
]
OPTIONAL_MODEL_FILES = ["preprocessor_config.json"]
VOCABULARY_PATTERN = "vocabulary.*"
REPAIRED_VOCABULARY_FILENAME = "vocabulary.json"


def get_max_file_size_bytes(config: Optional[Dict[str, Any]] = None) -> int:
    """Get effective max upload size in bytes from configuration."""
    if config is None:
        from backend.config_manager import get_config

        config = get_config()

    upload_config = config.get("upload_parameters", {})
    max_file_size_mb = upload_config.get("max_file_size_mb", DEFAULT_MAX_FILE_SIZE_MB)
    if not isinstance(max_file_size_mb, int) or max_file_size_mb < 1:
        raise ValueError("upload_parameters.max_file_size_mb must be an integer >= 1")
    return max_file_size_mb * 1024 * 1024


def _get_model_snapshot_dir(
    model_size: str, cache_dir: Optional[str] = None
) -> tuple[Optional[Path], Optional[str]]:
    """Return the newest cached snapshot directory for a model, if present."""
    cache_dir = _get_whisper_cache_dir(cache_dir)

    repo_id = MODEL_REPO_MAP.get(model_size)
    if not repo_id:
        return None, None

    cache_name = f"models--{repo_id.replace('/', '--')}"
    snapshots_dir = Path(cache_dir) / cache_name / "snapshots"
    if not snapshots_dir.exists():
        return None, repo_id

    snapshot_dirs = sorted(
        (path for path in snapshots_dir.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not snapshot_dirs:
        return None, repo_id

    return snapshot_dirs[0], repo_id


def _find_matching_snapshot_files(snapshot_dir: Path, pattern: str) -> list[Path]:
    """Return sorted files in a snapshot directory matching a required pattern."""
    return sorted(path for path in snapshot_dir.glob(pattern) if path.is_file())


def _display_pattern_name(pattern: str) -> str:
    """Normalize wildcard patterns to user-facing missing file names."""
    return REPAIRED_VOCABULARY_FILENAME if pattern == VOCABULARY_PATTERN else pattern


def repair_model_snapshot(snapshot_dir: Path) -> list[str]:
    """Repair a partial snapshot by synthesizing vocabulary.json from tokenizer.json."""
    if _find_matching_snapshot_files(snapshot_dir, VOCABULARY_PATTERN):
        return []

    tokenizer_path = snapshot_dir / "tokenizer.json"
    if not tokenizer_path.exists():
        return []

    try:
        with tokenizer_path.open("r", encoding="utf-8") as handle:
            tokenizer_payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "tokenizer.json is invalid and cannot repair vocabulary"
        ) from exc

    tokenizer_model = tokenizer_payload.get("model", {})
    vocab_entries = tokenizer_model.get("vocab")
    if not isinstance(vocab_entries, dict) or not vocab_entries:
        raise RuntimeError("tokenizer.json does not contain a usable vocabulary map")

    combined_vocab: Dict[str, int] = {}
    for token, index in vocab_entries.items():
        if not isinstance(token, str) or not isinstance(index, int):
            raise RuntimeError("tokenizer.json contains invalid vocabulary entries")
        combined_vocab[token] = index

    for token_info in tokenizer_payload.get("added_tokens", []):
        token = token_info.get("content")
        index = token_info.get("id")
        if isinstance(token, str) and isinstance(index, int):
            combined_vocab[token] = index

    if not combined_vocab:
        raise RuntimeError("tokenizer.json does not contain any vocabulary entries")

    max_index = max(combined_vocab.values())
    if max_index < 0:
        raise RuntimeError("tokenizer vocabulary contains invalid token IDs")

    vocabulary: list[Optional[str]] = [None] * (max_index + 1)
    for token, index in combined_vocab.items():
        vocabulary[index] = token

    if any(token is None for token in vocabulary):
        raise RuntimeError(
            "tokenizer vocabulary IDs are sparse; cannot synthesize vocabulary"
        )

    vocabulary_path = snapshot_dir / REPAIRED_VOCABULARY_FILENAME
    with vocabulary_path.open("w", encoding="utf-8") as handle:
        json.dump(vocabulary, handle)

    logger.info("Synthesized missing vocabulary file at %s", vocabulary_path)
    return [REPAIRED_VOCABULARY_FILENAME]


def get_model_status(model_size: str, cache_dir: Optional[str] = None) -> Dict:
    """Get detailed model cache status.

    Args:
        model_size: Whisper model size.
        cache_dir: Cache directory path. Defaults to WHISPER_CACHE_DIR.

    Returns:
        Dict with cached status, missing files, and total size.
    """
    snapshot_dir, repo_id = _get_model_snapshot_dir(model_size, cache_dir)
    if not repo_id:
        return {"cached": False, "error": "Unknown model"}

    if snapshot_dir is None:
        return {
            "cached": False,
            "repo_id": repo_id,
            "missing_files": [
                _display_pattern_name(pattern) for pattern in REQUIRED_MODEL_PATTERNS
            ],
            "optional_missing_files": OPTIONAL_MODEL_FILES.copy(),
            "repairable_files": [],
        }

    existing_files: list[str] = []
    missing_files: list[str] = []
    optional_missing_files: list[str] = []
    repairable_files: list[str] = []
    total_size = 0

    for pattern in REQUIRED_MODEL_PATTERNS:
        matches = _find_matching_snapshot_files(snapshot_dir, pattern)
        if matches:
            chosen_file = matches[0]
            existing_files.append(chosen_file.name)
            total_size += chosen_file.stat().st_size
        else:
            missing_name = _display_pattern_name(pattern)
            missing_files.append(missing_name)
            if (
                pattern == VOCABULARY_PATTERN
                and (snapshot_dir / "tokenizer.json").exists()
            ):
                repairable_files.append(missing_name)

    for filename in OPTIONAL_MODEL_FILES:
        file_path = snapshot_dir / filename
        if file_path.exists():
            existing_files.append(filename)
            total_size += file_path.stat().st_size
        else:
            optional_missing_files.append(filename)

    return {
        "cached": not missing_files,
        "repo_id": repo_id,
        "snapshot_dir": str(snapshot_dir),
        "existing_files": existing_files,
        "missing_files": missing_files,
        "optional_missing_files": optional_missing_files,
        "repairable_files": repairable_files,
        "total_size": total_size,
    }


def download_model_files(
    model_size: str,
    cache_dir: Optional[str] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> str:
    """Download or refresh a Whisper snapshot, then validate and repair it.

    Args:
        model_size: Whisper model size.
        cache_dir: Cache directory path. Defaults to WHISPER_CACHE_DIR.
        progress_callback: Optional callback for progress updates.

    Returns:
        Path to the downloaded model directory.
    """
    repo_id = MODEL_REPO_MAP.get(model_size)
    if not repo_id:
        raise ValueError(f"Unknown model: {model_size}")

    cache_dir = _ensure_whisper_cache_dir(cache_dir)

    if progress_callback:
        progress_callback(
            "downloading",
            f"Downloading {model_size} model files...",
            0.25,
        )

    downloaded_path = Path(
        download_model_snapshot(
            model_size,
            cache_dir=cache_dir,
            local_files_only=False,
        )
    )

    repaired_files = repair_model_snapshot(downloaded_path)
    if repaired_files and progress_callback:
        progress_callback(
            "loading",
            f"Repairing cached metadata: {', '.join(repaired_files)}",
            0.8,
        )

    status = get_model_status(model_size, cache_dir)
    if not status.get("cached"):
        missing_files = ", ".join(status.get("missing_files", []))
        raise RuntimeError(
            f"Model cache is incomplete after refresh. Missing files: {missing_files}"
        )

    if progress_callback:
        progress_callback("loading", "Model files ready", 0.9)

    return str(downloaded_path)


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal attacks.

    Args:
        filename: Original filename.

    Returns:
        Sanitized filename safe for filesystem operations.
    """
    safe_name = re.sub(r"[/\\]", "_", filename)
    safe_name = re.sub(r"\.\.", "_", safe_name)
    safe_name = re.sub(r"[^a-zA-Z0-9._-]", "_", safe_name)
    safe_name = safe_name.strip("_.")
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
    status = get_model_status(model_size, cache_dir)
    return bool(status.get("cached"))


def ensure_model_files(
    model_size: str,
    cache_dir: Optional[str] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> str:
    """Ensure model files exist locally and return the ready snapshot directory."""
    cache_dir = _get_whisper_cache_dir(cache_dir)

    if progress_callback:
        progress_callback("checking", "Checking model cache...", 0.0)

    status = get_model_status(model_size, cache_dir)
    snapshot_dir = status.get("snapshot_dir")
    if status.get("cached") and snapshot_dir:
        return snapshot_dir

    if snapshot_dir and status.get("repairable_files"):
        if progress_callback:
            progress_callback("loading", "Repairing cached model metadata...", 0.2)
        repaired_files = repair_model_snapshot(Path(snapshot_dir))
        if repaired_files:
            status = get_model_status(model_size, cache_dir)
            if status.get("cached") and status.get("snapshot_dir"):
                return str(status["snapshot_dir"])

    return download_model_files(model_size, cache_dir, progress_callback)


def is_model_loaded(model_size: str, device: str) -> bool:
    """Return whether a model/device pair is already instantiated in memory."""
    return f"{model_size}_{device}" in model_cache


async def get_whisper_model(
    model_size: str, device: str, progress_callback: Optional[ProgressCallback] = None
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

        model_path = await asyncio.to_thread(
            ensure_model_files, model_size, None, progress_callback
        )
        logger.info(
            "Loading Whisper model '%s' on '%s' from %s",
            model_size,
            device,
            model_path,
        )
        if progress_callback:
            progress_callback("loading", f"Loading {model_size} on {device}...", 0.95)

        compute_type = "int8" if device == "cpu" else "int8_float16"

        model = await asyncio.to_thread(
            WhisperModel,
            model_path,
            device=device,
            compute_type=compute_type,
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

    # Quick check without lock (may be stale but avoids lock contention)
    if model_key in model_cache:
        return model_cache[model_key]

    # Acquire lock for thread-safe loading
    with sync_model_cache_lock:
        # Double-check inside lock to prevent duplicate loading
        if model_key in model_cache:
            return model_cache[model_key]

        model_path = ensure_model_files(model_size)
        logger.info(
            "Loading Whisper model '%s' on '%s' from %s",
            model_size,
            device,
            model_path,
        )

        compute_type = "int8" if device == "cpu" else "int8_float16"
        model = WhisperModel(
            model_path,
            device=device,
            compute_type=compute_type,
        )
        model_cache[model_key] = model
        logger.info(f"Whisper model '{model_key}' loaded and cached.")
        return model
