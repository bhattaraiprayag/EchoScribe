"""Session and job cleanup functionality with TTL support."""

import atexit
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from config_manager import get_config


logger = logging.getLogger(__name__)

_temp_dir: Optional[str] = None

DEFAULT_SESSION_TTL_MINUTES = 60
DEFAULT_JOB_RETENTION_MINUTES = 120
DEFAULT_CLEANUP_INTERVAL_SECONDS = 300


def get_cleanup_config() -> Dict[str, Any]:
    """Get cleanup configuration from config.yaml or defaults.

    Returns:
        Dictionary with cleanup parameters.
    """
    config = get_config()
    cleanup_params = config.get("cleanup_parameters", {})

    return {
        "session_ttl_minutes": cleanup_params.get(
            "session_ttl_minutes", DEFAULT_SESSION_TTL_MINUTES
        ),
        "job_retention_minutes": cleanup_params.get(
            "job_retention_minutes", DEFAULT_JOB_RETENTION_MINUTES
        ),
        "cleanup_interval_seconds": cleanup_params.get(
            "cleanup_interval_seconds", DEFAULT_CLEANUP_INTERVAL_SECONDS
        ),
    }


@dataclass
class SessionInfo:
    """Wrapper for TranscriptionSession with TTL tracking."""

    session_id: str
    session: Any  # TranscriptionSession
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)

    def update_activity(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = time.time()

    def is_expired(self, ttl_seconds: float) -> bool:
        """Check if session has expired based on TTL.

        Args:
            ttl_seconds: Time-to-live in seconds.

        Returns:
            True if session has been inactive longer than TTL.
        """
        return (time.time() - self.last_activity) > ttl_seconds


@dataclass
class JobInfo:
    """Wrapper for transcription job with TTL tracking."""

    job_id: str
    status: str
    result: str = ""
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    cancelled: bool = False

    def mark_completed(self, status: str, result: str) -> None:
        """Mark job as completed with result.

        Args:
            status: Final status (completed, error, cancelled).
            result: Transcription result or error message.
        """
        self.status = status
        self.result = result
        self.completed_at = time.time()

    def cancel(self) -> None:
        """Mark job as cancelled."""
        self.cancelled = True

    def is_expired(self, retention_seconds: float) -> bool:
        """Check if completed job has expired based on retention period.

        Args:
            retention_seconds: Retention period in seconds.

        Returns:
            True if job is completed and past retention period.
        """
        if self.completed_at is None:
            return False

        return (time.time() - self.completed_at) > retention_seconds

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses.

        Returns:
            Dictionary with status, result, and cancelled fields.
        """
        return {
            "status": self.status,
            "result": self.result,
            "cancelled": self.cancelled
        }


class CleanupManager:
    """Manages cleanup of expired sessions and jobs."""

    def __init__(
        self,
        session_ttl_seconds: float = DEFAULT_SESSION_TTL_MINUTES * 60,
        job_retention_seconds: float = DEFAULT_JOB_RETENTION_MINUTES * 60,
    ):
        """Initialize cleanup manager.

        Args:
            session_ttl_seconds: Session TTL in seconds.
            job_retention_seconds: Completed job retention in seconds.
        """
        self.session_ttl_seconds = session_ttl_seconds
        self.job_retention_seconds = job_retention_seconds

    def cleanup(
        self,
        sessions: Dict[str, SessionInfo],
        jobs: Dict[str, JobInfo],
    ) -> Tuple[List[str], List[str]]:
        """Remove expired sessions and jobs.

        Args:
            sessions: Dictionary mapping session_id to SessionInfo.
            jobs: Dictionary mapping job_id to JobInfo.

        Returns:
            Tuple of removed session IDs and removed job IDs.
        """
        removed_sessions = []
        removed_jobs = []

        expired_session_ids = [
            sid for sid, info in sessions.items()
            if info.is_expired(self.session_ttl_seconds)
        ]

        for session_id in expired_session_ids:
            session_info = sessions.pop(session_id)
            self._cleanup_session_resources(session_info)
            removed_sessions.append(session_id)
            logger.info(f"[{session_id}] Session expired and cleaned up")

        expired_job_ids = [
            jid for jid, info in jobs.items()
            if info.is_expired(self.job_retention_seconds)
        ]

        for job_id in expired_job_ids:
            jobs.pop(job_id)
            removed_jobs.append(job_id)
            logger.info(f"[{job_id}] Job expired and cleaned up")

        return removed_sessions, removed_jobs

    def _cleanup_session_resources(self, session_info: SessionInfo) -> None:
        """Clean up resources associated with a session.

        Args:
            session_info: The session to clean up.
        """
        try:
            session = session_info.session
            if hasattr(session, 'temp_file') and session.temp_file:
                temp_path = session.temp_file.name
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    logger.info(
                        f"[{session_info.session_id}] Temp file deleted: "
                        f"{temp_path}"
                    )
        except Exception as e:
            logger.error(
                f"[{session_info.session_id}] "
                f"Error cleaning session resources: {e}"
            )


def get_temp_dir() -> str:
    """Get or create the dedicated temp directory for EchoScribe.

    Returns:
        Path to the temp directory.
    """
    global _temp_dir

    if _temp_dir is not None and os.path.exists(_temp_dir):
        return _temp_dir

    # Create temp directory in system temp location
    _temp_dir = os.path.join(tempfile.gettempdir(), "echoscribe_temp")
    os.makedirs(_temp_dir, exist_ok=True)
    logger.info(f"Temp directory created: {_temp_dir}")

    return _temp_dir


def cleanup_orphaned_temp_files(
    temp_dir: str,
    max_age_seconds: float = 3600
) -> int:
    """Remove orphaned temp files older than max_age_seconds.

    Args:
        temp_dir: Path to temp directory.
        max_age_seconds: Maximum age in seconds for temp files.

    Returns:
        Number of files removed.
    """
    if not os.path.exists(temp_dir):
        return 0

    removed_count = 0
    current_time = time.time()

    try:
        for filename in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, filename)
            if not os.path.isfile(file_path):
                continue

            try:
                file_age = current_time - os.path.getmtime(file_path)
                # When max_age_seconds <= 0, delete all files regardless of age
                # This also handles potential clock precision issues on Windows
                if max_age_seconds <= 0 or file_age >= max_age_seconds:
                    os.unlink(file_path)
                    removed_count += 1
                    logger.info(f"Removed orphaned temp file: {file_path}")
            except OSError as e:
                logger.warning(f"Could not remove temp file {file_path}: {e}")
    except Exception as e:
        logger.error(f"Error during orphaned file cleanup: {e}")

    return removed_count


def cleanup_temp_directory() -> None:
    """Clean up entire temp directory at shutdown."""
    global _temp_dir

    if _temp_dir and os.path.exists(_temp_dir):
        try:
            for filename in os.listdir(_temp_dir):
                file_path = os.path.join(_temp_dir, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except OSError as e:
                    logger.warning(f"Could not remove {file_path}: {e}")
            logger.info(f"Temp directory cleaned: {_temp_dir}")
        except Exception as e:
            logger.error(f"Error cleaning temp directory: {e}")
