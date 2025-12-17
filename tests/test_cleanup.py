# tests/test_cleanup.py
"""Tests for session and job cleanup functionality."""

import asyncio
import os
import time
import pytest
from unittest.mock import MagicMock, patch


class TestSessionCleanup:
    """Tests for session cleanup with TTL."""

    def test_session_info_has_timestamps(self):
        """SessionInfo should track creation and last activity times."""
        from cleanup import SessionInfo

        before = time.time()
        info = SessionInfo(session_id="test-123", session=MagicMock())
        after = time.time()

        assert before <= info.created_at <= after
        assert before <= info.last_activity <= after

    def test_session_info_update_activity(self):
        """update_activity should update last_activity timestamp."""
        from cleanup import SessionInfo

        info = SessionInfo(session_id="test-123", session=MagicMock())
        old_activity = info.last_activity

        time.sleep(0.01)  # Small delay
        info.update_activity()

        assert info.last_activity > old_activity

    def test_session_info_is_expired(self):
        """is_expired should return True for sessions past TTL."""
        from cleanup import SessionInfo

        info = SessionInfo(session_id="test-123", session=MagicMock())

        # Not expired with large TTL
        assert not info.is_expired(ttl_seconds=3600)

        # Manually set old timestamp
        info.last_activity = time.time() - 120

        # Expired with small TTL
        assert info.is_expired(ttl_seconds=60)


class TestJobCleanup:
    """Tests for transcription job cleanup."""

    def test_job_info_has_timestamps(self):
        """JobInfo should track creation and completion times."""
        from cleanup import JobInfo

        before = time.time()
        info = JobInfo(job_id="job-123", status="processing")
        after = time.time()

        assert before <= info.created_at <= after
        assert info.completed_at is None

    def test_job_info_mark_completed(self):
        """mark_completed should set completed_at and update status."""
        from cleanup import JobInfo

        info = JobInfo(job_id="job-123", status="processing")
        assert info.completed_at is None

        before = time.time()
        info.mark_completed("completed", "Transcription result")
        after = time.time()

        assert info.status == "completed"
        assert info.result == "Transcription result"
        assert before <= info.completed_at <= after

    def test_job_info_is_expired_not_completed(self):
        """In-progress jobs should never be expired."""
        from cleanup import JobInfo

        info = JobInfo(job_id="job-123", status="processing")
        # Even with very old timestamp
        info.created_at = time.time() - 7200  # 2 hours ago

        # In-progress jobs are never expired
        assert not info.is_expired(retention_seconds=60)

    def test_job_info_is_expired_completed(self):
        """Completed jobs should expire after retention period."""
        from cleanup import JobInfo

        info = JobInfo(job_id="job-123", status="processing")
        info.mark_completed("completed", "result")

        # Not expired immediately
        assert not info.is_expired(retention_seconds=3600)

        # Manually set old completion time
        info.completed_at = time.time() - 120

        # Expired after retention period
        assert info.is_expired(retention_seconds=60)


class TestCleanupManager:
    """Tests for the cleanup manager."""

    def test_cleanup_manager_removes_expired_sessions(self):
        """Expired sessions should be removed during cleanup."""
        from cleanup import SessionInfo, CleanupManager

        manager = CleanupManager(session_ttl_seconds=60, job_retention_seconds=120)

        # Add sessions
        active_session = SessionInfo(session_id="active", session=MagicMock())
        expired_session = SessionInfo(session_id="expired", session=MagicMock())
        expired_session.last_activity = time.time() - 120  # 2 minutes ago

        sessions = {"active": active_session, "expired": expired_session}
        jobs = {}

        removed_sessions, removed_jobs = manager.cleanup(sessions, jobs)

        assert "expired" in removed_sessions
        assert "active" not in removed_sessions
        assert "active" in sessions
        assert "expired" not in sessions

    def test_cleanup_manager_removes_expired_jobs(self):
        """Expired completed jobs should be removed during cleanup."""
        from cleanup import JobInfo, CleanupManager

        manager = CleanupManager(session_ttl_seconds=60, job_retention_seconds=60)

        # Add jobs
        active_job = JobInfo(job_id="active", status="processing")
        completed_job = JobInfo(job_id="completed", status="processing")
        completed_job.mark_completed("completed", "result")
        completed_job.completed_at = time.time() - 120  # 2 minutes ago

        sessions = {}
        jobs = {"active": active_job, "completed": completed_job}

        removed_sessions, removed_jobs = manager.cleanup(sessions, jobs)

        assert "completed" in removed_jobs
        assert "active" not in removed_jobs
        assert "active" in jobs
        assert "completed" not in jobs

    def test_cleanup_manager_keeps_in_progress_jobs(self):
        """In-progress jobs should never be removed regardless of age."""
        from cleanup import JobInfo, CleanupManager

        manager = CleanupManager(session_ttl_seconds=60, job_retention_seconds=60)

        # Old in-progress job
        old_job = JobInfo(job_id="old-processing", status="processing")
        old_job.created_at = time.time() - 7200  # 2 hours ago

        sessions = {}
        jobs = {"old-processing": old_job}

        removed_sessions, removed_jobs = manager.cleanup(sessions, jobs)

        assert "old-processing" not in removed_jobs
        assert "old-processing" in jobs

    def test_cleanup_manager_cleans_temp_files(self):
        """Cleanup should remove associated temp files for expired sessions."""
        import os
        import tempfile
        from cleanup import SessionInfo, CleanupManager

        manager = CleanupManager(session_ttl_seconds=60, job_retention_seconds=120)

        # Create a temp file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pcm")
        temp_file.write(b"test data")
        temp_file.close()

        # Create expired session with temp file
        mock_session = MagicMock()
        mock_session.temp_file = MagicMock()
        mock_session.temp_file.name = temp_file.name

        expired_session = SessionInfo(session_id="expired", session=mock_session)
        expired_session.last_activity = time.time() - 120

        sessions = {"expired": expired_session}
        jobs = {}

        assert os.path.exists(temp_file.name)

        manager.cleanup(sessions, jobs)

        # Temp file should be deleted
        assert not os.path.exists(temp_file.name)


class TestCleanupConfig:
    """Tests for cleanup configuration."""

    def test_default_cleanup_parameters(self):
        """Default cleanup parameters should be set."""
        from cleanup import get_cleanup_config

        config = get_cleanup_config()

        assert "session_ttl_minutes" in config
        assert "job_retention_minutes" in config
        assert "cleanup_interval_seconds" in config
        assert config["session_ttl_minutes"] > 0
        assert config["job_retention_minutes"] > 0

    def test_cleanup_parameters_from_config(self):
        """Cleanup should read parameters from config.yaml."""
        from cleanup import get_cleanup_config

        with patch('cleanup.get_config') as mock_config:
            mock_config.return_value = {
                "cleanup_parameters": {
                    "session_ttl_minutes": 30,
                    "job_retention_minutes": 60,
                    "cleanup_interval_seconds": 120
                }
            }

            config = get_cleanup_config()

            assert config["session_ttl_minutes"] == 30
            assert config["job_retention_minutes"] == 60
            assert config["cleanup_interval_seconds"] == 120


class TestTempDirectoryManager:
    """Tests for temporary directory management."""

    def test_get_temp_dir_creates_directory(self):
        """get_temp_dir should create the temp directory if it doesn't exist."""
        from cleanup import get_temp_dir
        import shutil

        temp_dir = get_temp_dir()
        assert os.path.exists(temp_dir)
        assert os.path.isdir(temp_dir)
        assert "echoscribe" in temp_dir.lower()

    def test_get_temp_dir_returns_same_path(self):
        """get_temp_dir should return the same path on repeated calls."""
        from cleanup import get_temp_dir

        dir1 = get_temp_dir()
        dir2 = get_temp_dir()
        assert dir1 == dir2

    def test_cleanup_orphaned_temp_files(self):
        """cleanup_orphaned_temp_files should remove old temp files."""
        import tempfile
        from cleanup import get_temp_dir, cleanup_orphaned_temp_files

        temp_dir = get_temp_dir()

        # Create a temp file
        test_file = os.path.join(temp_dir, "temp_test_orphan.pcm")
        with open(test_file, "w") as f:
            f.write("test")

        assert os.path.exists(test_file)

        # Cleanup should remove it (it's orphaned since no session owns it)
        cleanup_orphaned_temp_files(temp_dir, max_age_seconds=0)

        assert not os.path.exists(test_file)

    def test_cleanup_orphaned_preserves_recent_files(self):
        """cleanup_orphaned_temp_files should preserve recent files."""
        import tempfile
        from cleanup import get_temp_dir, cleanup_orphaned_temp_files

        temp_dir = get_temp_dir()

        # Create a temp file
        test_file = os.path.join(temp_dir, "temp_test_recent.pcm")
        with open(test_file, "w") as f:
            f.write("test")

        assert os.path.exists(test_file)

        # Cleanup with large max_age should preserve it
        cleanup_orphaned_temp_files(temp_dir, max_age_seconds=3600)

        assert os.path.exists(test_file)

        # Clean up after test
        os.unlink(test_file)
