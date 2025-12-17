# tests/test_model_cache.py
"""Tests for model cache concurrency safety."""

import asyncio
import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

pytestmark = pytest.mark.asyncio


class TestIsModelCached:
    """Tests for is_model_cached function."""

    def test_cached_model_returns_true(self):
        """is_model_cached should return True when model exists with model.bin."""
        from utils import is_model_cached, MODEL_REPO_MAP

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create proper cache structure for distil-large-v3
            model_dir = Path(tmpdir) / "models--Systran--faster-distil-whisper-large-v3"
            snapshot_dir = model_dir / "snapshots" / "abc123"
            snapshot_dir.mkdir(parents=True)
            (snapshot_dir / "model.bin").touch()

            result = is_model_cached("distil-large-v3", tmpdir)
            assert result is True

    def test_missing_model_returns_false(self):
        """is_model_cached should return False when model directory doesn't exist."""
        from utils import is_model_cached

        with tempfile.TemporaryDirectory() as tmpdir:
            result = is_model_cached("distil-large-v3", tmpdir)
            assert result is False

    def test_missing_snapshot_returns_false(self):
        """is_model_cached should return False when snapshots directory is missing."""
        from utils import is_model_cached

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create model dir but no snapshots
            model_dir = Path(tmpdir) / "models--Systran--faster-distil-whisper-large-v3"
            model_dir.mkdir(parents=True)

            result = is_model_cached("distil-large-v3", tmpdir)
            assert result is False

    def test_missing_model_bin_returns_false(self):
        """is_model_cached should return False when model.bin is missing."""
        from utils import is_model_cached

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create snapshot dir but no model.bin
            model_dir = Path(tmpdir) / "models--Systran--faster-distil-whisper-large-v3"
            snapshot_dir = model_dir / "snapshots" / "abc123"
            snapshot_dir.mkdir(parents=True)
            # Don't create model.bin

            result = is_model_cached("distil-large-v3", tmpdir)
            assert result is False

    def test_unknown_model_returns_false(self):
        """is_model_cached should return False for models not in repo map."""
        from utils import is_model_cached

        with tempfile.TemporaryDirectory() as tmpdir:
            result = is_model_cached("unknown-model", tmpdir)
            assert result is False

    def test_all_standard_models_have_mapping(self):
        """All standard Whisper model sizes should have repo mappings."""
        from utils import MODEL_REPO_MAP

        standard_models = ["tiny", "base", "small", "medium", "large-v3", "distil-large-v3"]
        for model in standard_models:
            assert model in MODEL_REPO_MAP, f"Model '{model}' missing from MODEL_REPO_MAP"


class TestModelCacheLocking:
    """Tests for model cache thread safety."""

    async def test_get_whisper_model_returns_model(self):
        """get_whisper_model should return a WhisperModel instance."""
        from utils import get_whisper_model, model_cache, model_cache_lock

        # Clear cache
        model_cache.clear()

        # Mock WhisperModel, is_model_cached, and download_model_files
        mock_model = MagicMock()
        with patch('utils.WhisperModel', return_value=mock_model), \
             patch('utils.is_model_cached', return_value=False), \
             patch('utils.download_model_files', return_value='/mock/path'):
            result = await get_whisper_model("tiny", "cpu")
            assert result == mock_model

    async def test_get_whisper_model_caches_result(self):
        """Second call should return cached model, not load again."""
        from utils import get_whisper_model, model_cache

        model_cache.clear()

        mock_model = MagicMock()
        with patch('utils.WhisperModel', return_value=mock_model) as mock_class, \
             patch('utils.is_model_cached', return_value=False), \
             patch('utils.download_model_files', return_value='/mock/path'):
            # First call
            result1 = await get_whisper_model("tiny", "cpu")
            # Second call
            result2 = await get_whisper_model("tiny", "cpu")

            # Should only load once
            assert mock_class.call_count == 1
            assert result1 is result2

    async def test_different_models_cached_separately(self):
        """Different model sizes should be cached separately."""
        from utils import get_whisper_model, model_cache

        model_cache.clear()

        mock_model_tiny = MagicMock(name="tiny")
        mock_model_base = MagicMock(name="base")

        def create_model(model_size, **kwargs):
            if model_size == "tiny":
                return mock_model_tiny
            else:
                return mock_model_base

        with patch('utils.WhisperModel', side_effect=create_model), \
             patch('utils.is_model_cached', return_value=False), \
             patch('utils.download_model_files', return_value='/mock/path'):
            result_tiny = await get_whisper_model("tiny", "cpu")
            result_base = await get_whisper_model("base", "cpu")

            assert result_tiny is not result_base
            assert "tiny_cpu" in model_cache
            assert "base_cpu" in model_cache

    async def test_concurrent_requests_load_model_once(self):
        """Multiple concurrent requests for same model should only load once."""
        from utils import get_whisper_model, model_cache

        model_cache.clear()

        load_count = 0
        load_started = asyncio.Event()
        load_can_complete = asyncio.Event()

        def slow_load(*args, **kwargs):
            nonlocal load_count
            load_count += 1
            load_started.set()
            return MagicMock()

        with patch('utils.WhisperModel', side_effect=slow_load), \
             patch('utils.is_model_cached', return_value=False), \
             patch('utils.download_model_files', return_value='/mock/path'):
            # Start multiple concurrent requests
            tasks = [
                get_whisper_model("tiny", "cpu"),
                get_whisper_model("tiny", "cpu"),
                get_whisper_model("tiny", "cpu"),
            ]
            results = await asyncio.gather(*tasks)

            # All results should be the same model
            assert results[0] is results[1]
            assert results[1] is results[2]

            # Model should only be loaded once
            assert load_count == 1

    async def test_model_cache_lock_exists(self):
        """model_cache_lock should be an asyncio.Lock."""
        from utils import model_cache_lock

        assert isinstance(model_cache_lock, asyncio.Lock)

    async def test_get_whisper_model_uses_correct_compute_type(self):
        """CPU should use int8, CUDA should use int8_float16."""
        from utils import get_whisper_model, model_cache

        model_cache.clear()

        with patch('utils.WhisperModel') as mock_class, \
             patch('utils.is_model_cached', return_value=False), \
             patch('utils.download_model_files', return_value='/mock/path'):
            mock_class.return_value = MagicMock()

            # Test CPU compute type
            await get_whisper_model("tiny", "cpu")
            call_kwargs = mock_class.call_args
            assert call_kwargs.kwargs.get('compute_type') == "int8"

        model_cache.clear()

        with patch('utils.WhisperModel') as mock_class, \
             patch('utils.is_model_cached', return_value=False), \
             patch('utils.download_model_files', return_value='/mock/path'):
            mock_class.return_value = MagicMock()

            # Test CUDA compute type
            await get_whisper_model("tiny", "cuda")
            call_kwargs = mock_class.call_args
            assert call_kwargs.kwargs.get('compute_type') == "int8_float16"

    async def test_get_whisper_model_uses_local_files_only_when_cached(self):
        """When model is cached, local_files_only should be True."""
        from utils import get_whisper_model, model_cache

        model_cache.clear()

        with patch('utils.WhisperModel') as mock_class, \
             patch('utils.is_model_cached', return_value=True):
            mock_class.return_value = MagicMock()

            await get_whisper_model("tiny", "cpu")
            call_kwargs = mock_class.call_args
            assert call_kwargs.kwargs.get('local_files_only') is True

    async def test_get_whisper_model_downloads_when_not_cached(self):
        """When model is not cached, download_model_files is called first, then local_files_only=True."""
        from utils import get_whisper_model, model_cache

        model_cache.clear()

        with patch('utils.WhisperModel') as mock_class, \
             patch('utils.is_model_cached', return_value=False), \
             patch('utils.download_model_files', return_value='/mock/path') as mock_download:
            mock_class.return_value = MagicMock()

            await get_whisper_model("tiny", "cpu")

            # download_model_files should be called when not cached
            mock_download.assert_called_once()

            # After download, local_files_only should be True
            call_kwargs = mock_class.call_args
            assert call_kwargs.kwargs.get('local_files_only') is True


class TestGetModelStatus:
    """Tests for get_model_status function."""

    def test_get_model_status_cached_model(self):
        """get_model_status should return cached=True when model exists."""
        from utils import get_model_status

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create proper cache structure
            model_dir = Path(tmpdir) / "models--Systran--faster-whisper-tiny"
            snapshot_dir = model_dir / "snapshots" / "abc123"
            snapshot_dir.mkdir(parents=True)
            (snapshot_dir / "model.bin").touch()
            (snapshot_dir / "config.json").touch()
            (snapshot_dir / "tokenizer.json").touch()

            status = get_model_status("tiny", tmpdir)
            assert status["cached"] is True
            assert "model.bin" in status["existing_files"]
            assert "repo_id" in status

    def test_get_model_status_missing_model(self):
        """get_model_status should return cached=False when model doesn't exist."""
        from utils import get_model_status

        with tempfile.TemporaryDirectory() as tmpdir:
            status = get_model_status("tiny", tmpdir)
            assert status["cached"] is False
            assert "missing_files" in status

    def test_get_model_status_unknown_model(self):
        """get_model_status should return error for unknown model."""
        from utils import get_model_status

        status = get_model_status("unknown-model-xyz")
        assert status["cached"] is False
        assert "error" in status

    def test_get_model_status_returns_repo_id(self):
        """get_model_status should include repo_id in response."""
        from utils import get_model_status, MODEL_REPO_MAP

        status = get_model_status("base")
        assert "repo_id" in status
        assert status["repo_id"] == MODEL_REPO_MAP["base"]

    def test_get_model_status_partial_cache(self):
        """get_model_status should report missing files correctly."""
        from utils import get_model_status, REQUIRED_MODEL_FILES

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create cache with only some files
            model_dir = Path(tmpdir) / "models--Systran--faster-whisper-small"
            snapshot_dir = model_dir / "snapshots" / "abc123"
            snapshot_dir.mkdir(parents=True)
            (snapshot_dir / "model.bin").touch()
            # Only model.bin, missing other files

            status = get_model_status("small", tmpdir)
            assert status["cached"] is True  # model.bin exists
            assert len(status["missing_files"]) > 0
            assert "tokenizer.json" in status["missing_files"]


class TestRequiredModelFiles:
    """Tests for REQUIRED_MODEL_FILES constant."""

    def test_required_files_exist(self):
        """REQUIRED_MODEL_FILES should be defined and non-empty."""
        from utils import REQUIRED_MODEL_FILES

        assert REQUIRED_MODEL_FILES is not None
        assert isinstance(REQUIRED_MODEL_FILES, list)
        assert len(REQUIRED_MODEL_FILES) > 0

    def test_required_files_contain_essential_files(self):
        """REQUIRED_MODEL_FILES should contain model.bin and config.json."""
        from utils import REQUIRED_MODEL_FILES

        assert "model.bin" in REQUIRED_MODEL_FILES
        assert "config.json" in REQUIRED_MODEL_FILES
        assert "tokenizer.json" in REQUIRED_MODEL_FILES


class TestDownloadModelFiles:
    """Tests for download_model_files function."""

    def test_download_model_files_calls_hf_hub_download(self):
        """download_model_files should call hf_hub_download for each required file."""
        from utils import download_model_files, REQUIRED_MODEL_FILES

        with patch('utils.hf_hub_download') as mock_download, \
             patch('utils.HfFileSystem') as mock_fs:
            mock_download.return_value = "/mock/path/model.bin"
            mock_fs_instance = MagicMock()
            mock_fs_instance.info.return_value = {"size": 1000}
            mock_fs.return_value = mock_fs_instance

            download_model_files("tiny", "/tmp/cache")

            # Should be called for each required file
            assert mock_download.call_count == len(REQUIRED_MODEL_FILES)

    def test_download_model_files_progress_callback(self):
        """download_model_files should call progress callback."""
        from utils import download_model_files

        progress_calls = []

        def callback(status, message, progress):
            progress_calls.append((status, message, progress))

        with patch('utils.hf_hub_download') as mock_download, \
             patch('utils.HfFileSystem') as mock_fs:
            mock_download.return_value = "/mock/path/file"
            mock_fs_instance = MagicMock()
            mock_fs_instance.info.return_value = {"size": 1000}
            mock_fs.return_value = mock_fs_instance

            download_model_files("tiny", "/tmp/cache", callback)

            # Progress callback should have been called
            assert len(progress_calls) > 0
            # Should have a "checking" status at start
            assert any(call[0] == "checking" for call in progress_calls)

    def test_download_model_files_unknown_model_raises(self):
        """download_model_files should raise ValueError for unknown model."""
        from utils import download_model_files

        with pytest.raises(ValueError, match="Unknown model"):
            download_model_files("nonexistent-model-xyz")

    def test_download_model_files_returns_path(self):
        """download_model_files should return the model directory path."""
        from utils import download_model_files

        with patch('utils.hf_hub_download') as mock_download, \
             patch('utils.HfFileSystem') as mock_fs:
            mock_download.return_value = "/cache/models/snapshots/abc/model.bin"
            mock_fs_instance = MagicMock()
            mock_fs_instance.info.return_value = {"size": 1000}
            mock_fs.return_value = mock_fs_instance

            result = download_model_files("tiny", "/tmp/cache")

            assert result is not None
            assert isinstance(result, str)


class TestProgressCallback:
    """Tests for progress callback functionality in model loading."""

    async def test_get_whisper_model_calls_progress_callback(self):
        """get_whisper_model should call progress callback during loading."""
        from utils import get_whisper_model, model_cache

        model_cache.clear()
        progress_calls = []

        def callback(status, message, progress):
            progress_calls.append((status, message, progress))

        with patch('utils.WhisperModel') as mock_class, \
             patch('utils.is_model_cached', return_value=True):
            mock_class.return_value = MagicMock()

            await get_whisper_model("tiny", "cpu", callback)

            # Should have progress calls
            assert len(progress_calls) > 0

    async def test_get_whisper_model_sends_ready_status(self):
        """get_whisper_model should send 'ready' status when done."""
        from utils import get_whisper_model, model_cache

        model_cache.clear()
        progress_calls = []

        def callback(status, message, progress):
            progress_calls.append((status, message, progress))

        with patch('utils.WhisperModel') as mock_class, \
             patch('utils.is_model_cached', return_value=True):
            mock_class.return_value = MagicMock()

            await get_whisper_model("tiny", "cpu", callback)

            # Should have a "ready" status at the end
            assert any(call[0] == "ready" for call in progress_calls)

    async def test_get_whisper_model_cached_returns_immediately(self):
        """get_whisper_model should return immediately if model is in cache."""
        from utils import get_whisper_model, model_cache

        # Pre-populate cache
        mock_model = MagicMock()
        model_cache["tiny_cpu"] = mock_model

        progress_calls = []

        def callback(status, message, progress):
            progress_calls.append((status, message, progress))

        result = await get_whisper_model("tiny", "cpu", callback)

        assert result is mock_model
        # Should still call callback with ready status
        assert any(call[0] == "ready" for call in progress_calls)

        # Clean up
        model_cache.clear()
