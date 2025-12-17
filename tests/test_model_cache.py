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

        # Mock WhisperModel and is_model_cached
        mock_model = MagicMock()
        with patch('utils.WhisperModel', return_value=mock_model), \
             patch('utils.is_model_cached', return_value=False):
            result = await get_whisper_model("tiny", "cpu")
            assert result == mock_model

    async def test_get_whisper_model_caches_result(self):
        """Second call should return cached model, not load again."""
        from utils import get_whisper_model, model_cache

        model_cache.clear()

        mock_model = MagicMock()
        with patch('utils.WhisperModel', return_value=mock_model) as mock_class, \
             patch('utils.is_model_cached', return_value=False):
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
             patch('utils.is_model_cached', return_value=False):
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
             patch('utils.is_model_cached', return_value=False):
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
        """CPU should use int8, CUDA should use float16."""
        from utils import get_whisper_model, model_cache

        model_cache.clear()

        with patch('utils.WhisperModel') as mock_class, \
             patch('utils.is_model_cached', return_value=False):
            mock_class.return_value = MagicMock()

            # Test CPU compute type
            await get_whisper_model("tiny", "cpu")
            call_kwargs = mock_class.call_args
            assert call_kwargs.kwargs.get('compute_type') == "int8"

        model_cache.clear()

        with patch('utils.WhisperModel') as mock_class, \
             patch('utils.is_model_cached', return_value=False):
            mock_class.return_value = MagicMock()

            # Test CUDA compute type
            await get_whisper_model("tiny", "cuda")
            call_kwargs = mock_class.call_args
            assert call_kwargs.kwargs.get('compute_type') == "float16"

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
        """When model is not cached, local_files_only should be False."""
        from utils import get_whisper_model, model_cache

        model_cache.clear()

        with patch('utils.WhisperModel') as mock_class, \
             patch('utils.is_model_cached', return_value=False):
            mock_class.return_value = MagicMock()

            await get_whisper_model("tiny", "cpu")
            call_kwargs = mock_class.call_args
            assert call_kwargs.kwargs.get('local_files_only') is False
